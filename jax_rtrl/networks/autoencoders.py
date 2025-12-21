"""Convolutional neural network autoencoders built with flax."""

from dataclasses import dataclass, field
from simple_parsing import Serializable
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from typing import Literal
from flax import linen as nn
from jax.nn import softmax
from jax_rtrl.models.cells.ctrnn import CTRNNCell
from jax_rtrl.util.checkpointing import restore_config, restore_params

conv_presets = {
    "legacy_small": [(16, (3, 3)), (16, (3, 3)), (16, (3, 3))],
    "small": [
        (16, (3, 3), 2),
        (16, (3, 3), 2),
        (16, (3, 3), 2),
    ],
}


@dataclass
class ConvLayerConfig:
    features: int = 16
    kernel_size: tuple[int, ...] = (3, 3)
    strides: tuple[int, ...] | int = 1
    pooling: Literal["avg", "max", None] = None
    pool_size: tuple[int, int] | int = (2, 2)


@dataclass
class ConvConfig(Serializable):
    preset: str | None = "small"
    layers: list[tuple[int, tuple[int, ...], int]] | None = None
    latent_size: int = 16

    def __post_init__(self):
        assert self.preset is not None or self.layers is not None, (
            "Either preset or layers must be defined."
        )

    def get_layers(self) -> list[ConvLayerConfig]:
        """Get list of ConvLayerParams."""
        layers = self.layers
        if layers is None:
            layers = conv_presets[self.preset]

        return [ConvLayerConfig(*c) for c in conv_presets[self.preset]]

    def get_strides(self) -> int:
        """Get total stride along given axis."""
        all_strides = [np.array(_l.strides) for _l in self.get_layers()]
        num_dims = max(1, max([s.ndim for s in all_strides]))
        strides = np.ones(num_dims, dtype=int)
        for l_conf in self.get_layers():
            strides *= l_conf.strides
        return strides


class ConvEncoder(nn.Module):
    """2D-Convolutional Encoder.

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html"""

    latent_size: int = 16
    config: ConvConfig = field(default_factory=ConvConfig)

    @nn.compact
    def __call__(self, x):
        """Encode given Image."""
        # Encode observation using CNN
        for l_conf in self.config.get_layers():
            x = nn.Conv(
                features=l_conf.features,
                kernel_size=l_conf.kernel_size,
                strides=l_conf.strides,
            )(x)
            x = nn.gelu(x)  # Apply activation function
            # Optionally apply Pooling
            if l_conf.pooling:
                _pool_fn = nn.max_pool if l_conf.pooling == "max" else nn.avg_pool
                x = _pool_fn(x, window_shape=l_conf.pool_size)

        x = x.flatten()  # Image grid to single feature vector
        return nn.Dense(features=self.latent_size)(x)


class ConvDecoder(nn.Module):
    """2D-Convolutional Decoder.

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html
    """

    img_shape: tuple[int]
    tanh_output: bool = False  # sigmoid otherwise
    config: ConvConfig = field(default_factory=ConvConfig)

    @nn.compact
    def __call__(self, x):
        """Decode Image from latent vector."""
        # initial img shape depends on number of layers
        xy_shape = np.array(self.img_shape[:2]) / self.config.get_strides()
        if any(xy_shape != xy_shape.astype(int)):
            raise ValueError(
                "The img_shape must be divisible by sum of strides of the decoding layers."
            )

        x = nn.Dense(features=int(xy_shape.prod()))(x)
        x = nn.relu(x)
        x = x.reshape(*[int(n) for n in xy_shape], -1)
        # Decode observation using transposed CNN
        for l_conf in self.config.get_layers():
            x = nn.ConvTranspose(
                features=l_conf.features,
                kernel_size=l_conf.kernel_size,
                strides=l_conf.strides,
            )(x)
            x = nn.gelu(x)  # Apply activation function
            # Optionally apply Pooling
            if l_conf.pooling:
                _pool_fn = nn.max_pool if l_conf.pooling == "max" else nn.avg_pool
                x = _pool_fn(x, window_shape=l_conf.pool_size)

        x = nn.ConvTranspose(features=self.img_shape[-1], kernel_size=(3, 3))(x)
        x = nn.tanh(x) if self.tanh_output else nn.sigmoid(x)
        return x


@dataclass
class AutoencoderConfig(Serializable):
    latent_size: int = 32
    encoder_cfg: ConvConfig = field(default_factory=ConvConfig)
    decoder_cfg: ConvConfig = field(default_factory=ConvConfig)


class Autoencoder(nn.Module):
    """Deterministic 2D-Autoencoder for dimension reduction."""

    img_shape: tuple[int]
    config: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    tanh_output: bool = False

    def setup(self) -> None:
        """Initialize submodules."""
        super().setup()
        self.enc = ConvEncoder(self.config.latent_size, self.config.encoder_cfg)
        self.dec = ConvDecoder(
            self.img_shape,
            tanh_output=self.tanh_output,
            config=self.config.decoder_cfg,
        )

    def encode(self, x, *_):
        """Encode given Image."""
        return self.enc(x)

    def decode(self, latent):
        """Decode Image from latent vector."""
        return self.dec(latent)

    def __call__(self, x, *_):
        """Encode then decode. Returns prediction and latent vector."""
        latent = self.enc(x)
        pred = self.dec(latent)
        return pred, latent


class Autoencoder_RNN(nn.Module):
    """Deterministic 2D-Autoencoder that also contains an RNN."""

    img_shape: tuple[int]
    latent_size: int = 128
    num_units: int = 32
    use_cnn: bool = True

    @nn.compact
    def __call__(self, x, a, carry=None):
        """Take in an image, action and hidden state to predict next image."""
        # Encode observation using CNN
        obs_size = x.shape[-1]

        if self.use_cnn:
            x = ConvEncoder()(x)

        # Action FC
        a = nn.Dense(features=self.latent_size)(a)
        a = nn.tanh(a)

        # Encoded observation FC
        e = nn.Dense(features=self.latent_size)(x)
        x = nn.tanh(e)

        # Concatenate
        x = jnp.concatenate([x, a], axis=-1)

        # Step LSTM
        rnn = CTRNNCell(self.num_units)
        if carry is None:
            carry = rnn.initialize_carry(jax.random.key(0), x.shape)
        carry, x = rnn(carry, x)

        # Concatenate last LSTM state with latent
        x = jnp.concatenate([x, e], axis=-1)
        latent = nn.Dense(features=self.latent_size)(x)
        if self.use_cnn:
            pred = ConvDecoder(self.img_shape)(latent)
        else:
            pred = nn.Dense(features=obs_size)(latent)
        return carry, pred, latent


@dataclass
class DSAEConfig:
    """Deep Spatial Auto-Encoder using flax.

    Taken from https://github.com/gorosgobe/dsae-torch/tree/master

    References:
        [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
        Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
        Available at: https://arxiv.org/pdf/1509.06113.pdf
        [2]: https://github.com/tensorflow/tensorflow/issues/6271
    """

    c_hid_enc: int = 16
    c_hid_dec: int = 16
    latent_size: int = 20
    temperature: float | None = None
    tanh_output: bool = True
    norm: str | None = "batch"
    decoder: str = "Conv"
    twin_bottleneck: bool = True
    head_type: str = "Small"


def get_image_coordinates(h, w, normalise):
    x_range = jnp.arange(w, dtype=jnp.float32)
    y_range = jnp.arange(h, dtype=jnp.float32)
    if normalise:
        x_range = (x_range / (w - 1)) * 2 - 1
        y_range = (y_range / (h - 1)) * 2 - 1
    image_x = jnp.tile(x_range[None, :], (h, 1))
    image_y = jnp.tile(y_range[:, None], (1, w))
    return jnp.stack((image_x, image_y), axis=-1)


class SpatialSoftArgmax(nn.Module):
    """Applies a spatial soft argmax over the input images.
    :param temperature: The temperature parameter (float). If None, it is learnt.
    :param normalise: Should spatial features be normalised to range [-1, 1]?
    """

    temperature: float = None
    normalise: bool = False

    @nn.compact
    def __call__(self, x):
        """Apply Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (H, W, C)
        :return: Spatial features (one point per channel), of size (C, 2)
        """
        h, w, c = x.shape[-3:]
        # Reshape to (C, H, W)
        x = x.transpose((2, 0, 1))
        _temperature = (
            self.param("temperature", lambda _: jnp.ones(1))
            if self.temperature is None
            else jnp.array([self.temperature])
        )
        spatial_softmax_per_map = softmax(x.reshape(c, h * w) / _temperature, axis=-1)
        spatial_softmax = spatial_softmax_per_map.reshape(c, h, w).squeeze()
        spatial_softmax = spatial_softmax.transpose((1, 2, 0))

        # calculate image coordinate maps, size (H, W, 2)
        image_coordinates = get_image_coordinates(h, w, normalise=self.normalise)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax[..., None]
        image_coordinates = image_coordinates[:, :, None]
        out = jnp.sum(expanded_spatial_softmax * image_coordinates, axis=(0, 1))
        return out


class DSAE_Encoder(nn.Module):
    """Creates a Deep Spatial Autoencoder encoder"""

    head_type: str = "Small"
    c_hid: int = 32
    temperature: float = None
    tanh_output: bool = False
    norm: str | None = None
    latent_size: int = 64
    twin_bottleneck: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        def norm(x):
            if self.norm == "batch":
                x = nn.BatchNorm()(x, use_running_average=not train)
            return x

        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.gelu(norm(x))
        if self.head_type == "Large":
            x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
            x = nn.gelu(norm(x))
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.gelu(norm(x))
        if self.head_type == "Large":
            x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3))(x)
            x = nn.gelu(norm(x))
        x = nn.Conv(features=4 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.gelu(norm(x))
        x = nn.Conv(features=self.latent_size, kernel_size=(3, 3))(x)
        out = SpatialSoftArgmax(
            temperature=self.temperature, normalise=self.tanh_output
        )(x)
        if self.twin_bottleneck:
            vec_out = nn.tanh(nn.Dense(self.latent_size)(x.flatten()))
            return out, vec_out
        else:
            return out


class LinearDecoder(nn.Module):
    """
    Creates a Linear Image decoder used in the Deep Spatial Autoencoder
    :param image_output_size: (height, width) of the output, grayscale image
    :param normalise: True if output in range [-1, 1], False for range [0, 1]
    """

    img_shape: tuple
    tanh_output: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=int(np.prod(self.img_shape)))(x)
        x = x.reshape(*[int(n) for n in self.img_shape[:2]])
        return nn.tanh(x) if self.tanh_output else x


class SimpleConvDecoder(nn.Module):
    """2D-Convolutional Decoder.

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html"""

    img_shape: tuple[int]
    c_hid: int = 16
    tanh_output: bool = True

    @nn.compact
    def __call__(self, x):
        """Decode Image from latent vector."""
        xy_shape = (
            np.array(self.img_shape[:2]) / 4
        )  # initial img shape depends on number of layers
        if any(xy_shape != xy_shape.astype(int)):
            raise ValueError("The img x- and y-shapes must be divisible by 4.")

        x = nn.Dense(features=int(xy_shape.prod()) * self.c_hid * 2)(x)
        x = nn.relu(x)
        x = x.reshape(*[int(n) for n in xy_shape], -1)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=self.img_shape[-1], kernel_size=(3, 3), strides=2
        )(x)
        return nn.tanh(x) if self.tanh_output else x


class DeepSpatialAutoencoder(nn.Module):
    """A deep spatial autoencoder."""

    image_output_size: tuple
    config: DSAEConfig = field(default_factory=DSAEConfig)

    def setup(self):
        self.encoder = DSAE_Encoder(
            c_hid=self.config.c_hid_enc,
            temperature=self.config.temperature,
            tanh_output=self.config.tanh_output,
            norm=self.config.norm,
            latent_size=self.config.latent_size,
            twin_bottleneck=self.config.twin_bottleneck,
            head_type=self.config.head_type,
        )
        if self.config.decoder == "SimpleConv":
            self.decoder = SimpleConvDecoder(
                img_shape=self.image_output_size,
                tanh_output=self.config.tanh_output,
                c_hid=self.config.c_hid_dec,
            )
        elif self.config.decoder == "Linear":
            self.decoder = LinearDecoder(
                img_shape=self.image_output_size, tanh_output=self.config.tanh_output
            )
        elif self.config.decoder == "Conv":
            self.decoder = ConvDecoder(
                img_shape=self.image_output_size,
                tanh_output=self.config.tanh_output,
                c_hid=self.config.c_hid_dec,
            )

    def encode(self, x, train: bool = True):
        """Encode given Image."""
        return self.encoder(x, train=train)

    def decode(self, latent):
        """Decode Image from latent vector."""
        return self.decoder(latent)

    def get_full_latent_size(self):
        # 2 * for coordinates of spatial + 1 * vector latent
        return 3 * self.config.latent_size

    def __call__(self, x, train: bool = True):
        enc_out = self.encoder(x, train=train)
        if self.config.twin_bottleneck:
            spatial_features, vec_features = enc_out
            features = jnp.concatenate(
                [spatial_features.flatten(), vec_features], axis=-1
            )
        else:
            features = enc_out.flatten()
        return self.decoder(features), features


def restore_cnn_from_ckpt(
    ckpt_path: str, restored_config: ConvConfig | AutoencoderConfig = None, **inputs
) -> ConvEncoder | Autoencoder:
    """Restore a CNN from a checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file.
    restored_config : ConvConfig
        Configuration for the restored CNN. If None, the configuration will be loaded from the checkpoint.
    **inputs : dict
        Inputs required to initialize the policy module.

    Returns
    -------
    ConvEncoder | Autoencoder
        The restored CNN module.
    """
    if restored_config is None:
        restored_config = restore_config(ckpt_path)
        # Try to infer config class
        if "encoder_params" in restored_config:
            restored_config = AutoencoderConfig.from_dict(restored_config)
        else:
            restored_config = ConvConfig.from_dict(restored_config)
    if restored_config.__class__ == AutoencoderConfig:
        cnn = Autoencoder(
            img_shape=inputs.get("x").shape[1:],
            config=restored_config,
        )
    else:
        cnn = ConvEncoder(config=restored_config)
    target = cnn.lazy_init(jax.random.PRNGKey(0), **inputs)
    variables = restore_params(ckpt_path, tree=target)
    return cnn.bind(variables)


if __name__ == "__main__":
    # Test encoder implementation
    model = Autoencoder_RNN()

    # Random key for initialization
    rng = jrandom.PRNGKey(0)
    # Example images as input
    imgs = jrandom.normal(rng, (32, 32, 1))
    # Example actions as input
    a = jrandom.normal(rng, (3,))
    print("Input shape: ", imgs.shape)

    # Initialize parameters of encoder with random key and images
    out, params = model.init_with_output(rng, imgs, a)
    print("Output shapes: ")
    for k in range(len(out)):
        try:
            print(k, out[k].shape)
        except AttributeError:
            # Assuming its a tuple
            print(k, [o.shape for o in out[k]])
