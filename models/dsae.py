"""Deep Spatial Auto-Encoder using flax.

Taken from https://github.com/gorosgobe/dsae-torch/tree/master

References:
    [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    Available at: https://arxiv.org/pdf/1509.06113.pdf
    [2]: https://github.com/tensorflow/tensorflow/issues/6271
"""

from dataclasses import dataclass, field
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softmax
import numpy as np

from jax_rtrl.models.neural_networks import ConvDecoder


@dataclass
class DSAEConfig:
    c_hid_enc: int = 16
    c_hid_dec: int = 16
    latent_size: int = 64
    temperature: float | None = None
    tanh_output: bool = True
    g_slow_factor: float = 1e-6
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
        _temperature = self.param("temperature", lambda _: jnp.ones(1)) if self.temperature is None else jnp.array([self.temperature])
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
        out = SpatialSoftArgmax(temperature=self.temperature, normalise=self.tanh_output)(x)
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
        xy_shape = np.array(self.img_shape[:2]) / 4  # initial img shape depends on number of layers
        if any(xy_shape != xy_shape.astype(int)):
            raise ValueError("The img x- and y-shapes must be divisible by 4.")

        x = nn.Dense(features=int(xy_shape.prod()) * self.c_hid * 2)(x)
        x = nn.relu(x)
        x = x.reshape(*[int(n) for n in xy_shape], -1)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.img_shape[-1], kernel_size=(3, 3), strides=2)(x)
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
            self.decoder = SimpleConvDecoder(img_shape=self.image_output_size, tanh_output=self.config.tanh_output, c_hid=self.config.c_hid_dec)
        elif self.config.decoder == "Linear":
            self.decoder = LinearDecoder(img_shape=self.image_output_size, tanh_output=self.config.tanh_output)
        elif self.config.decoder == "Conv":
            self.decoder = ConvDecoder(img_shape=self.image_output_size, tanh_output=self.config.tanh_output, c_hid=self.config.c_hid_dec)

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
            features = jnp.concatenate([spatial_features.flatten(), vec_features], axis=-1)
        else:
            features = enc_out.flatten()
        return self.decoder(features), features

    def dsae_g_slow_loss(self, ft_minus1=None, ft=None, ft_plus1=None):
        """Compute Loss for deep spatial autoencoder.
        For the start of a trajectory, where ft_minus1 = ft, simply pass in ft_minus1=ft, ft=ft
        For the end of a trajectory, where ft_plus1 = ft, simply pass in ft=ft, ft_plus1=ft
        :param reconstructed: Reconstructed, grayscale image
        :param target: Target, grayscale image
        :param ft_minus1: Features produced by the encoder for the previous image in the trajectory to the target one
        :param ft: Features produced by the encoder for the target image
        :param ft_plus1: Features produced by the encoder for the next image in the trajectory to the target one
        :param pixel_weights: An array with same dimensions as the image. For weighting each pixel differently in the loss
        :return: A tuple (mse, g_slow) where mse = the MSE reconstruction loss and g_slow = g_slow contribution term ([1])
        """
        g_slow_contrib = 0.0
        if ft_minus1 is not None:
            g_slow_contrib = jnp.mean((ft_plus1 - ft - (ft - ft_minus1)) ** 2)
        return g_slow_contrib * self.config.g_slow_factor
