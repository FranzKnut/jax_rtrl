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
    """Default parameters are the ones used in [1].

    :param image_output_size: Reconstructed image size
    :param in_channels: Number of channels of input image
    :param out_channels: Output channels of each conv layer in the encoder.
    :param latent_dimension: Input dimension for decoder
    :param temperature: Temperature parameter, None if it is to be learnt
    :param normalise: Should spatial features be normalised to [-1, 1]?
    """

    channels: list[int] = field(default_factory=lambda: [32, 32, 64, 64])
    temperature: float | None = None
    tanh_output: bool = True
    g_slow_factor: float = 0
    c_hid_dec: int = 16
    norm: str | None = None


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
    """Creates a Deep Spatial Autoencoder encoder
    :param out_channels: Output channels for each of the layers. The last output channel corresponds to half the
    size of the low-dimensional latent representation.
    :param temperature: Temperature for spatial soft argmax operation. See SpatialSoftArgmax.
    :param normalise: Normalisation of spatial features. See SpatialSoftArgmax.
    """

    out_channels: tuple
    temperature: float = None
    tanh_output: bool = False
    norm: str | None = None

    @nn.compact
    def __call__(self, x, train: bool = True):
        def norm(x):
            if self.norm == "batch":
                x = nn.BatchNorm()(x, use_running_average=not train)
            return x

        x = nn.Conv(features=self.out_channels[0], kernel_size=(7, 7))(x)
        x = nn.relu(norm(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=self.out_channels[1], kernel_size=(5, 5))(x)
        x = nn.relu(norm(x))
        x = nn.Conv(features=self.out_channels[2], kernel_size=(3, 3))(x)
        x = nn.relu(norm(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=self.out_channels[3], kernel_size=(3, 3))(x)
        x = nn.relu(norm(x))
        out = SpatialSoftArgmax(temperature=self.temperature, normalise=self.tanh_output)(x)
        return out


class LinearDecoder(nn.Module):
    """
    Creates a Linear Image decoder used in the Deep Spatial Autoencoder
    :param image_output_size: (height, width) of the output, grayscale image
    :param normalise: True if output in range [-1, 1], False for range [0, 1]
    """

    img_shape: tuple
    normalise: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=int(np.prod(self.img_shape)))(x)
        x = x.reshape(*[int(n) for n in self.img_shape[:2]])
        activ = nn.tanh if self.normalise else nn.sigmoid
        x = activ(x)
        return x


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
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.img_shape[-1], kernel_size=(5, 5), strides=2)(x)
        activ = nn.tanh if self.tanh_output else nn.sigmoid
        x = activ(x)
        return x


class DeepSpatialAutoencoder(nn.Module):
    """A deep spatial autoencoder."""

    image_output_size: tuple
    config: DSAEConfig = field(default_factory=DSAEConfig)

    def setup(self):
        self.encoder = DSAE_Encoder(out_channels=self.config.channels, temperature=self.config.temperature, tanh_output=self.config.tanh_output, norm=self.config.norm)
        self.decoder = ConvDecoder(img_shape=self.image_output_size, tanh_output=self.config.tanh_output)

    def encode(self, x, train: bool = True):
        """Encode given Image."""
        return self.encoder(x, train=train)

    def decode(self, latent):
        """Decode Image from latent vector."""
        return self.decoder(latent)

    def __call__(self, x, train: bool = True):
        spatial_features = self.encoder(x, train=train)
        c, _2 = spatial_features.shape
        return self.decoder(spatial_features.flatten()), spatial_features

    def dsae_loss(self, reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None, pixel_weights=None):
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
        pixel_diff = target - reconstructed
        if pixel_weights is not None:
            pixel_diff *= pixel_weights
        mse_loss = jnp.mean(pixel_diff**2)
        g_slow_contrib = 0.0
        loss_info = {"reconstruction_loss": mse_loss}
        if ft_minus1 is not None:
            g_slow_contrib = jnp.mean((ft_plus1 - ft - (ft - ft_minus1)) ** 2)
            loss_info["g_slow"] = g_slow_contrib * self.config.g_slow_factor
        return mse_loss + g_slow_contrib, loss_info
