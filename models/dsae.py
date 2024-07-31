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

    channels: tuple = (64, 32, 16)
    temperature: float = None
    normalise: bool = True


def get_image_coordinates(h, w, normalise):
    x_range = jnp.arange(w, dtype=jnp.float32)
    y_range = jnp.arange(h, dtype=jnp.float32)
    if normalise:
        x_range = (x_range / (w - 1)) * 2 - 1
        y_range = (y_range / (h - 1)) * 2 - 1
    image_x = jnp.tile(x_range[None, :], (h, 1))
    image_y = jnp.tile(y_range[:, None], (1, w))
    return image_x, image_y


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
        _temperature = self.param("temperature", lambda _: jnp.ones(1)) if self.temperature is None else jnp.array([self.temperature])
        spatial_softmax_per_map = softmax(x.reshape(h * w, -1).transpose() / _temperature, axis=0)
        spatial_softmax = spatial_softmax_per_map.reshape(-1, c, h, w).squeeze()

        # calculate image coordinate maps
        image_x, image_y = get_image_coordinates(h, w, normalise=self.normalise)
        # size (H, W, 2)
        image_coordinates = jnp.stack((image_x, image_y), axis=-1)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax[..., None]
        image_coordinates = image_coordinates[None, None, ...]
        out = jnp.sum(expanded_spatial_softmax * image_coordinates, axis=(2, 3))
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
    normalise: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=self.out_channels[0], kernel_size=(7, 7), strides=(2, 2))(x)
        x = nn.relu(nn.BatchNorm()(x, use_running_average=not train))
        x = nn.Conv(features=self.out_channels[1], kernel_size=(5, 5))(x)
        x = nn.relu(nn.BatchNorm()(x, use_running_average=not train))
        x = nn.Conv(features=self.out_channels[2], kernel_size=(5, 5))(x)
        x = nn.relu(nn.BatchNorm()(x, use_running_average=not train))
        out = SpatialSoftArgmax(temperature=self.temperature, normalise=self.normalise)(x)
        return out


class DSAE_Decoder(nn.Module):
    """
    Creates a Deep Spatial Autoencoder decoder
    :param image_output_size: (height, width) of the output, grayscale image
    :param normalise: True if output in range [-1, 1], False for range [0, 1]
    """

    image_output_size: tuple
    normalise: bool = True
    c_hid: int = 8

    @nn.compact
    def __call__(self, x):
        xy_shape = np.array(self.image_output_size[:2]) / (2 * 2)  # initial img shape depends on number of layers
        if any(xy_shape != xy_shape.astype(int)):
            raise ValueError("The img x- and y-shapes must be divisible by 4.")

        x = nn.Dense(features=int(xy_shape.prod()) * self.c_hid)(x)
        x = nn.relu(x)
        x = x.reshape(*[int(n) for n in xy_shape], -1)
        x = nn.ConvTranspose(features=self.c_hid // 2, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.image_output_size[-1], kernel_size=(3, 3), strides=2)(x)
        activ = nn.tanh if self.normalise else nn.sigmoid
        x = activ(x)
        return x


class DeepSpatialAutoencoder(nn.Module):
    """A deep spatial autoencoder."""

    image_output_size: tuple
    config: DSAEConfig = field(default_factory=DSAEConfig)

    def setup(self):
        self.encoder = DSAE_Encoder(
            out_channels=self.config.channels,
            temperature=self.config.temperature,
            normalise=self.config.normalise,
        )
        self.decoder = DSAE_Decoder(image_output_size=self.image_output_size, normalise=self.config.normalise)

    def encode(self, x, train: bool = True):
        """Encode given Image."""
        return self.encoder(x, train=train)

    def decode(self, latent):
        """Decode Image from latent vector."""
        return self.decoder(latent)

    def __call__(self, x, train: bool = True):
        spatial_features = self.encoder(x, train=train)
        n, c, _2 = spatial_features.shape
        return self.decoder(spatial_features.reshape(n, c * 2)), spatial_features


def dsae_loss(reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None):
    """Compute Loss for deep spatial autoencoder.
    For the start of a trajectory, where ft_minus1 = ft, simply pass in ft_minus1=ft, ft=ft
    For the end of a trajectory, where ft_plus1 = ft, simply pass in ft=ft, ft_plus1=ft
    :param reconstructed: Reconstructed, grayscale image
    :param target: Target, grayscale image
    :param ft_minus1: Features produced by the encoder for the previous image in the trajectory to the target one
    :param ft: Features produced by the encoder for the target image
    :param ft_plus1: Features produced by the encoder for the next image in the trajectory to the target one
    :return: A tuple (mse, g_slow) where mse = the MSE reconstruction loss and g_slow = g_slow contribution term ([1])
    """
    mse_loss = jnp.sum((reconstructed - target) ** 2)
    g_slow_contrib = 0.0
    loss_info = {"reconstruction_loss": mse_loss}
    if ft_minus1 is not None:
        g_slow_contrib = jnp.sum((ft_plus1 - ft - (ft - ft_minus1)) ** 2)
        loss_info["g_slow"] = g_slow_contrib
    return mse_loss + g_slow_contrib, loss_info
