"""Deep Spatial Auto-Encoder using flax.

Taken from https://github.com/gorosgobe/dsae-torch/tree/master"""

from dataclasses import dataclass
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softmax


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
    temperature: float = None
    normalise: bool = False

    def setup(self):
        self.temperature = self.param(
            "temperature", lambda rng: jnp.ones(1) if self.temperature is None else jnp.array([self.temperature])
        )

    def __call__(self, x):
        n, c, h, w = x.shape
        spatial_softmax_per_map = softmax(x.reshape(n * c, h * w) / self.temperature, axis=-1)
        spatial_softmax = spatial_softmax_per_map.reshape(n, c, h, w)

        image_x, image_y = get_image_coordinates(h, w, normalise=self.normalise)
        image_coordinates = jnp.stack((image_x, image_y), axis=-1)

        expanded_spatial_softmax = spatial_softmax[..., None]
        image_coordinates = image_coordinates[None, None, ...]
        out = jnp.sum(expanded_spatial_softmax * image_coordinates, axis=(2, 3))
        return out


class DSAE_Encoder(nn.Module):
    in_channels: int
    out_channels: tuple
    temperature: float = None
    normalise: bool = False

    def setup(self, train: bool = True):
        self.conv1 = nn.Conv(features=self.out_channels[0], kernel_size=(7, 7), strides=(2, 2))
        self.batch_norm1 = nn.BatchNorm(use_running_average=not train)
        self.conv2 = nn.Conv(features=self.out_channels[1], kernel_size=(5, 5))
        self.batch_norm2 = nn.BatchNorm(use_running_average=not train)
        self.conv3 = nn.Conv(features=self.out_channels[2], kernel_size=(5, 5))
        self.batch_norm3 = nn.BatchNorm(use_running_average=not train)
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=self.temperature, normalise=self.normalise)

    def __call__(self, x):
        out_conv1 = nn.relu(self.batch_norm1(self.conv1(x)))
        out_conv2 = nn.relu(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = nn.relu(self.batch_norm3(self.conv3(out_conv2)))
        out = self.spatial_soft_argmax(out_conv3)
        return out


class DSAE_Decoder(nn.Module):
    image_output_size: tuple
    normalise: bool = True

    def setup(self):
        self.channels, self.height, self.width = self.image_output_size
        self.decoder = nn.Dense(features=self.channels * self.height * self.width)
        self.activ = nn.tanh if self.normalise else nn.sigmoid

    def __call__(self, x):
        out = self.activ(self.decoder(x))
        out = out.reshape(-1, self.channels, self.height, self.width)
        return out

@dataclass
class DSAEConfig:
    channels: tuple = (64, 32, 16)
    latent_dimension: int = 32
    temperature: float = None
    normalise: bool = False
    
class DeepSpatialAutoencoder(nn.Module):
    image_output_size: tuple = (60, 60)
    in_channels: int = 3
    channels: tuple = (64, 32, 16)
    latent_dimension: int = 32
    temperature: float = None
    normalise: bool = False

    def setup(self):
        if self.channels[-1] * 2 != self.latent_dimension:
            raise ValueError("Spatial SoftArgmax produces a location (x,y) per feature map!")
        self.encoder = DSAE_Encoder(
            in_channels=self.in_channels,
            out_channels=self.channels,
            temperature=self.temperature,
            normalise=self.normalise,
        )
        self.decoder = DSAE_Decoder(image_output_size=self.image_output_size)

    def __call__(self, x):
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.shape
        return self.decoder(spatial_features.reshape(n, c * 2))


def dsae_loss(reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None):
    mse_loss = jnp.sum((reconstructed - target) ** 2)
    g_slow_contrib = 0.0
    if ft_minus1 is not None:
        g_slow_contrib = jnp.sum((ft_plus1 - ft - (ft - ft_minus1)) ** 2)
    return mse_loss, g_slow_contrib
