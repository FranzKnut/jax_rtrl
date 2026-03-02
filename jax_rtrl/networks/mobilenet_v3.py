#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2022 Nobuo Tsukamoto
This software is released under the MIT License.
See the LICENSE file in the project root for more information.
"""

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import jax.nn as jnn
from flax import linen as nn

ModuleDef = Any

""" MobileNet V3 models for Flax.

    Reference:
        Searching for MobileNetV3
        https://arxiv.org/abs/1905.02244

        https://github.com/keras-team/keras-applications/blob/06fbeb0f16e1304f239b2296578d1c50b15a983a/keras_applications/mobilenet_v3.py
"""


def _make_divisible(v, divisor=8, min_value=None):
    """
    Reference:
        https://github.com/keras-team/keras/blob/v2.8.0/keras/applications/mobilenet_v2.py#L505
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SeBlock(nn.Module):
    filters: int
    se_ratio: int
    conv: ModuleDef

    @nn.compact
    def __call__(self, x):
        inputs = x
        filters = _make_divisible(self.filters * self.se_ratio)
        in_filters = x.shape[-1]

        x = jnp.mean(x, axis=(1, 2), keepdims=True)
        x = x.reshape(-1, 1, 1, in_filters)

        x = self.conv(filters, kernel_size=(1, 1), padding="same")(x)
        x = nn.relu(x)

        x = self.conv(self.filters, kernel_size=(1, 1), padding="same")(x)
        x = jnn.hard_sigmoid(x)

        return jnp.multiply(inputs, x)


class InvertedResBlockMobileNetV3(nn.Module):
    """Inverted ResNet block for MobileNet V3."""

    expansion: float
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    se_ratio: float
    block_id: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable = None
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        inputs = x
        in_filters = x.shape[-1]

        se_bolock = partial(
            SeBlock,
            conv=self.conv,
        )

        prefix = "block_{:02}_".format(self.block_id)

        if self.block_id != 1:
            # Expand
            x = self.conv(
                features=int(in_filters * self.expansion),
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                name=prefix + "expand",
            )(x)
            x = self.norm(name=prefix + "expand_bn")(x)
            x = self.act(x)

        # Depthwise
        dw_filters = x.shape[-1]
        x = self.conv(
            features=dw_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="SAME",
            feature_group_count=dw_filters,
            name=prefix + "depthwise",
        )(x)
        x = self.norm(name=prefix + "depthwise_bn")(x)
        x = self.act(x)

        if self.se_ratio:
            x = se_bolock(
                filters=_make_divisible(in_filters * self.expansion),
                se_ratio=self.se_ratio,
            )(x)

        # Project
        x = self.conv(
            features=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            name=prefix + "project",
        )(x)
        x = self.norm(name=prefix + "project_bn")(x)

        if in_filters == self.filters and self.strides == (1, 1):
            x = x + inputs

        return x


class MobileNetV3Backbone(nn.Module):
    """MobileNet V3 Backbone."""

    alpha: float
    layers: Dict
    dtype: Any = jnp.float32
    relu: Callable = nn.relu
    h_swish: Callable = jnn.hard_swish
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm

    @nn.compact
    def __call__(self, x):
        inverted_res_block = partial(InvertedResBlockMobileNetV3, conv=self.conv, norm=self.norm)

        first_block_filters = _make_divisible(16 * self.alpha, 8)

        x = self.conv(
            first_block_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            name="conv_init",
        )(x)
        x = self.norm()(x)
        x = self.h_swish(x)

        for block_id, layer in self.layers.items():
            x = inverted_res_block(
                expansion=layer["exp"],
                filters=layer["filters"],
                kernel_size=layer["kernel"],
                strides=layer["strides"],
                se_ratio=layer["se_ratio"],
                act=self.h_swish if layer["h_swish"] else self.relu,
                block_id=block_id,
            )(x)

        if self.alpha > 1.0:
            filters = _make_divisible(x.shape[-1] * 6 * self.alpha, 8)
        else:
            filters = _make_divisible(x.shape[-1] * 6, 8)

        x = self.conv(filters, kernel_size=(1, 1), name="conv_1")(x)
        x = self.norm()(x)
        x = self.h_swish(x)

        return jnp.asarray(x, self.dtype)


class MobileNetV3(nn.Module):
    """MobileNet V3."""

    alpha: float
    num_classes: int
    layers: Dict
    last_block_filters: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
        )
        backbone = partial(
            MobileNetV3Backbone,
            alpha=self.alpha,
            layers=self.layers,
            dtype=self.dtype,
            conv=conv,
            norm=norm,
        )

        x = backbone()(x)

        x = jnp.mean(x, axis=(1, 2), keepdims=True)

        x = conv(self.last_block_filters, kernel_size=(1, 1), name="conv_2")(x)
        x = jnn.hard_swish(x)

        x = conv(self.num_classes, kernel_size=(1, 1), name="conv_3")(x)
        x = x.reshape((x.shape[0], -1))  # flatten

        return jnp.asarray(x, self.dtype)


# fmt: off
Large = {
    "1": {"exp": 1, "filters": 16, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    "2": {"exp": 4, "filters": 24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": False},
    "3": {"exp": 3, "filters": 24, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    "4": {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": False},
    "5": {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": False},
    "6": {"exp": 3, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": False},
    "7": {"exp": 6, "filters": 80, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": True},
    "8": {"exp": 2.5, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    "9": {"exp": 2.3, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    "10": {"exp": 2.3, "filters": 80, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": True},
    "11": {"exp": 6, "filters": 112, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "12": {"exp": 6, "filters": 112, "kernel": (3, 3), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "13": {"exp": 6, "filters": 160, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    "14": {"exp": 6, "filters": 160, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "15": {"exp": 6, "filters": 160, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
}

Small = {
    "1": {"exp": 1, "filters": 16, "kernel": (3, 3), "strides": (2, 2), "se_ratio": 0.25, "h_swish": False},
    "2": {"exp": 4.5, "filters": 24, "kernel": (3, 3), "strides": (2, 2), "se_ratio": None, "h_swish": False},
    "3": {"exp": 88. / 24, "filters": 24, "kernel": (3, 3), "strides": (1, 1), "se_ratio": None, "h_swish": False},
    "4": {"exp": 4, "filters": 40, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    "5": {"exp": 6, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "6": {"exp": 6, "filters": 40, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "7": {"exp": 3, "filters": 48, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "8": {"exp": 3, "filters": 48, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "9": {"exp": 6, "filters": 96, "kernel": (5, 5), "strides": (2, 2), "se_ratio": 0.25, "h_swish": True},
    "10": {"exp": 6, "filters": 96, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
    "11": {"exp": 6, "filters": 96, "kernel": (5, 5), "strides": (1, 1), "se_ratio": 0.25, "h_swish": True},
}
# fmt: on
