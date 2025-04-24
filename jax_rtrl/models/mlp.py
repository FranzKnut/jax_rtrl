"""Neural networks built with flax."""

from dataclasses import field
from typing import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class FADense(nn.Dense):
    """Dense Layer with feedback alignment."""

    f_align: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        if self.f_align:
            B = self.variable(
                "falign",
                "B",
                self.kernel_init,
                self.make_rng() if self.has_rng("params") else None,
                (jnp.shape(x)[-1], self.features),
                self.param_dtype,
            ).value
        else:
            B = self.param(
                "kernel",
                self.kernel_init,
                (jnp.shape(x)[-1], self.features),
                self.param_dtype,
            )

        def f(mdl, x, _B):
            return nn.Dense.__call__(mdl, x)

        def fwd(mdl, x, _B):
            """Forward pass with tmp for backward pass."""
            return nn.Dense.__call__(mdl, x), (x, _B)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _B = tmp
            grads = {"params": {"kernel": jnp.einsum("...X,...Y->YX", y_bar, _x)}}
            if self.use_bias:
                grads["params"]["bias"] = jnp.einsum("...X->X", y_bar)
            x_grad = jnp.einsum("YX,...X->...Y", _B, y_bar)
            return (grads, x_grad, jnp.zeros_like(_B))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)

        return fa_grad(self, x, B)


class FAAffine(nn.Module):
    """Affine Layer with feedback alignment."""

    features: int
    f_align: bool = True
    offset: int = 0

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        a = self.param("a", nn.initializers.normal(), (self.features,))
        b = self.param("b", nn.initializers.zeros, (self.features,))

        def s(x):
            return x[..., self.offset : self.features + self.offset]

        def f(mdl, x, a, b):
            return a * s(x) + b

        def fwd(mdl, x, a, b):
            """Forward pass with tmp for backward pass."""
            return a * s(x) + b, (x, a)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(res, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _a = res
            grads = {"params": {"a": s(_x) * y_bar, "b": y_bar}}
            x_bar = jnp.zeros_like(_x)
            x_bar = x_bar.at[..., self.offset : self.features + self.offset].set(
                y_bar if not self.f_align else y_bar * _a
            )
            return (grads, x_bar, jnp.zeros_like(a), jnp.zeros_like(b))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return fa_grad(self, x, a, b)


class MLP(nn.Module):
    """MLP built with Flax.

    activation_fn is applied after every layer except the last one.
    If f_align is true, each layer uses feedback alignment instead of backpropagation.
    """

    layers: list
    activation_fn: Callable = jax.nn.relu
    f_align: bool = False

    @nn.compact
    def __call__(self, x):
        """Call MLP."""
        for size in self.layers[:-1]:
            x = self.activation_fn(FADense(size, f_align=self.f_align)(x))
        x = FADense(self.layers[-1], f_align=self.f_align)(x)
        return x


class RBFLayer(nn.Module):
    """Gaussian Radial Basis Function Layer."""

    output_size: int
    c_initializer: nn.initializers.Initializer = nn.initializers.normal(1)

    @nn.compact
    def __call__(self, x):
        """Compute the distance to centers."""
        c = self.param("centers", self.c_initializer, (self.output_size, x.shape[-1]))
        beta = self.param("beta", nn.initializers.ones_init(), (self.output_size, 1))
        x = x.reshape(x.shape[:-1] + (1, x.shape[-1]))
        z = jnp.exp(-beta * (x - c) ** 2)
        return jnp.sum(z, axis=-1)


class MLPEnsemble(nn.Module):
    """Ensemble of CTRNN cells."""

    num_modules: int
    model: type = MLP
    out_size: int | None = None
    out_dist: str | None = None
    kwargs: dict = field(default_factory=dict)
    skip_connection: bool = False

    @nn.compact
    def __call__(self, x):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters
        ----------
        h : List
            of rnn submodule states
        x : Array
            input
        training : bool, optional, by default False
            If true, returns one value per submodule in order to train them independently,
            If false, mean of submodules or a Mixed Distribution is returned.

        Returns
        -------
        _type_
            _description_
        """
        outs = []
        for i in range(self.num_modules):
            # Loop over rnn submodules
            out = self.model(**self.kwargs, name=f"mlp{i}")(x)
            # Optional Skip connection
            if self.skip_connection:
                # FIXME: That's not what a skip-connection is!
                out = jnp.concatenate([x, out], axis=-1)
            # Make distribution for each submodule
            if self.out_size is not None:
                out = DistributionLayer(self.out_size, self.out_dist)(out)
            outs.append(out)

        if not self.out_dist:
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-2), *outs)

        else:
            # Last dim is batch in distrax
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-1), *outs)
            outs = distrax.MixtureSameFamily(
                distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs
            )

        return outs


class DistributionLayer(nn.Module):
    """Parameterized distribution output layer."""

    out_size: int
    distribution: str = "Normal"
    eps: float = 0.01
    f_align: bool = False

    @nn.compact
    def __call__(self, x):
        """Make the distribution from given vector."""
        if self.distribution == "Normal":
            x = FADense(2 * self.out_size, f_align=self.f_align)(x)
            loc, scale = jnp.split(x, 2, axis=-1)
            return distrax.Normal(loc, jax.nn.softplus(scale) + self.eps)
        elif self.distribution == "Categorical":
            out_size = (
                np.prod(self.out_size)
                if isinstance(self.out_size, tuple)
                else self.out_size
            )
            x = FADense(out_size, f_align=self.f_align)(x)
            if isinstance(self.out_size, tuple):
                x = x.reshape(self.out_size)
            return distrax.Categorical(logits=x)
        else:
            # Becomes deterministic
            return FADense(self.out_size, f_align=self.f_align)(x)
