"""Neural networks built with flax."""

from dataclasses import field
from typing import Callable, Literal

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax_rtrl.models import distributions
import numpy as np

from jax_rtrl.util.jax_util import get_normalization_fn


class FADense(nn.Dense):
    """Dense Layer with feedback alignment."""

    f_align: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        if not self.f_align:
            return nn.Dense.__call__(self, x)

        B = self.variable(
            "falign",
            "B",
            self.kernel_init,
            self.make_rng() if self.has_rng("params") else None,
            (x.shape[-1], self.features),
            self.param_dtype,
        ).value

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
    """Multi-Layer Perceptron (MLP) implementation using Flax.

    Attributes:
        layers (list): List of integers specifying the number of units in each layer.
                      The last element determines the output dimension.
        activation_fn (Callable, optional): Activation function applied after each
                                          hidden layer. Defaults to jax.nn.relu.
        f_align (bool, optional): If True, uses feedback alignment instead of
                                 standard backpropagation. Defaults to False.
        kernel_init (nn.initializers.Initializer, optional): Weight initialization
                                                            strategy. Defaults to
                                                            lecun_normal().
        norm (str | None, optional): Type of normalization to apply. Can be 'layer'
                                    for layer normalization, 'batch' for batch
                                    normalization, or None for no normalization.
                                    Defaults to None.
    Note:
        The activation function is applied after every layer except the output layer.
        Normalization (if specified) is applied before each layer transformation.
    TODO:
        Add support for dropout regularization.
    """

    layers: list
    activation_fn: Callable = jax.nn.silu
    f_align: bool = False
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    norm: str | None = None  # 'layer' or 'batch'

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Call MLP."""
        for size in self.layers[:-1]:
            x = FADense(size, f_align=self.f_align, kernel_init=self.kernel_init)(x)
            x = get_normalization_fn(self.norm, training=training)(x)
            x = self.activation_fn(x)
        x = get_normalization_fn(self.norm, training=training)(x)
        x = FADense(
            self.layers[-1], f_align=self.f_align, kernel_init=self.kernel_init
        )(x)
        return x


class MLPCell(nn.RNNCellBase):
    """Wrapper to use Single-Layer-Perceptron as RNN cell."""

    num_units: int

    def setup(self):
        """Initialize MLP cell."""
        self.mlp = MLP(layers=[self.num_units])

    @nn.compact
    def __call__(self, carry, x, training: bool = True):
        """Call MLP cell."""
        out = self.mlp(x, training=training)
        return carry, out

    def initialize_carry(self, rng, input_shape):
        return None


class MLPEnsemble(nn.Module):
    """Ensemble of CTRNN cells."""

    num_modules: int = 1
    model: type = MLP
    out_size: int | None = None
    out_dist: str | None = None
    kwargs: dict = field(default_factory=dict)
    skip_connection: bool = False

    @nn.compact
    def __call__(self, x, training: bool = True):  # noqa
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
            out = self.model(**self.kwargs, name=f"mlp{i}")(x, training=training)
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


def straight_through_wrapper(  # pylint: disable=invalid-name
    Distribution,
) -> distrax.DistributionLike:
    """Wrap a distribution to use straight-through gradient for samples."""

    def sample(self, seed, sample_shape=()):  # pylint: disable=g-doc-args
        """Sampling with straight through biased gradient estimator.

        Sample a value from the distribution, but backpropagate through the
        underlying probability to compute the gradient.

        References:
          [1] Yoshua Bengio, Nicholas Léonard, Aaron Courville, Estimating or
          Propagating Gradients Through Stochastic Neurons for Conditional
          Computation, https://arxiv.org/abs/1308.3432

        Args:
          seed: a random seed.
          sample_shape: the shape of the required sample.

        Returns:
          A sample with straight-through gradient.
        """
        # pylint: disable=protected-access
        obj = Distribution(probs=self._probs, logits=self._logits)
        assert isinstance(obj, (distrax.Categorical, distrax.Bernoulli))
        sample = obj.sample(seed=seed, sample_shape=sample_shape)
        probs = obj.probs
        padded_probs = _pad(probs, sample.shape)

        if isinstance(obj, distrax.Categorical):
            sample = jax.nn.one_hot(sample, obj.probs.shape[-1])
        # Keep sample unchanged, but add gradient through probs.
        sample += padded_probs - jax.lax.stop_gradient(padded_probs)
        return sample

    def mode(self):
        """Return the mode of the distribution."""
        obj = Distribution(probs=self._probs, logits=self._logits)
        assert isinstance(obj, (distrax.Categorical, distrax.Bernoulli))
        return obj.mode()

    def _pad(probs, shape):
        """Grow probs to have the same number of dimensions as shape."""
        while len(probs.shape) < len(shape):
            probs = probs[None]
        return probs

    parent_name = Distribution.__name__
    # Return a new object, overriding sample.
    return type(
        "StraightThrough" + parent_name,
        (Distribution,),
        {"sample": sample, "mode": mode},
    )


class DistributionLayer(nn.Module):
    """
    Parameterized distribution output layer.

    An optional MLP is followed by a linear projection whose outputs
    parameterize the distribution. String names are resolved with
    `distributions.get_distribution`; heads deriving from
    `distributions.DistributionHead` decide how many outputs they need and how
    to build themselves. A callable `distribution` gets the MLP output
    directly, and `None` returns the projection itself.
    """

    out_size: int | tuple[int, ...]
    distribution: str | Callable | None = "LogStddevNormal"
    layers: tuple[int, ...] = ()
    mapping: Literal["dense", "affine"] = "dense"
    eps_unimix: float = 0.0  # Uniform mixing for Categorical/Bernoulli
    loc_bounds: (
        float | tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] | None
    ) = None  # A float or tuple of (min, max) bounds for the location parameter of the distribution, either a global bound or a tuple of bounds for each output dimension. If None, no bounds are applied.
    scale_bounds: (
        float | tuple[float, float] | tuple[tuple[float, ...], tuple[float, ...]] | None
    ) = 0  # A float or tuple of (min, max) bounds for the scale parameter of the distribution similar to loc_bounds. If None, no bounds are applied.
    num_bins: int = 255  # Only used by TwoHot
    dist_kwargs: dict | None = None  # extra head config, e.g. MixedCategorical's
    activation_fn: Callable | None = None  # Also applied after the MLP if set
    f_align: bool = False
    norm: str | None = None  # 'layer' or 'batch'
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x, training: bool = True):
        """Make the distribution from given vector."""
        if self.layers:
            x = MLP(
                layers=self.layers,
                activation_fn=self.activation_fn or jax.nn.relu,
                f_align=self.f_align,
                kernel_init=self.kernel_init,
                norm=self.norm,
            )(x, training=training)
            if self.activation_fn is not None:
                x = self.activation_fn(x)

        if isinstance(self.distribution, Callable):
            return self.distribution(x)

        dist_cls = distributions.get_distribution(self.distribution)
        cfg = dict(
            loc_bounds=self.loc_bounds,
            scale_bounds=self.scale_bounds,
            eps_unimix=self.eps_unimix,
            num_bins=self.num_bins,
            **(self.dist_kwargs or {}),
        )
        is_head = dist_cls is not None and issubclass(
            dist_cls, distributions.DistributionHead
        )
        if is_head:
            num_params = dist_cls.num_params(self.out_size, **cfg)
        else:
            num_params = int(np.prod(self.out_size))

        if self.mapping == "dense":
            x = FADense(num_params, f_align=self.f_align, kernel_init=self.kernel_init)(x)
        elif self.mapping == "affine":
            assert x.shape[-1] >= num_params, (
                f"Input dimension {x.shape[-1]} must be greater than or equal to output dimension {num_params} for affine mapping."
            )
            x = FAAffine(num_params, f_align=self.f_align)(x)
        else:
            raise ValueError(f"Invalid mapping type: {self.mapping}.")

        if dist_cls is None:
            return x
        if not is_head:
            return dist_cls(x)

        if issubclass(dist_cls, distributions.Scaled) and self.loc_bounds is None:
            cfg["learned_bounds"] = (
                self.param("min_val", nn.initializers.constant(-1.0), self.out_size),
                self.param("max_val", nn.initializers.constant(1.0), self.out_size),
            )
        return dist_cls.from_params(x, self.out_size, **cfg)
