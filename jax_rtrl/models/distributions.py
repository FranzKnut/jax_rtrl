"""Distributions implemented with distrax."""

from numbers import Number
from typing import Any

from chex import PRNGKey
import distrax
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from jax_rtrl.util.jax_util import sigmoid_between


class UniformMixture(distrax.MixtureSameFamily):
    """A uniform mixture of distributions."""

    def __init__(self, components):
        # HACK: Distrax MixtureSameFamily expects batch_dim in the last dimension? so we swap axes here.
        components = jax.tree.map(lambda d: jax.numpy.swapaxes(d, 0, -1), components)
        super().__init__(
            distrax.Categorical(logits=jax.numpy.zeros(components.batch_shape)),
            components,
        )

    def mode(self):
        """Return the mode of the mixture distribution."""
        try:
            return self.mean()
        except NotImplementedError:
            print(
                "WARNING: Mean is not implemented for this mixture distribution. Using mean of modes."
            )
            return jax.numpy.mean(self.components_distribution.mode(), axis=-1)


class JointVectorDist(distrax.Joint):
    """A joint distribution where events are concatenated to be vectors."""

    def log_prob(self, value):
        """Compute joint log probability."""
        return super().log_prob(jnp.split(value, value.shape[-1], axis=-1))

    def variance(self) -> Any:
        """Compute joint variance."""
        def _variance(leaf):
            if hasattr(leaf, "variance"):
                try:
                    return leaf.variance()
                except NotImplementedError:
                    pass
            if hasattr(leaf, "distribution") and hasattr(leaf.distribution, "variance"):
                try:
                    return leaf.distribution.variance()
                except NotImplementedError:
                    pass
            else:
                raise NotImplementedError(
                    f"Variance not implemented for distribution {type(leaf)}"
                )
        return jnp.stack([_variance(leaf) for leaf in self._distributions], axis=-1)

    def sample(self, seed: PRNGKey, sample_shape: tuple[int, ...] = ()):
        """Sample from the joint distribution."""
        return jnp.stack(
            distrax.Joint.sample(self, seed=seed, sample_shape=sample_shape), axis=-1
        )


def _as_bounds(bounds) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a number `b` or a `(low, high)` pair to `(low, high)` arrays."""
    if isinstance(bounds, Number):
        bounds = (-bounds, bounds)
    low, high = bounds
    # numpy stays concrete under jit, unlike jnp
    return np.asarray(low), np.asarray(high)


class DistributionHead:
    """
    Mixin for distributions that can be built from a flat network output.

    Subclasses define how many outputs they need for a given event size and
    how to turn those outputs into a distribution. Config keywords that a
    subclass doesn't use are ignored, so `DistributionLayer` can pass the
    same set to every head.
    """

    @classmethod
    def num_params(cls, out_size, **cfg) -> int:
        """Number of network outputs needed for an event of shape `out_size`."""
        return int(np.prod(out_size))

    @classmethod
    def from_params(cls, x: jax.Array, out_size, **cfg):
        """Build the distribution from network outputs `x`."""
        raise NotImplementedError


class Deterministic(DistributionHead, distrax.Deterministic):
    """Point mass located at the network output."""

    @classmethod
    def from_params(cls, x, out_size, **_):
        """Use `x` as the location."""
        return cls(x)


class LocScaleHead(DistributionHead):
    """
    Head for distributions parameterized by `loc` and `scale`.

    Notes
    -----
    `loc_bounds` squashes the location into `(low, high)` with a sigmoid, or
    into `(-b, b)` with a tanh if it is a number `b`. `scale_bounds` squashes
    the scale into `(low, high)`, or adds a number `b` to a softplus.
    """

    @classmethod
    def num_params(cls, out_size, **_) -> int:
        """Twice the event size, for location and scale."""
        return 2 * int(np.prod(out_size))

    @classmethod
    def _bound_loc(cls, loc, loc_bounds):
        if isinstance(loc_bounds, tuple):
            return sigmoid_between(loc, *loc_bounds)
        if isinstance(loc_bounds, Number):
            return jax.nn.tanh(loc) * loc_bounds
        return loc

    @classmethod
    def _bound_scale(cls, scale, scale_bounds):
        if isinstance(scale_bounds, tuple):
            return sigmoid_between(scale, *scale_bounds)
        if isinstance(scale_bounds, Number):
            return jax.nn.softplus(scale) + scale_bounds
        return scale

    @classmethod
    def from_params(cls, x, out_size, loc_bounds=None, scale_bounds=0.0, **_):
        """Split `x` into bounded location and scale."""
        loc, scale = jnp.split(x, 2, axis=-1)
        return cls(
            cls._bound_loc(loc, loc_bounds), cls._bound_scale(scale, scale_bounds)
        )


class Normal(LocScaleHead, distrax.Normal):
    """Normal distribution head."""

    @classmethod
    def _bound_scale(cls, scale, scale_bounds):
        assert not isinstance(scale_bounds, Number) or scale_bounds >= 0, (
            "scale_bounds must be non-negative for Normal distributions"
        )
        return super()._bound_scale(scale, scale_bounds)


class LogStddevNormal(LocScaleHead, distrax.LogStddevNormal):
    """Normal head parameterized by the log standard deviation."""


class NormalTanh(LocScaleHead, distrax.Transformed):
    """A Normal distribution followed by a Tanh transformation."""
    
    eps: float = 1e-6  # Small constant to avoid numerical issues

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self._normal, name)

    def __init__(self, loc, scale):
        self._normal = distrax.Normal(loc, scale)
        super().__init__(
            self._normal,
            # distrax.Independent(self._normal, 1),
            distrax.Tanh(),
            # distrax.Block(distrax.Tanh(), 1),
        )

    def mode(self) -> jax.Array:
        sample = self.distribution.mode()
        return self.bijector.forward(sample)

    def entropy(self) -> jax.Array:
        print("WARNING: Using base distribution's entropy in place of the true one!")
        return self.distribution.entropy()

    def variance(self) -> jax.Array:
        print("WARNING: Using base distribution's variance in place of the true one!")
        return self.distribution.variance()
    
    def log_prob(self, value: jax.Array) -> jax.Array:
        """Compute log probability of a value under the distribution."""
        value = jnp.clip(value, -1 + self.eps, 1 - self.eps)  # Clip to avoid numerical issues
        return super().log_prob(value)

    @classmethod
    def _bound_loc(cls, loc, loc_bounds):
        return loc  # Tanh already bounds the location


class Beta(DistributionHead, distrax.Beta):
    """Beta distribution head on [0, 1]."""

    @classmethod
    def num_params(cls, out_size, **_) -> int:
        """Twice the event size, for both concentrations."""
        return 2 * int(np.prod(out_size))

    @classmethod
    def from_params(cls, x, out_size, **_):
        """Map both halves of `x` to concentrations above one."""
        # Concentrations > 1 keep the density unimodal, so `mode` is defined
        alpha, beta = jnp.split(jax.nn.softplus(x) + 1.0, 2, axis=-1)
        return cls(alpha, beta)


class Scaled(DistributionHead, distrax.Transformed):
    """
    Base distribution `base` mapped affinely onto `loc_bounds`.

    Subclasses set `base` and may override `_shift_factor` to choose how the
    base support maps onto `(low, high)`. By default `[0, 1]` maps onto it.
    If `loc_bounds` is None, `DistributionLayer` passes learned bounds as
    `learned_bounds`.
    """

    base: type[DistributionHead]

    def __init__(self, distribution, shift, factor):
        super().__init__(distribution, distrax.ScalarAffine(shift, factor))

    @classmethod
    def num_params(cls, out_size, **cfg) -> int:
        """Same as the base distribution."""
        return cls.base.num_params(out_size, **cfg)

    @staticmethod
    def _shift_factor(low, high):
        return low, high - low

    @classmethod
    def from_params(cls, x, out_size, loc_bounds=None, learned_bounds=None, **cfg):
        """Build the base distribution and scale it to the bounds."""
        dist = cls.base.from_params(x, out_size, loc_bounds=loc_bounds, **cfg)
        bounds = loc_bounds if loc_bounds is not None else learned_bounds
        assert bounds is not None, f"{cls.__name__} needs loc_bounds"
        shift, factor = cls._shift_factor(*_as_bounds(bounds))
        # Full shape keeps pytree leaves consistent, e.g. for SSM axis swaps.
        shift = jnp.broadcast_to(shift, dist.batch_shape)
        factor = jnp.broadcast_to(factor, dist.batch_shape)
        return cls(dist, shift, factor)

    def mode(self) -> jax.Array:
        """Mode of the base distribution, transformed."""
        return self.bijector.forward(self.distribution.mode())

    def variance(self) -> jax.Array:
        """Base variance times the squared scale."""
        return self.bijector.scale**2 * self.distribution.variance()


class ScaledNormal(Scaled):
    """Normal distribution scaled from [0, 1] to the bounds."""

    base = Normal


class ScaledLogStddevNormal(Scaled):
    """LogStddevNormal distribution scaled from [0, 1] to the bounds."""

    base = LogStddevNormal


class ScaledNormalTanh(Scaled):
    """NormalTanh distribution scaled from [-1, 1] to the bounds."""

    base = NormalTanh

    @staticmethod
    def _shift_factor(low, high):
        return (low + high) / 2, (high - low) / 2


class ScaledBeta(Scaled):
    """Beta distribution scaled from [0, 1] to the bounds."""

    base = Beta


class DiscreteHead(DistributionHead):
    """
    Head for discrete distributions with optional uniform mixing.

    Notes
    -----
    With `eps_unimix > 0` the probabilities are blended with a uniform
    distribution to prevent collapse (DreamerV3). Otherwise clipped logits are
    used directly, so distrax computes a stable `log_softmax` and exploding
    weights can't produce `inf - inf = NaN`.
    """

    @classmethod
    def _probs(cls, x) -> jax.Array:
        raise NotImplementedError

    @classmethod
    def _num_classes(cls, x) -> int:
        raise NotImplementedError

    @classmethod
    def from_params(cls, x, out_size, eps_unimix=0.0, **_):
        """Reshape `x` to `out_size` and build the distribution."""
        if isinstance(out_size, tuple):
            x = x.reshape(*x.shape[:-1], *out_size)
        if eps_unimix > 0:
            probs = cls._probs(x) * (1 - eps_unimix) + eps_unimix / cls._num_classes(x)
            return cls(probs=probs)
        return cls(logits=jnp.clip(x, -20.0, 20.0))


class Categorical(DiscreteHead, distrax.Categorical):
    """Categorical head over the last axis of `out_size`."""

    @classmethod
    def _probs(cls, x):
        return jax.nn.softmax(x, axis=-1)

    @classmethod
    def _num_classes(cls, x):
        return x.shape[-1]


class Bernoulli(DiscreteHead, distrax.Bernoulli):
    """Independent Bernoulli head per output."""

    @classmethod
    def _probs(cls, x):
        return jax.nn.sigmoid(x)

    @classmethod
    def _num_classes(cls, x):
        return 2


def two_hot(x: jax.Array, bins: jax.Array) -> jax.Array:
    """
    Encode `x` as weights on its two neighbouring bins.

    Parameters
    ----------
    x : jax.Array
        Values to encode, clipped to the bin range.
    bins : jax.Array
        Sorted bin centers of shape (num_bins,).

    Returns
    -------
    jax.Array
        Array of shape (*x.shape, num_bins) whose expectation over `bins` is `x`.
    """
    num_bins = bins.shape[0]
    x = jnp.clip(x, bins[0], bins[-1])
    idx = jnp.clip(jnp.searchsorted(bins, x, side="right") - 1, 0, num_bins - 2)
    lo, hi = bins[idx], bins[idx + 1]
    w_hi = ((x - lo) / (hi - lo))[..., None]
    return (1 - w_hi) * jax.nn.one_hot(idx, num_bins) + w_hi * jax.nn.one_hot(
        idx + 1, num_bins
    )


@struct.dataclass
class TwoHot(DistributionHead):
    """
    Categorical over evenly spaced bins, trained with two-hot targets (DreamerV3).

    Mimics the `distrax.Normal` interface. `logits` keeps the bins flattened
    into the last axis, so it has the same rank as a Normal head's `loc`.
    """

    logits: jax.Array  # (..., event_size * num_bins)
    num_bins: int = struct.field(pytree_node=False)
    low: float = struct.field(pytree_node=False)
    high: float = struct.field(pytree_node=False)

    @classmethod
    def num_params(cls, out_size, num_bins=255, **_) -> int:
        """One logit per bin and event dimension."""
        return int(np.prod(out_size)) * num_bins

    @classmethod
    def from_params(cls, x, out_size, loc_bounds=None, num_bins=255, **_):
        """
        Use `x` as flattened bin logits over `loc_bounds`.

        Parameters
        ----------
        x : jax.Array, shape (..., prod(out_size) * num_bins)
            Bin logits, bins varying fastest.
        out_size : int or tuple of int
            Event shape.
        loc_bounds : float or (float, float)
            Range spanned by the bins; a number `b` means `(-b, b)`. Required.
        num_bins : int
            Number of bins per event dimension.
        """
        assert loc_bounds is not None, "TwoHot needs loc_bounds for the bin range"
        low, high = _as_bounds(loc_bounds)
        return cls(x, num_bins, float(low), float(high))

    @property
    def bins(self) -> jax.Array:
        """Bin centers of shape (num_bins,)."""
        return jnp.linspace(self.low, self.high, self.num_bins)

    @property
    def bin_logits(self) -> jax.Array:
        """Logits of shape (..., event_size, num_bins)."""
        return self.logits.reshape(*self.logits.shape[:-1], -1, self.num_bins)

    @property
    def probs(self) -> jax.Array:
        """Bin probabilities of shape (..., event_size, num_bins)."""
        return jax.nn.softmax(self.bin_logits, axis=-1)

    def mean(self) -> jax.Array:
        """Expected bin value."""
        return self.probs @ self.bins

    def mode(self) -> jax.Array:
        """Point prediction, the expectation as in DreamerV3."""
        return self.mean()

    @property
    def loc(self) -> jax.Array:
        """Alias of `mean` for code written against Normal heads."""
        return self.mean()

    def variance(self) -> jax.Array:
        """Variance of the bin values."""
        return jnp.sum(self.probs * (self.bins - self.mean()[..., None]) ** 2, -1)

    def stddev(self) -> jax.Array:
        """Standard deviation of the bin values."""
        return jnp.sqrt(self.variance())

    @property
    def scale(self) -> jax.Array:
        """Alias of `stddev` for code written against Normal heads."""
        return self.stddev()

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Negative two-hot cross entropy, elementwise over the event axis."""
        target = jax.lax.stop_gradient(two_hot(x, self.bins))
        return jnp.sum(target * jax.nn.log_softmax(self.bin_logits, axis=-1), -1)

    def sample(self, seed: PRNGKey) -> jax.Array:
        """Sample a bin value per event dimension."""
        return self.bins[jax.random.categorical(seed, self.bin_logits)]

    def entropy(self) -> jax.Array:
        """Entropy of the bin distribution."""
        return -jnp.sum(self.probs * jax.nn.log_softmax(self.bin_logits, -1), -1)


_PAD_LOGIT = -1e9  # padded classes of a shorter categorical axis


def _split_columns(num_classes) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Indices of the continuous (0 classes) and categorical columns."""
    cont = tuple(i for i, k in enumerate(num_classes) if k == 0)
    cat = tuple(i for i, k in enumerate(num_classes) if k > 0)
    return cont, cat


def encode_mixed_action(value, num_classes, mask=None) -> jax.Array:
    """
    Continuous columns followed by one one-hot per categorical column.

    Parameters
    ----------
    value : jax.Array, shape (..., len(num_classes))
        Raw actions; categorical columns hold integer codes (possibly as floats).
    num_classes : sequence of int
        Classes per column, 0 for a continuous one.
    mask : jax.Array, optional
        Same shape as `value`. A categorical entry that's False encodes as
        all zeros rather than as level 0.

    Returns
    -------
    jax.Array, shape (..., n_cont + sum(num_classes))
    """
    cont_idx, cat_idx = _split_columns(num_classes)
    parts = [value[..., cont_idx]]
    for i in cat_idx:
        k = num_classes[i]
        code = jnp.clip(jnp.round(value[..., i]), 0, k - 1).astype(jnp.int32)
        one_hot = jax.nn.one_hot(code, k, dtype=value.dtype)
        if mask is not None:
            one_hot = one_hot * mask[..., i, None]
        parts.append(one_hot)
    return jnp.concatenate(parts, axis=-1)


@struct.dataclass
class MixedCategorical(DistributionHead):
    """
    Continuous head over some columns, an independent categorical over the rest.

    Methods return one value per action column in the original order.
    Categorical columns hold integer codes (as floats).

    Notes
    -----
    `logits` holds per-axis log-probabilities padded with a large negative
    value. Methods renormalize, so ensemble-averaged logits stay valid.
    """

    cont: Any  # distribution over the continuous columns
    logits: jax.Array  # (..., n_cat, max(num_classes))
    num_classes: tuple[int, ...] = struct.field(pytree_node=False)

    @classmethod
    def _parse(cls, out_size, num_classes, cont_dist):
        num_classes = tuple(int(k) for k in num_classes)
        assert len(num_classes) == int(np.prod(out_size)), (
            f"num_classes has {len(num_classes)} entries for out_size {out_size}"
        )
        cont_idx, cat_idx = _split_columns(num_classes)
        assert cont_idx and cat_idx, "MixedCategorical needs both column kinds"
        return num_classes, len(cont_idx), get_distribution(cont_dist)

    @classmethod
    def num_params(cls, out_size, num_classes=(), cont_dist="Normal", **cfg) -> int:
        """Continuous head's parameters plus one logit per class."""
        num_classes, n_cont, cont_cls = cls._parse(out_size, num_classes, cont_dist)
        return cont_cls.num_params(n_cont, **cfg) + sum(num_classes)

    @classmethod
    def from_params(
        cls, x, out_size, num_classes=(), cont_dist="Normal", eps_unimix=0.0, **cfg
    ):
        """
        Split `x` into the continuous head's parameters and per-axis logits.

        Parameters
        ----------
        x : jax.Array, shape (..., num_params)
            Continuous head's parameters, then one logit per class of each
            categorical column in column order.
        out_size : int or tuple of int
            Event shape; must have `len(num_classes)` elements.
        num_classes : sequence of int
            Classes per column, 0 for a continuous one. Needs both kinds.
        cont_dist : str
            Name of the head for the continuous columns.
        eps_unimix : float
            Weight of a uniform mixed into each categorical axis. Bounds an
            axis's NLL by ``log(K / eps_unimix)`` for `K` classes.
        **cfg
            Forwarded to the continuous head.
        """
        num_classes, n_cont, cont_cls = cls._parse(out_size, num_classes, cont_dist)
        n_cont_params = cont_cls.num_params(n_cont, **cfg)
        cont = cont_cls.from_params(x[..., :n_cont_params], n_cont, **cfg)
        k_max = max(num_classes)
        axes, start = [], n_cont_params
        for k in num_classes:
            if k == 0:
                continue
            logp = jax.nn.log_softmax(x[..., start : start + k], axis=-1)
            start += k
            if eps_unimix > 0:
                logp = jnp.log(jnp.exp(logp) * (1 - eps_unimix) + eps_unimix / k)
            pad = [(0, 0)] * (logp.ndim - 1) + [(0, k_max - k)]
            axes.append(jnp.pad(logp, pad, constant_values=_PAD_LOGIT))
        return cls(cont, jnp.stack(axes, axis=-2), num_classes)

    @property
    def _cont_idx(self) -> tuple[int, ...]:
        return _split_columns(self.num_classes)[0]

    @property
    def _cat_idx(self) -> tuple[int, ...]:
        return _split_columns(self.num_classes)[1]

    @property
    def _cat_sizes(self) -> np.ndarray:
        return np.array([self.num_classes[i] for i in self._cat_idx])

    @property
    def log_probs(self) -> jax.Array:
        """Normalized class log-probabilities, (..., n_cat, max(num_classes))."""
        return jax.nn.log_softmax(self.logits, axis=-1)

    @property
    def probs(self) -> jax.Array:
        """Class probabilities, (..., n_cat, max(num_classes))."""
        return jax.nn.softmax(self.logits, axis=-1)

    def _merge(self, cont_vals, cat_vals) -> jax.Array:
        dtype = jnp.result_type(cont_vals, jnp.float32)
        out = jnp.zeros((*cont_vals.shape[:-1], len(self.num_classes)), dtype)
        out = out.at[..., self._cont_idx].set(cont_vals)
        return out.at[..., self._cat_idx].set(cat_vals.astype(dtype))

    def _codes(self, value) -> jax.Array:
        cat = jnp.round(value[..., self._cat_idx])
        return jnp.clip(cat, 0, self._cat_sizes - 1).astype(jnp.int32)

    def _cat_log_prob(self, codes) -> jax.Array:
        return jnp.take_along_axis(self.log_probs, codes[..., None], -1)[..., 0]

    def log_prob(self, value: jax.Array) -> jax.Array:
        """Per-column log-probability; categorical codes are rounded and clipped."""
        cont_lp = self.cont.log_prob(value[..., self._cont_idx])
        return self._merge(cont_lp, self._cat_log_prob(self._codes(value)))

    def mode(self) -> jax.Array:
        """Continuous mode and the most likely code per categorical column."""
        return self._merge(self.cont.mode(), jnp.argmax(self.logits, -1))

    def mean(self) -> jax.Array:
        """Continuous mean; categorical columns report their mode, as codes are nominal."""
        return self._merge(self.cont.mean(), jnp.argmax(self.logits, -1))

    def variance(self) -> jax.Array:
        """Continuous variance; zero for categorical columns."""
        cont_var = self.cont.variance()
        return self._merge(cont_var, jnp.zeros(self.logits.shape[:-1]))

    def stddev(self) -> jax.Array:
        """Continuous standard deviation; zero for categorical columns."""
        return jnp.sqrt(self.variance())

    def entropy(self) -> jax.Array:
        """Per-column entropy."""
        cat_ent = -jnp.sum(self.probs * self.log_probs, -1)
        return self._merge(self.cont.entropy(), cat_ent)

    def _sample_codes(self, seed, sample_shape) -> jax.Array:
        shape = tuple(sample_shape) + self.logits.shape[:-1]
        return jax.random.categorical(seed, self.logits, axis=-1, shape=shape)

    def sample(self, seed: PRNGKey, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Raw actions: continuous samples and integer codes."""
        return self.sample_and_log_prob(seed=seed, sample_shape=sample_shape)[0]

    def sample_and_log_prob(self, seed: PRNGKey, sample_shape: tuple[int, ...] = ()):
        """Raw actions and their per-column log-probabilities."""
        seed_cont, seed_cat = jax.random.split(seed)
        cont, cont_lp = self.cont.sample_and_log_prob(
            seed=seed_cont, sample_shape=sample_shape
        )
        codes = self._sample_codes(seed_cat, sample_shape)
        return self._merge(cont, codes), self._merge(cont_lp, self._cat_log_prob(codes))

    def _encode_one_hots(self, cont, one_hots) -> jax.Array:
        parts = [cont] + [one_hots[..., j, : self._cat_sizes[j]] for j in range(len(self._cat_idx))]
        return jnp.concatenate(parts, axis=-1)

    def _straight_through(self, scores) -> jax.Array:
        soft = jax.nn.softmax(scores, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(scores, -1), scores.shape[-1], dtype=soft.dtype)
        return hard + (soft - jax.lax.stop_gradient(soft))  # exactly `hard` forward

    def rsample_encoded(self, seed: PRNGKey, temperature: float = 1.0):
        """
        Reparameterized sample in the `encode_mixed_action` layout.

        Categorical columns use straight-through Gumbel-softmax: exact one-hots
        forward, the relaxed sample's gradient backward.

        Returns
        -------
        encoded : jax.Array, shape (..., n_cont + sum(K))
        log_prob : jax.Array, shape (..., len(num_classes))
            Per column, of the hard sample.
        """
        seed_cont, seed_cat = jax.random.split(seed)
        cont, cont_lp = self.cont.sample_and_log_prob(seed=seed_cont)
        gumbel = jax.random.gumbel(seed_cat, self.logits.shape, self.logits.dtype)
        one_hots = self._straight_through((self.log_probs + gumbel) / temperature)
        cat_lp = jnp.sum(jax.lax.stop_gradient(one_hots) * self.log_probs, -1)
        return self._encode_one_hots(cont, one_hots), self._merge(cont_lp, cat_lp)

    def mode_encoded(self) -> jax.Array:
        """Mode in the `encode_mixed_action` layout, straight-through on categorical columns."""
        return self._encode_one_hots(self.cont.mode(), self._straight_through(self.logits))

    def encode(self, value: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """Encode raw actions the way `rsample_encoded` lays out its samples."""
        return encode_mixed_action(value, self.num_classes, mask)


_ALIASES = {"beta": ScaledBeta}


def get_distribution(name: str | None) -> type | None:
    """
    Look up a distribution class by name.

    Searches this module first, then distrax. `None` means no distribution.
    """
    if name is None:
        return None
    if name in _ALIASES:
        return _ALIASES[name]
    cls = globals().get(name)
    if isinstance(cls, type) and issubclass(cls, distrax.Distribution | DistributionHead):
        return cls
    return getattr(distrax, name)
