"""Neural networks built with flax."""

from dataclasses import dataclass, field
from typing import Callable, Tuple
from chex import PRNGKey
import numpy as np
import jax
import jax.numpy as jnp
import distrax
import flax.linen as nn

from jax_rtrl.models.jax_util import zeros_like_tree

from .ctrnn import CTRNNCell, OnlineCTRNNCell


class FADense(nn.Dense):
    """Dense Layer with feedback alignment."""

    f_align: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)

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

        def f(mdl, x, B):
            return nn.Dense.__call__(mdl, x)

        def fwd(mdl, x, B):
            """Forward pass with tmp for backward pass."""
            return nn.Dense.__call__(mdl, x), (x, B)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _B = tmp
            grads = {"params": {"kernel": jnp.einsum("...X,...Y->YX", y_bar, _x)}}
            if self.use_bias:
                grads["params"]["bias"] = jnp.einsum("...X->X", y_bar)
            # if self.f_align:
            #     grads['params']['B'] = jnp.zeros_like(B)
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
    If f_align is true, each layer uses feedback alignment instead of backpropagation."""

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
    def __call__(self, x, rng=None):  # noqa
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
                out = jnp.concatenate([x, out], axis=-1)
            # Make distribution for each submodule
            if self.out_size is not None:
                out = DistributionLayer(self.out_size, self.out_dist)(out)
            outs.append(out)

        if not self.out_dist:
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-2), *outs)
            if rng is not None:
                outs = jnp.mean(outs, axis=0)
        else:
            # Last dim is batch in distrax
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-1), *outs)
            outs = distrax.MixtureSameFamily(distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs)
            if rng is not None:
                outs = outs.sample(seed=rng)

        return outs


class MultiLayerRNN(nn.RNNCellBase):
    """Multilayer RNN."""

    layers: list
    rnn_cls: nn.RNNCellBase = CTRNNCell
    rnn_kwargs: dict = field(default_factory=dict)

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        shapes = zip(self.layers, [input_shape, *[(s,) for s in self.layers[:-1]]])
        return [self.rnn_cls(size, **self.rnn_kwargs).initialize_carry(rng, in_size) for size, in_size in shapes]

    def setup(self):
        self.rnns = [self.rnn_cls(size, **self.rnn_kwargs, name=f"rnn_{i}") for i, size in enumerate(self.layers)]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @nn.compact
    def __call__(self, carries, x, **kwargs):
        """Call MLP."""
        for i, rnn in enumerate(self.rnns):
            carries[i], x = rnn(carries[i], x, **kwargs)
        return carries, x


class FAMultiLayerRNN(MultiLayerRNN):
    """MultiLayer RNN with different modes of backpropagating error signals."""

    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
    fa_type: str = "bp"

    @nn.compact
    def __call__(self, carries, x, **kwargs):
        if self.fa_type == "bp":
            return MultiLayerRNN.__call__(self, carries, x, **kwargs)
        elif self.fa_type == "dfa":
            Bs = [
                self.variable(
                    "falign",
                    f"B{i}",
                    self.kernel_init,
                    self.make_rng() if self.has_rng("params") else None,
                    (size, self.layers[-1]),
                ).value
                for i, size in enumerate(self.layers[:-1])
            ]
        else:
            raise ValueError("unknown fa_type: " + self.fa_type)

        def f(mdl, _carries, x, _Bs):
            return MultiLayerRNN.__call__(mdl, _carries, x, **kwargs)

        def fwd(mdl: MultiLayerRNN, _carries, x, _Bs):
            """Forward pass with tmp for backward pass."""
            vjps = []
            for i, rnn in enumerate(mdl.rnns):
                (_carries[i], x), vjp_func = jax.vjp(rnn.apply, rnn.variables, _carries[i], x, **kwargs)
                vjps.append(vjp_func)

            return (_carries, x), (vjps, _Bs)

        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            vjps, _Bs = tmp
            grads = zeros_like_tree(self.variables["params"])
            grads_inputs = []
            for i in range(len(self.rnns)):
                y_t = _Bs[i] @ y_bar[1] if i < len(self.rnns) - 1 else y_bar[1]
                params_t, *inputs_t = vjps[i]((y_bar[0][i], y_t))
                grads[f"rnn_{i}"] = params_t["params"]
                grads_inputs.append(inputs_t[0])
                if i == 0:
                    x_grad = inputs_t[1]

            return ({"params": grads}, grads_inputs, x_grad, zeros_like_tree(_Bs))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)

        return fa_grad(self, carries, x, Bs)


@dataclass
class RNNEnsembleConfig:
    num_modules: int
    layers: tuple[int]
    model: type = OnlineCTRNNCell
    out_size: int | None = None
    out_dist: str | None = None
    input_layers: tuple[int] | None = None  # TODO
    output_layers: tuple[int] | None = None
    fa_type: str = "bp"
    rnn_kwargs: dict = field(default_factory=dict)
    skip_connection: bool = False


class RNNEnsemble(nn.RNNCellBase):
    """Ensemble of RNN cells."""

    config: RNNEnsembleConfig

    def setup(self):
        self.rnns = [
            FAMultiLayerRNN(
                self.config.layers,
                rnn_cls=self.config.model,
                rnn_kwargs=self.config.rnn_kwargs,
                fa_type=self.config.fa_type,
                name=f"rnns_{i}",
            )
            for i in range(self.config.num_modules)
        ]
        if self.config.output_layers:
            self.mlps_out = [
                MLP(self.config.output_layers, f_align=self.config.rnn_kwargs.get("f_align", False), name=f"mlps_{i}")
                for i in range(self.config.num_modules)
            ]
        if self.config.out_size is not None:
            # Make distribution for each submodule
            self.dists = [
                DistributionLayer(self.config.out_size, self.config.out_dist, name=f"dists_{i}")
                for i in range(self.config.num_modules)
            ]

    def _postprocessing(self, outs, x):
        for i in range(self.config.num_modules):
            out = outs[i]
            # Optional Skip connection
            if self.config.skip_connection:
                out = jnp.concatenate([x, out], axis=-1)
            # Outup FF layers
            if self.config.output_layers:
                out = self.mlps_out[i](out)
            if self.config.out_size is not None:
                # Make distribution for each submodule
                outs[i] = self.dists[i](out)

        # Aggregate outputs
        if not self.config.out_dist:
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-2), *outs)
        else:
            # Last dim is batch in distrax
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-1), *outs)
            outs = distrax.MixtureSameFamily(distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs)
        return outs

    def __call__(self, h: jax.Array | None = None, x: jax.Array = None, rng=None, **call_args):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters
        ----------
        h : List
            of rnn submodule states
        x : Array
            input
        rng : PRNGKey, optional,
            if given, returns one value per submodule in order to train them independently,
            If None, mean of submodules or a Mixed Distribution is returned.

        Returns
        -------
        _type_
            _description_
        """
        if h is None:
            h = self.initialize_carry(jax.random.key(0), x.shape)
        outs = []
        carry_out = []
        for i in range(self.config.num_modules):
            # Loop over rnn submodules
            carry, out = self.rnns[i](h[i], x, **call_args)
            carry_out.append(carry)
            outs.append(out)

        # Post-process and aggregate outputs
        outs = self._postprocessing(outs, x)

        if rng is not None:
            outs = outs.sample(seed=rng) if self.config.out_dist else outs.mean(axis=0)
        return carry_out, outs

    def _loss(self, hidden, loss_fn, target, x):
        # Post-process and aggregate outputs
        outs = self._postprocessing(hidden, x)
        return loss_fn(outs, target)

    @nn.nowrap
    def loss_and_grad(self, params, loss_fn, carry, hidden, x, target):
        # Compute gradient of loss wrt to network output
        loss, (grads, df_dh) = jax.value_and_grad(self.apply, argnums=[0, 1])(
            params, [c[-1][0].real for c in carry], loss_fn, target, x, method=self._loss
        )

        # Compute parameter gradients for each RNN
        for i, (c, d) in enumerate(zip(carry, df_dh)):
            grads["params"][f"rnns_{i}"] = self.config.model.rtrl_gradient(
                c, d, plasticity=self.config.rnn_kwargs["plasticity"]
            )[0]
        return loss, grads

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return [
            FAMultiLayerRNN(
                self.config.layers, rnn_cls=self.config.model, rnn_kwargs=self.config.rnn_kwargs
            ).initialize_carry(rng, input_shape)
            for _ in range(self.config.num_modules)
        ]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @staticmethod
    def clip_tau(params):
        """HACK: clip tau to > 1.0"""
        for k in params["params"]["rnn"]:
            for _l in params["params"]["rnn"][k]:
                params["params"]["rnn"][k][_l]["tau"] = jnp.clip(params["params"]["rnn"][k][_l]["tau"], min=1.0)
        return params


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
            out_size = np.prod(self.out_size) if isinstance(self.out_size, tuple) else self.out_size
            x = FADense(out_size, f_align=self.f_align)(x)
            if isinstance(self.out_size, tuple):
                x = x.reshape(self.out_size)
            return distrax.Categorical(logits=x)
        else:
            # Becomes deterministic
            return FADense(self.out_size, f_align=self.f_align)(x)
