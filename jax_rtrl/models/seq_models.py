"""Sequence models implemented with Flax."""

# Import necessary modules and libraries
from dataclasses import dataclass, field
from functools import partial
import re
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn

from jax_rtrl.models.cells import CELL_TYPES
from jax_rtrl.models.s5 import S5Config

from .jax_util import get_matching_leaves, set_matching_leaves, zeros_like_tree
from .mlp import MLP, DistributionLayer, FADense


@dataclass
class SequenceLayerConfig:
    """Configuration for SequenceLayer.

    Attributes:
        d_output: Output layer size.
        dropout: Dropout probability.
        norm: Type of normalization to use ('layer' or 'batch').
        glu: Whether to use Gated Linear Unit structure.
        skip_connection: Whether to use skip connections.
        activation: Activation function applied after the sequence model.
    """

    dropout: float = 0.0
    norm: str = "layer"
    glu: bool = False
    skip_connection: bool = False  # FIXME: Implemented wrongly
    activation: str | None = None


@dataclass
class RNNEnsembleConfig:
    """Configuration for RNNEnsemble.

    Attributes:
        model_name: Name of the RNN model.
        num_modules: Number of RNN modules in the ensemble.
        layers: Tuple specifying the number of layers in each module.
        num_blocks: Number of input chunks for parallel processing.
        glu: Whether to use Gated Linear Unit structure.
        out_size: Output size of the model.
        out_dist: Type of output distribution.
        input_layers: Configuration for input layers (if any).
        output_layers: Configuration for output layers (if any).
        fa_type: Feedback alignment type ('bp', 'fa', 'dfa').
        rnn_kwargs: Additional arguments for the RNN model.
        layer_config: Configuration for the sequence layer.
    """

    model_name: str | None
    num_modules: int
    layers: tuple[int]
    num_blocks: int = 1
    glu: bool = True
    out_size: int | None = None
    out_dist: str | None = None
    input_layers: tuple[int] | None = None  # TODO
    output_layers: tuple[int] | None = None
    fa_type: str = "bp"
    rnn_kwargs: dict = field(default_factory=dict)
    layer_config: SequenceLayerConfig = field(default_factory=SequenceLayerConfig)

    def __post_init__(self):
        """Post-initialization logic for RNNEnsembleConfig."""
        # Handle special cases for model_kwargs
        match = self.model_name and re.search(r"(rflo|rtrl)", self.model_name)
        if match:
            self.rnn_kwargs["plasticity"] = match.group(1)
        elif self.model_name in ["s5", "s5_rtrl"]:
            self.rnn_kwargs = {"config": S5Config(**self.rnn_kwargs)}
        if self.model_name not in ["bptt", "rtrl", "rflo"]:
            self.rnn_kwargs.pop("wiring", None)
            self.rnn_kwargs.pop("wiring_kwargs", None)
            print(
                f"WARNING specifying wiring does not work with model {self.model_name}. Deleting from kwargs"
            )


class SequenceLayer(nn.Module):
    """Single layer with normalization, sequence model, dropout, and optional GLU.

    Attributes:
        seq: Sequence module (e.g., RNN, LSTM).
        config: Configuration for the sequence layer.
        training: Whether the model is in training mode (affects dropout behavior).
    """

    rnn_cls: type[nn.RNNCellBase]  # RNN class
    rnn_size: int  # Size of the RNN layer
    d_output: int = None  # output layer size, defaults to 2 * hidden_size
    config: SequenceLayerConfig = field(
        default_factory=SequenceLayerConfig
    )  # configuration for the layer
    training: bool = True  # training mode (dropout in training mode only)
    num_blocks: int = 1  # Number of input chunks for parallel processing
    rnn_kwargs: dict = field(default_factory=dict)  # Additional arguments for the RNN

    @nn.nowrap
    def _make_rnn(self):
        return make_rnn(
            self.rnn_cls,
            self.rnn_size,
            num_blocks=self.num_blocks,
            **self.rnn_kwargs,
        )

    def setup(self):
        """Initialize the RNN module."""
        self.seq = self._make_rnn()

    @nn.compact
    def __call__(self, hidden, inputs, **kwargs):
        """Apply, layer norm, seq model, dropout and GLU in that order."""
        if self.config.norm in ["layer"]:
            normalization = nn.LayerNorm()
        elif self.config.norm in ["batch"]:
            normalization = nn.BatchNorm(
                use_running_average=not self.training, axis_name="batch"
            )
        else:
            normalization = lambda x: x  # noqa
        drop = nn.Dropout(
            self.config.dropout, broadcast_dims=[0], deterministic=not self.training
        )

        x = normalization(inputs)  # pre normalization
        hidden, x = self.seq(hidden, x, **kwargs)  # call seq model
        if self.config.activation is not None:
            x = getattr(nn, self.config.activation)(x)
        if self.config.skip_connection:
            x = x + inputs
        x = nn.Dense(self.d_output or hidden.shape[-1])(drop(x))
        if self.config.glu:
            # Gated Linear Unit
            x *= jax.nn.sigmoid(nn.Dense(self.d_output or hidden.shape[-1])(x))
        x = drop(x)

        return hidden, x

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry (hidden state) for the sequence model."""
        return self._make_rnn().initialize_carry(rng, input_shape)


class MultiLayerRNN(nn.RNNCellBase):
    """Multilayer RNN with configurable layers and sequence processing.

    Attributes:
        sizes: List of layer sizes.
        rnn_cls: Class of the RNN cell to use.
        rnn_kwargs: Additional arguments for the RNN cell.
        num_blocks: Number of input chunks for parallel processing.
        layer_config: Configuration for the sequence layer.
    """

    sizes: list[int]
    rnn_cls: type[nn.RNNCellBase]
    rnn_kwargs: dict = field(default_factory=dict)
    num_blocks: int = 1
    layer_config: SequenceLayerConfig = field(default_factory=SequenceLayerConfig)

    @nn.nowrap
    def _make_layers(self):
        """Create the Sequence submodels."""
        return [
            SequenceLayer(
                rnn_cls=self.rnn_cls,
                rnn_size=size,
                num_blocks=self.num_blocks,
                rnn_kwargs=self.rnn_kwargs,
                d_output=2 * size,
                config=self.layer_config,
                name=f"layer_{i}",
            )
            for i, size in enumerate(self.sizes)
        ]

    def setup(self):
        """Initialize submodules."""
        self.layers = self._make_layers()

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry (hidden state) for the RNN layers."""
        batch_size = input_shape[0:1] if len(input_shape) > 1 else ()
        shapes = [self.sizes[:1], *[(s,) for s in self.sizes[:-1]]]
        layers = self._make_layers()
        return [
            _l.initialize_carry(rng, batch_size + in_size)
            for _l, in_size in zip(layers, shapes)
        ]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @nn.compact
    def __call__(self, carries, x, **kwargs):
        """Call MLP."""
        x = FADense(self.sizes[0], name="input")(x)
        for i, rnn in enumerate(self.layers):
            carries[i], x = rnn(carries[i], x, **kwargs)
        return carries, x


class FAMultiLayerRNN(MultiLayerRNN):
    """MultiLayer RNN with feedback alignment for error signal propagation.

    Attributes:
        kernel_init: Initializer for feedback alignment weights.
        fa_type: Feedback alignment type ('bp', 'fa', 'dfa').
    """

    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal(
        in_axis=-1, out_axis=-2
    )
    fa_type: str = "bp"

    @nn.compact
    def __call__(self, carries, x, **kwargs):
        """Call the MultiLayerRNN with the given carries and input x."""
        if self.fa_type == "bp":
            return MultiLayerRNN.__call__(self, carries, x, **kwargs)
        elif self.fa_type in ["fa", "dfa"]:
            Bs = [
                self.variable(
                    "falign",
                    f"B{i}",
                    self.kernel_init,
                    self.make_rng() if self.has_rng("params") else None,
                    (size, self.sizes[-1]),
                ).value
                for i, size in enumerate(self.sizes[:-1])
            ]
        else:
            raise ValueError("unknown fa_type: " + self.fa_type)

        def f(mdl, _carries, x, _Bs):
            return MultiLayerRNN.__call__(mdl, _carries, x, **kwargs)

        def fwd(mdl: MultiLayerRNN, _carries, x, _Bs):
            """Forward pass with tmp for backward pass."""
            vjps = []
            for i, rnn in enumerate(mdl.sizes):
                (_carries[i], x), vjp_func = jax.vjp(
                    rnn.apply, rnn.variables, _carries[i], x, **kwargs
                )
                vjps.append(vjp_func)

            return (_carries, x), (vjps, _Bs)

        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            vjps, _Bs = tmp
            grads = zeros_like_tree(self.variables["params"])
            grads_inputs = []
            for i in range(len(self.sizes)):
                y_t = _Bs[i] @ y_bar[1] if i < len(self.sizes) - 1 else y_bar[1]
                params_t, *inputs_t = vjps[i]((y_bar[0][i], y_t))
                grads[f"rnn_{i}"] = params_t["params"]
                grads_inputs.append(inputs_t[0])
                if i == 0:
                    x_grad = inputs_t[1]

            return ({"params": grads}, grads_inputs, x_grad, zeros_like_tree(_Bs))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)

        return fa_grad(self, carries, x, Bs)


class BlockWrapper(nn.RNNCellBase):
    """Wrapper for a blocked RNN submodel with input chunking and vmap processing.

    Attributes:
        rnn_submodel: RNN submodel to wrap.
        num_blocks: Number of input chunks for parallel processing.
    """

    size: int  # Size of the RNN layer
    rnn_cls: type[nn.RNNCellBase]  # RNN class
    num_blocks: int  # Number of chunks to split the input into
    rnn_kwargs: dict = field(
        default_factory=dict
    )  # Additional arguments for the RNN class

    def setup(self):
        self.block_rnns = self._make_rnn()

    @nn.nowrap
    def _make_rnn(self):
        """Create the RNN submodel."""
        return nn.vmap(
            self.rnn_cls,
            variable_axes={"params": 0, "wiring": 0},  # Separate parameters per chunk
            split_rngs={"params": True, "dropout": True, "wiring": True},
            in_axes=-2,
            out_axes=-2,
            axis_size=self.num_blocks,
            # methods=["__call__", "initialize_carry"],
        )(self.size // self.num_blocks, **self.rnn_kwargs)

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry (hidden state) for the RNN submodel."""
        assert input_shape[-1] % self.num_blocks == 0, (
            "input dimension must be divisible by num_blocks."
        )
        block_shape = (*input_shape[:-1], input_shape[-1] // self.num_blocks)
        rng = jax.random.split(rng, self.num_blocks)
        # Block shape is the same for all blocks
        _rnn = self.rnn_cls(self.size // self.num_blocks, **self.rnn_kwargs)
        init_fn = partial(_rnn.initialize_carry, input_shape=block_shape)
        # TODO: Since vmap init does not work, Try looping over blocks and concatenate
        return jax.vmap(init_fn, in_axes=-2, out_axes=-2)(rng)

    @nn.compact
    def __call__(self, carry, x, **kwargs):
        """Split input, process with vmap over RNN submodel, and combine output."""
        # Split input into chunks and stack along a new dimension
        x_split = jnp.reshape(
            x, (*x.shape[:-1], self.num_blocks, x.shape[-1] // self.num_blocks)
        )

        # Vmap over the RNN submodel
        carry, y = self.block_rnns(carry, x_split, **kwargs)

        # Combine output chunks
        y_combined = y.reshape(x.shape)
        return carry, y_combined


def make_rnn(
    rnn_cls: type[nn.RNNCellBase],
    size: int,
    num_blocks: int = 1,
    name: str = None,
    **kwargs,
) -> nn.RNNCellBase:
    """Create an RNN module with optional input chunking.

    Parameters:
        rnn_cls: Class of the RNN cell to instantiate.
        size: Size of the RNN layer.
        num_blocks: Number of input chunks for parallel processing.
        name: Name of the RNN module.
        **kwargs: Additional arguments for the RNN class.

    Returns:
        An instance of the specified RNN class or an RNNWrapper if num_blocks > 1.
    """
    if num_blocks > 1:
        return BlockWrapper(
            size=size,
            rnn_cls=rnn_cls,
            num_blocks=num_blocks,
            name=f"{name}_block" if name else None,
            rnn_kwargs=kwargs,
        )
    else:
        return rnn_cls(size, name=name, **kwargs)


class RNNEnsemble(nn.RNNCellBase):
    """Ensemble of RNN cells with optional output processing.

    Attributes:
        config: Configuration for the RNN ensemble.
    """

    config: RNNEnsembleConfig

    def setup(self):
        """Initialize submodules."""
        # TODO: Use BlockWrapper for ensemble
        self.ensembles = [
            FAMultiLayerRNN(
                self.config.layers,
                rnn_cls=CELL_TYPES[self.config.model_name],
                rnn_kwargs=self.config.rnn_kwargs,
                num_blocks=self.config.num_blocks,
                fa_type=self.config.fa_type,
                name=f"ensembles_{i}",
                layer_config=self.config.layer_config,  # Use the config directly
            )
            for i in range(self.config.num_modules)
        ]
        if self.config.output_layers:
            self.mlps_out = [
                MLP(
                    self.config.output_layers,
                    f_align=self.config.rnn_kwargs.get("f_align", False),
                    name=f"mlps_{i}",
                )
                for i in range(self.config.num_modules)
            ]
        if self.config.out_size is not None:
            # Make distribution for each submodule
            self.dists = [
                DistributionLayer(
                    self.config.out_size, self.config.out_dist, name=f"dists_{i}"
                )
                for i in range(self.config.num_modules)
            ]

    def _postprocessing(self, outs, x):
        for i in range(self.config.num_modules):
            out = outs[i]
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
            outs = distrax.MixtureSameFamily(
                distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs
            )
        return outs

    def __call__(
        self, h: jax.Array | None = None, x: jax.Array = None, rng=None, **call_args
    ):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters:
        h : List
            of rnn submodule states
        x : Array
            input
        rng : PRNGKey, optional,
            if given, returns one value per submodule in order to train them independently,
            If None, mean of submodules or a Mixed Distribution is returned.

        Returns:
        _type_
            _description_
        """
        if h is None:
            h = self.initialize_carry(jax.random.key(0), x.shape)
        outs = []
        carry_out = []
        for i in range(self.config.num_modules):
            # loop over rnn submodules
            carry, out = self.ensembles[i](h[i], x, **call_args)
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
        """Compute loss and gradient asynchronuously."""
        # Compute gradient of loss wrt to network output
        loss, (grads, df_dh) = jax.value_and_grad(self.apply, argnums=[0, 1])(
            params,
            [c[-1][0].real for c in carry],
            loss_fn,
            target,
            x,
            method=self._loss,
        )

        # Compute parameter gradients for each RNN
        for i, (c, d) in enumerate(zip(carry, df_dh)):
            grads["params"][f"layers_{i}"] = CELL_TYPES[
                self.config.model_name
            ].rtrl_gradient(c, d, plasticity=self.config.rnn_kwargs["plasticity"])[0]
        return loss, grads

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return [
            FAMultiLayerRNN(
                self.config.layers,
                num_blocks=self.config.num_blocks,
                rnn_cls=CELL_TYPES[self.config.model_name],
                rnn_kwargs=self.config.rnn_kwargs,
            ).initialize_carry(rng, input_shape)
            for _ in range(self.config.num_modules)
        ]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @staticmethod
    def clip_tau(params):
        """HACK: clip tau to >= 1.0."""
        return set_matching_leaves(
            params,
            ".*tau.*",
            jax.tree.map(
                partial(jnp.clip, min=1.0), get_matching_leaves(params, ".*tau.*")
            ),
        )


def make_batched_model(model):
    """Parallelize model across a batch of input sequences using vmap.

    Parameters:
        model: Model to parallelize.
        batch_size: Size of the batch.

    Returns:
        A batched version of the model.
    """
    return nn.vmap(
        model,
        in_axes=0,
        out_axes=0,
        variable_axes={
            "params": None,
            "falign": None,
            "mask": None,
            "wiring": None,
        },
        methods=["__call__"],
        split_rngs={"params": False, "dropout": True},
    )
