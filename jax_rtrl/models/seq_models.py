"""Sequence models implemented with Flax."""

# Import necessary modules and libraries
from dataclasses import dataclass, field
from functools import partial
import re
from typing import Literal, Tuple

import distrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
from chex import PRNGKey
from flax import linen as nn

from jax_rtrl.models.cells import CELL_TYPES
from jax_rtrl.models.s5 import S5Config

from .jax_util import zeros_like_tree, get_normalization_fn
from .feedforward import MLP, DistributionLayer, FADense


@dataclass
class SequenceLayerConfig:
    """Configuration for SequenceLayer.

    Attributes:
        d_output: Output layer size.
        dropout: Dropout probability.
        norm: Type of normalization to use ('layer' or 'batch').
        glu: Whether to use Gated Linear Unit structure.
        skip_connection: Whether to use skip connections.
    """

    dropout: float = 0.0
    norm: str | None = None
    glu: bool = True
    skip_connection: bool = True


@dataclass
class RNNEnsembleConfig:
    """Configuration for RNNEnsemble.

    Attributes:
        model_name: Name of the RNN model.
        num_modules: Number of RNN modules in the ensemble.
        layers: Tuple specifying the number of layers in each module.
        method: Method for combining the outputs of the RNN modules.
        num_blocks: Number of input chunks for parallel processing.
        glu: Whether to use Gated Linear Unit structure.
        out_size: Output size of the model.
        out_dist: Type of output distribution.
        input_layers: Configuration for input layers (if any).
        output_layers: Configuration for output layers (if any).
        fa_type: Feedback alignment type ('bp', 'fa', 'dfa').
        ensemble_in_visible_prob: float = 1.0  # If < 1, sample mask from bernoulli for each input element
        ensemble_in_first_full: bool = True  # If True, use full input for the first module
        rnn_kwargs: Additional arguments for the RNN model.
        layer_config: Configuration for the sequence layer.
    """

    model_name: str | None
    layers: tuple[int, ...]
    method: Literal["linear", "dist", None] = None
    num_modules: int = 1
    num_blocks: int = 1
    out_size: int | None = None
    out_dist: str | None = None
    input_layers: tuple[int, ...] | None = None  # TODO
    output_layers: tuple[int, ...] | None = None
    fa_type: str = "bp"
    ensemble_in_visible_prob: float = 1.0
    ensemble_in_first_full: bool = True
    static_rng_seed: int = 0  # Used for ensemble input mask
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
            # print(
            #     f"WARNING specifying wiring does not work with model {self.model_name}. Deleting from kwargs"
            # )


class SequenceLayer(nn.Module):
    """Single layer with normalization, sequence model, dropout, and optional GLU.

    Attributes:
        seq: Sequence module (e.g., RNN, LSTM).
        config: Configuration for the sequence layer.
        training: Whether the model is in training mode (affects dropout behavior).
    """

    rnn_cls: type[nn.RNNCellBase]  # RNN class
    rnn_size: int  # Size of the RNN layer
    d_output: int = None  # output layer size, defaults to hidden_size
    config: SequenceLayerConfig = field(
        default_factory=SequenceLayerConfig
    )  # configuration for the layer
    num_blocks: int = 1  # Number of input chunks for parallel processing
    rnn_kwargs: dict = field(default_factory=dict)  # Additional arguments for the RNN

    def setup(self):
        """Initialize the RNN module."""
        self.seq = make_rnn(
            self.rnn_cls,
            self.rnn_size,
            num_blocks=self.num_blocks,
            **self.rnn_kwargs,
        )

    @nn.compact
    def __call__(self, hidden, inputs, training=True, *args, **kwargs):
        """Apply, layer norm, seq model, dropout and GLU in that order."""

        x = get_normalization_fn(self.config.norm, training=training)(
            inputs
        )  # input normalization
        x = nn.Dropout(
            self.config.dropout, broadcast_dims=[0], deterministic=not training
        )(x)  # input dropout

        # call seq model
        hidden, x = self.seq(hidden, x, *args, **kwargs)
        # hidden = get_normalization_fn(self.config.norm, training=training)(hidden)  # normalize hidden state
        # hidden dropout is not recommended for RNNs
        # TODO: implement zoneout (replacing by previous hidden state)
        # hidden = (drop(hidden[0]), *hidden[1:]) if isinstance(hidden, tuple) else drop(hidden)

        # Optional skip connection
        if self.config.skip_connection:
            x = x + inputs

        x = FADense(self.d_output or hidden.shape[-1])(x)
        if self.config.glu:
            # Gated Linear Unit
            x *= jax.nn.sigmoid(FADense(self.d_output or hidden.shape[-1])(x))

        x = get_normalization_fn(self.config.norm, training=training)(x)
        x = nn.Dropout(
            self.config.dropout, broadcast_dims=[0], deterministic=not training
        )(x)  # input dropout
        return hidden, x

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry (hidden state) for the sequence model."""
        return self.seq.initialize_carry(rng, input_shape)


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
                d_output=size,
                config=self.layer_config,
                name=f"layer_{i}",
            )
            for i, size in enumerate(self.sizes)
        ]

    def setup(self):
        """Initialize submodules."""
        self.layers = self._make_layers()
        self.input = FADense(self.sizes[0], name="input")

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry (hidden state) for the RNN layers."""
        batch_size = input_shape[0:1] if len(input_shape) > 1 else ()
        shapes = [self.sizes[:1], *[(s,) for s in self.sizes[:-1]]]
        # Create a list of tuples of initial states for each layer
        states = [
            _l.initialize_carry(rng, batch_size + in_size)
            for _l, in_size in zip(self.layers, shapes)
        ]
        return self._make_tuple_of_list_from_carries(states)

    def _make_tuple_of_list_from_carries(self, carries):
        """Convert list of tuples to a tuple of lists."""
        if isinstance(carries[0], tuple):
            return tuple([s[i] for s in carries] for i in range(len(carries[0])))
        else:
            return (carries,)

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @nn.compact
    def __call__(self, carries, x, training=True, *args, **kwargs):
        """Call MLP."""
        x = self.input(x)
        new_carries = []
        for i, rnn in enumerate(self.layers):
            _carry = (
                carries[0][i] if len(carries) <= 1 else tuple(c[i] for c in carries)
            )
            _carry, x = rnn(_carry, x, training, *args, **kwargs)
            new_carries.append(_carry)
        return self._make_tuple_of_list_from_carries(new_carries), x


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

    def setup(self):
        """Initialize submodules."""
        self.layers = self._make_layers()
        self.input = FADense(
            self.sizes[0], f_align=self.fa_type in ["fa", "dfa"], name="input"
        )

    @nn.compact
    def __call__(self, carries, x, training=True, *args, **kwargs):
        """Call the MultiLayerRNN with the given carries and input x."""
        if self.fa_type == "bp":
            return MultiLayerRNN.__call__(self, carries, x, training, *args, **kwargs)
        elif self.fa_type in ["fa", "dfa"]:
            Bs = [
                self.variable(
                    "falign",
                    f"B{i}",
                    self.kernel_init,
                    self.make_rng() if self.has_rng("params") else None,
                    (size, self.sizes[-1]),
                ).value
                for i, size in enumerate(self.sizes)
            ]
        else:
            raise ValueError("unknown fa_type: " + self.fa_type)

        def f(mdl, _carries, x, _Bs):
            return MultiLayerRNN.__call__(mdl, _carries, x, *args, **kwargs)

        def fwd(mdl: FAMultiLayerRNN, _carries, x, _Bs):
            """Forward pass with tmp for backward pass."""
            vjps = []
            x, vjp_in = nn.vjp(FADense.__call__, mdl.input, x)
            _new_carries = []
            for i, rnn in enumerate(mdl.layers):
                _carry = (
                    tuple(c[i] for c in _carries)
                    if isinstance(_carries, tuple)
                    else _carries[i]
                )
                (_carry, x), vjp_func = nn.vjp(SequenceLayer.__call__, rnn, _carry, x)
                _new_carries.append(_carry)
                vjps.append(vjp_func)

            return (mdl._make_tuple_of_list_from_carries(_new_carries), x), (
                vjp_in,
                vjps,
                _Bs,
            )

        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            vjp_in, vjps, _Bs = tmp
            y_t = y_bar[1]  # Initial output for the first layer
            grads = {}
            grads_inputs = []
            for i in range(len(self.sizes)):
                params_t, *inputs_t = vjps[i]((y_bar[0][i], y_t))
                grads[f"layer_{i}"] = params_t["params"]
                grads_inputs.append(inputs_t[0])
                if self.fa_type == "dfa":
                    # Direct feedback alignment
                    y_t = _Bs[i] @ y_bar[1]
                elif self.fa_type == "fa":
                    # Feedback alignment
                    y_t = _Bs[i] @ y_t

            in_grad, x_grad = vjp_in(y_t)
            grads["input"] = in_grad["params"]

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
        self.block_rnns = nn.vmap(
            self.rnn_cls,
            variable_axes={"params": 0, "wiring": 0},  # Separate parameters per chunk
            split_rngs={"params": True, "dropout": True, "wiring": True},
            in_axes=-2,
            out_axes=-2,
            axis_size=self.num_blocks,
            # methods=["__call__", "initialize_carry"],
        )(self.size // self.num_blocks, **self.rnn_kwargs)

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the carry (hidden state) for the RNN submodel."""
        assert input_shape[-1] % self.num_blocks == 0, (
            "input dimension must be divisible by num_blocks."
        )
        block_shape = (*input_shape[:-1], input_shape[-1] // self.num_blocks)
        rng = jax.random.split(rng, self.num_blocks)
        # Block shape is the same for all blocks
        return jax.vmap(
            partial(self.block_rnns.initialize_carry, input_shape=(block_shape)),
            out_axes=-2,
        )(rng)

    @nn.compact
    def __call__(self, carry, x, *args, **kwargs):
        """Split input, process with vmap over RNN submodel, and combine output."""
        # Split input into chunks and stack along a new dimension
        x_split = jnp.reshape(
            x, (*x.shape[:-1], self.num_blocks, x.shape[-1] // self.num_blocks)
        )

        # Give every block the same args
        (args, kwargs) = jax.tree.map(
            lambda a: jnp.tile(a, (self.num_blocks, 1)),
            (args, kwargs),
        )

        # Vmap over the RNN submodel
        carry, y = self.block_rnns(carry, x_split, *args, **kwargs)

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
    num_submodule_extra_args: int = 0  # set to number of additional args for submodule!

    def setup(self):
        """Initialize submodules."""
        self.ensembles = make_batched_model(
            FAMultiLayerRNN,
            split_rngs=True,
            # [h, x, training, (*call_args, **call_kwargs)]
            in_axes=(0, 0, None) + (None,) * self.num_submodule_extra_args,
            axis_size=self.config.num_modules,
        )(
            self.config.layers,
            rnn_cls=CELL_TYPES[self.config.model_name],
            rnn_kwargs=self.config.rnn_kwargs,
            num_blocks=self.config.num_blocks,
            fa_type=self.config.fa_type,
            name="ensemble",
            layer_config=self.config.layer_config,  # Use the config directly
        )

        if self.config.num_modules > 1:
            # Since we don't know the input shape yet, we just fix the rng here for creating the mask later
            self.ensemble_input_mask_rng = jrandom.PRNGKey(self.config.static_rng_seed)

        if self.config.output_layers:
            self.mlps_out = make_batched_model(
                MLP,
                split_rngs=True,
                axis_size=self.config.num_modules,
            )(
                self.config.output_layers,
                f_align=self.config.rnn_kwargs.get("f_align", False),
                name="mlps_out",
            )

        if self.config.method == "linear":
            self.combine_layer = FADense(
                self.config.num_modules,
                f_align=self.config.rnn_kwargs.get("f_align", False),
                name="combine_layer",
            )

        # Make distribution for each submodule
        self.dists = make_batched_model(
            DistributionLayer,
            # Last dim is batch in distrax
            split_rngs=True,
            out_axes=-1,
            axis_size=self.config.num_modules,
        )(
            self.config.out_size,
            distribution=self.config.out_dist,
            norm=self.config.layer_config.norm,
            name="dists",
        )

    def _postprocessing(self, outs, x):
        # Output FF layers
        if self.config.output_layers:
            outs = self.mlps_out(outs)

        # Combine Ensemble predictions
        if self.config.out_size is None or self.config.method is None:
            if self.config.out_size is not None or self.config.method is not None:
                print("WARNING: out_size or method is None, skipped output processing.")
            return outs

        else:
            _dists = self.dists(outs)
            if self.config.method == "linear":
                # Compute linear combination of outputs
                out_gates = self.combine_layer(
                    jnp.concatenate([outs.flatten(), x], axis=-1)
                )
                out_gates = jax.nn.softmax(out_gates, axis=-1)
                combined_dist = jax.tree.map(
                    lambda d: jnp.sum(d * out_gates[None], axis=-1), _dists
                )
            elif self.config.method == "dist":
                combined_dist = distrax.MixtureSameFamily(
                    distrax.Categorical(logits=jnp.zeros(_dists.loc.shape)), _dists
                )
            else:
                # Do not combine, return all distributions
                return _dists
        return combined_dist, _dists

    def __call__(
        self,
        h: jax.Array | None = None,
        x: jax.Array = None,
        training: bool = True,
        *call_args,
        **call_kwargs,
    ):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters:
        h : List
            of rnn submodule states
        x : Array
            input
        training : bool
            whether the model is in training mode (affects dropout behavior)
        call_args : tuple
            additional arguments for the RNN submodules
        call_kwargs : dict
            additional keyword arguments for the RNN submodules

        Returns:
        _type_
            _description_
        """
        # Mask x for ensemble modules
        x_tiled = x[None] * jnp.ones((self.config.num_modules,) + (1,) * (x.ndim))
        if self.config.num_modules > 1:
            ensemble_input_mask = jax.random.bernoulli(
                self.ensemble_input_mask_rng,
                self.config.ensemble_in_visible_prob,
                (self.config.num_modules,) + x.shape,
            )
            if self.config.ensemble_in_first_full:
                ensemble_input_mask.at[0].set(1.0)
            x_tiled = x_tiled * ensemble_input_mask
        elif self.config.ensemble_in_visible_prob < 1.0:
            print("WARNING: num_modules is 1 so ensemble_in_visible_prob is ignored.")

        h = h or self.initialize_carry(jax.random.key(0), x_tiled.shape)

        # call rnn submodules
        carry_out, outs = self.ensembles(
            h, x_tiled, training, *call_args, **call_kwargs
        )

        # Post-process and aggregate outputs
        outs = self._postprocessing(outs, x)

        return carry_out, outs

    def _loss(self, hidden, loss_fn, target, x):
        # Post-process and aggregate outputs
        outs = self._postprocessing(hidden, x)
        return loss_fn(outs, target)

    def loss_and_grad_async(self, loss_fn, carry, hidden, x, target):
        """Compute loss and gradient asynchronuously."""
        # Compute gradient of loss wrt to network output
        loss, (grads, df_dh) = jax.value_and_grad(self._loss, argnums=[0, 1])(
            [c[-1][0].real for c in carry],
            loss_fn,
            target,
            x,
            method=self._loss,
        )

        # Compute parameter gradients for each RNN
        # FIXME
        for i, (c, d) in enumerate(zip(carry, df_dh)):
            grads["params"][f"layers_{i}"] = CELL_TYPES[
                self.config.model_name
            ].rtrl_gradient(c, d, plasticity=self.config.rnn_kwargs["plasticity"])[0]
        return loss, grads

    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        # return self.ensembles.initialize_carry(rng, input_shape)
        return jax.vmap(
            self.ensembles.initialize_carry,
            in_axes=(None, None),
            out_axes=len(input_shape) - 1,
            axis_size=self.config.num_modules,
        )(rng, input_shape)

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


def make_batched_model(
    model,
    split_rngs: bool = False,
    in_axes: int | tuple[int, ...] = 0,
    out_axes: int | tuple[int, ...] = 0,
    axis_size=None,
    methods: list[str] = None,
):
    """Parallelize model across a batch of input sequences using vmap.

    Parameters:
        model: Model to parallelize.
        batch_size: Size of the batch.

    Returns:
        A batched version of the model.
    """
    if split_rngs:
        return nn.vmap(
            model,
            variable_axes={
                "params": 0,
                "wiring": 0,
                "falign": 0,
            },  # Separate parameters per chunk
            split_rngs={"params": True, "dropout": True, "wiring": True},
            in_axes=in_axes,
            out_axes=out_axes,
            axis_size=axis_size,
            methods=methods,
        )
    return nn.vmap(
        model,
        in_axes=in_axes,
        out_axes=out_axes,
        variable_axes={
            "params": None,
            "falign": None,
            "mask": None,
            "wiring": None,
        },
        methods=["__call__"],
        split_rngs={"params": False, "dropout": True},
    )
