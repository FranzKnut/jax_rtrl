"""Module for model definitions and utilities."""

import re
from turtle import mode

import flax
import flax.linen
import jax

from .ctrnn import CTRNNCell, OnlineCTRNNCell
from .lru import OnlineLRULayer
from .s5 import S5Config, StackedEncoderModel
from .seq_models import RNNEnsembleConfig

CELL_TYPES = {
    "bptt": CTRNNCell,
    "rflo": OnlineCTRNNCell,
    "rtrl": OnlineCTRNNCell,
    "lru": OnlineLRULayer,
    "lru_rtrl": OnlineLRULayer,
    "s5": StackedEncoderModel,
    "s5_rtrl": StackedEncoderModel,
    # "linear": StackedEncoderModel, # TODO: homogenize StackedEncoderModel and MultiLayerRNN
    # "gru": StackedEncoderModel,
}


def init_model(model: flax.linen.Module, sample_input, is_rnn: bool, rng_key=None):
    """Initialize a Flax model with the given sample input and optional random key.

    Args:
        model (flax.linen.Module): The Flax model to be initialized.
        sample_input: A sample input to initialize the model with.
        is_rnn (bool): A flag indicating whether the model is a recurrent neural network (RNN).
        rng_key (optional): A JAX random key for initialization. If not provided, a default key will be used.

    Returns:
        flax.core.FrozenDict: The initialized model parameters.
    """
    rng_key = rng_key or jax.random.PRNGKey(0)
    if is_rnn:
        carry = model.initialize_carry(rng_key, sample_input.shape)
        return model.init(rng_key, carry, sample_input)
    return model.init(rng_key, sample_input)


def make_rnn_ensemble_config(
    model_name,
    hidden_size,
    out_size=None,
    num_modules=1,
    num_layers=1,
    stochastic=False,
    model_kwargs=None,
    output_layers=None,
    skip_connection=False,
    glu=True,
):
    """Make configuration for an RNN ensemble model.

    Args:
        model_name (str): The name of the RNN model type.
        hidden_size (int): The size of the hidden layers.
        out_size (int, optional): The size of the output layer. Defaults to None.
        num_modules (int, optional): The number of modules in the ensemble. Defaults to 1.
        num_layers (int, optional): The number of layers in the RNN. Defaults to 1.
        stochastic (bool, optional): Whether the output distribution is stochastic. Defaults to False.
        model_kwargs (dict, optional): Additional keyword arguments for the model. Defaults to None.
        output_layers (list, optional): List of output layers. Defaults to None.
        skip_connection (bool, optional): Whether to use skip connections. Defaults to False.
        glu (bool, optional): Whether to use Gated Linear Units (GLU). Defaults to True.

    Returns:
        RNNEnsembleConfig: The configuration object for the RNN ensemble model.
    """
    model_cls = CELL_TYPES[model_name]
    kwargs = model_kwargs or {}
    match = re.search(r"(rflo|rtrl)", model_name)
    if match:
        kwargs["plasticity"] = match.group(1)
    elif model_name in ["s5", "s5_rtrl"]:
        kwargs = {"config": S5Config(**kwargs)}
    if model_name not in ["bptt", "rtrl", "rflo"]:
        if "wiring" in kwargs:
            # print(f"WARNING specifying wiring does not work with model {model_name}. Deleting from kwargs")
            del kwargs["wiring"]
        if "wiring_kwargs" in kwargs:
            # print(f"WARNING specifying wiring_kwargs does not work with model {model_name}. Deleting from kwargs")
            del kwargs["wiring_kwargs"]
    return RNNEnsembleConfig(
        layers=(hidden_size,) * num_layers,
        out_size=out_size,
        num_modules=num_modules,
        out_dist="Normal" if stochastic else None,
        rnn_kwargs=kwargs,
        output_layers=output_layers,
        skip_connection=skip_connection,
        model=model_cls,
        glu=glu,
    )
