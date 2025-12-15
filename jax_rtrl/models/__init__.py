"""Module for model definitions and utilities."""

import re

import flax
import flax.linen
import jax

from jax_rtrl.models.cells import CELL_TYPES, ONLINE_CELL_TYPES  # noqa
from jax_rtrl.models.feedforward import FADense, FAAffine

from .seq_models import RNNEnsembleConfig, SequenceLayerConfig


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
    num_layers=1,
    stochastic=False,
    model_kwargs=None,
    skip_connection=True,
    glu=False,
    f_align=False,
    norm=None,
    dropout=0.0,
    wiring=None,
    **kwargs,
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
        glu (bool, optional): Whether to use Gated Linear Units (GLU). Defaults to False.
        f_align (bool, optional): Whether to use Feedback Alignment. Defaults to False.
        norm (str, optional): Normalization type. Defaults to "layer".
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        TODO: Remove: every

    Returns:
        RNNEnsembleConfig: The configuration object for the RNN ensemble model.
    """
    rnn_kwargs = model_kwargs or {}

    layer_config = SequenceLayerConfig(
        norm=norm,
        dropout=dropout,
        skip_connection=skip_connection,
        glu=glu,
    )

    return RNNEnsembleConfig(
        model_name=model_name,
        layers=(hidden_size,) * num_layers,
        # out_size=out_size,
        out_dist="Normal" if stochastic else "Deterministic",
        rnn_kwargs=rnn_kwargs,
        fa_type="fa" if f_align else "bp",
        layer_config=layer_config,
        **kwargs,
    )
