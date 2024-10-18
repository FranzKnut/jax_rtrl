import re

from .neural_networks import RNNEnsembleConfig
from .s5 import StackedEncoderModel, S5Config
import flax
import flax.linen
import jax
from models.lru import OnlineLRULayer
from .ctrnn import CTRNNCell, OnlineCTRNNCell

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
    rng_key = rng_key or jax.random.PRNGKey(0)
    if is_rnn:
        carry = model.initialize_carry(rng_key, sample_input.shape)
        return model.init(rng_key, carry, sample_input)
    return model.init(rng_key, sample_input)


def make_rnn_ensemble_config(
    model_name,
    hidden_size,
    out_size,
    num_modules=1,
    num_layers=1,
    stochastic=False,
    model_kwargs=None,
    output_layers=None,
    skip_connection=False,
):
    model_cls = CELL_TYPES[model_name]
    kwargs = model_kwargs or {}
    match = re.search(r"(rflo|rtrl)", model_name)
    if match:
        kwargs["plasticity"] = match.group(1)
    elif model_name in ["s5", "s5_rtrl"]:
        kwargs = {"config": S5Config(**kwargs)}
    return RNNEnsembleConfig(
        layers=(hidden_size,) * num_layers,
        out_size=out_size,
        num_modules=num_modules,
        out_dist="Normal" if stochastic else None,
        rnn_kwargs=kwargs,
        output_layers=output_layers,
        skip_connection=skip_connection,
        model=model_cls,
    )
