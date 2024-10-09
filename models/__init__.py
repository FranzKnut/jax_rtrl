from optimizers import OptimizerConfig, make_optimizer
from .s5 import StackedEncoderModel, make_opt_s5
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
}


def init_model(model: flax.linen.Module, sample_input, is_rnn: bool, rng_key=None):
    rng_key = rng_key or jax.random.PRNGKey(0)
    if is_rnn:
        carry = model.initialize_carry(rng_key, sample_input.shape)
        return model.init(rng_key, carry, sample_input)
    return model.init(rng_key, sample_input)


def make_optimizer_for_model(model_name: str, config: OptimizerConfig):
    """Make optax optimizer for given model name and config."""
    if "s5" in model_name:
        return make_opt_s5(config)
    else:
        return make_optimizer(config)
