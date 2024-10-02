import flax
import flax.linen
import jax
from jax_rtrl.models.lru.lru_bptt import LRULayer
from jax_rtrl.models.lru.online_lru import OnlineLRUCell, OnlineLRULayer
from .ctrnn.ctrnn import CTRNNCell, OnlineCTRNNCell
from .neural_networks import FADense
from .neural_networks import MLP

CELL_TYPES = {
    "ctrnn": CTRNNCell,
    "rflo": OnlineCTRNNCell,
    "rtrl": OnlineCTRNNCell,
    "lru": LRULayer,
    "lru_rtrl": OnlineLRULayer,
}


def init_model(model: flax.linen.Module, sample_input, is_rnn: bool, rng_key=None):
    rng_key = rng_key or jax.random.PRNGKey(0)
    if is_rnn:
        carry = model.initialize_carry(rng_key, sample_input.shape)
        return model.init(rng_key, carry, sample_input)
    return model.init(rng_key, sample_input)
