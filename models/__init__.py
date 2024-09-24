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
    "lru_rtrl": OnlineLRULayer
}