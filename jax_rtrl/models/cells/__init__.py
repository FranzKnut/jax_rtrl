from jax_rtrl.models.cells.ctrnn import CTRNNCell, OnlineCTRNNCell
from jax_rtrl.models.cells.lru import OnlineLRULayer
from jax_rtrl.models.cells.ltc import LTCCell
from jax_rtrl.models.s5 import StackedEncoderModel

ONLINE_CELL_TYPES = {
    "rflo": OnlineCTRNNCell,
    "rtrl": OnlineCTRNNCell,
    "lru_rtrl": OnlineLRULayer,
    "s5_rtrl": StackedEncoderModel,
}


CELL_TYPES = {
    "bptt": CTRNNCell,
    "lru": OnlineLRULayer,
    "s5": StackedEncoderModel,
    "ltc": LTCCell,
    # "linear": StackedEncoderModel, # TODO: homogenize StackedEncoderModel and MultiLayerRNN
    # "gru": StackedEncoderModel,
    **ONLINE_CELL_TYPES,
}
