from jax_rtrl.models.cells.ctrnn import CTRNNCell, OnlineCTRNNCell
from jax_rtrl.models.cells.lrc import LRCCell, OnlineLRCCell
from jax_rtrl.models.cells.lru import OnlineLRULayer
from jax_rtrl.models.cells.ltc import LTCCell, OnlineLTCCell
from jax_rtrl.models.s5 import StackedEncoderModel

ONLINE_CELL_TYPES = {
    "eprop": OnlineCTRNNCell,
    "rflo": OnlineCTRNNCell,
    "snap0": OnlineCTRNNCell,
    "rtrl": OnlineCTRNNCell,
    "lru_rtrl": OnlineLRULayer,
    "s5_rtrl": StackedEncoderModel,
    "ltc_rtrl": OnlineLTCCell,
    "ltc_rflo": OnlineLTCCell,
    "lrc_rtrl": OnlineLRCCell,
    "ltc_snap0": OnlineLTCCell,
    "lrc_snap0": OnlineLRCCell,
}


CELL_TYPES = {
    "bptt": CTRNNCell,
    "lru": OnlineLRULayer,
    "s5": StackedEncoderModel,
    "ltc": LTCCell,
    "lrc": LRCCell,
    # "linear": StackedEncoderModel, # TODO: homogenize StackedEncoderModel and MultiLayerRNN
    # "gru": StackedEncoderModel,
    **ONLINE_CELL_TYPES,
}
