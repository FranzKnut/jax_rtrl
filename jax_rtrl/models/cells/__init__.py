from functools import partial

from jax_rtrl.models.cells.ctrnn import CTRNNCell, OnlineCTRNNCell
from jax_rtrl.models.cells.lrc import LRCCell, OnlineLRCCell
from jax_rtrl.models.cells.lru import OnlineLRULayer
from jax_rtrl.models.cells.ltc import LTCCell, OnlineLTCCell
from jax_rtrl.models.s5 import StackedEncoderModel
from jax_rtrl.models.feedforward import MLPCell

ONLINE_CELL_TYPES = {
    # CTRNN
    "eprop": partial(OnlineCTRNNCell, plasticity="eprop"),
    "rflo": partial(OnlineCTRNNCell, plasticity="rflo"),
    "snap0": partial(OnlineCTRNNCell, plasticity="snap0"),
    "rtrl": partial(OnlineCTRNNCell, plasticity="rtrl"),
    ## CTRNN Murray
    "murray_eprop": partial(OnlineCTRNNCell, plasticity="eprop", ode_type="murray"),
    "murray_rflo": partial(OnlineCTRNNCell, plasticity="rflo", ode_type="murray"),
    "murray_snap0": partial(OnlineCTRNNCell, plasticity="snap0", ode_type="murray"),
    "murray_rtrl": partial(OnlineCTRNNCell, plasticity="rtrl", ode_type="murray"),
    ## CTRNN Time Gate
    "ctrnn_tg_eprop": partial(OnlineCTRNNCell, plasticity="eprop", ode_type="tg"),
    "ctrnn_tg_rflo": partial(OnlineCTRNNCell, plasticity="rflo", ode_type="tg"),
    "ctrnn_tg_snap0": partial(OnlineCTRNNCell, plasticity="snap0", ode_type="tg"),
    "ctrnn_tg_rtrl": partial(OnlineCTRNNCell, plasticity="rtrl", ode_type="tg"),
    ## CTRNN Tau Sofplus
    "tau_sofplus_eprop": partial(
        OnlineCTRNNCell, plasticity="eprop", ode_type="tau_sofplus",
    ),
    "tau_sofplus_rflo": partial(
        OnlineCTRNNCell, plasticity="rflo", ode_type="tau_sofplus"
    ),
    "tau_sofplus_snap0": partial(
        OnlineCTRNNCell, plasticity="snap0", ode_type="tau_sofplus"
    ),
    "tau_sofplus_rtrl": partial(
        OnlineCTRNNCell, plasticity="rtrl", ode_type="tau_sofplus"
    ),
    # LRU
    "lru_rtrl": partial(OnlineLRULayer, plasticity="rtrl"),
    "s5_rtrl": StackedEncoderModel,
    # LTC
    # TODO: add ODE types for LTC
    "ltc_rtrl": partial(OnlineLTCCell, plasticity="rtrl"),
    "ltc_rflo": partial(OnlineLTCCell, plasticity="rflo"),
    "lrc_rtrl": partial(OnlineLRCCell, plasticity="rtrl"),
    "ltc_snap0": partial(OnlineLTCCell, plasticity="snap0"),
    # LRC
    "lrc_snap0": partial(OnlineLRCCell, plasticity="snap0"),
}


CELL_TYPES = {
    "bptt": CTRNNCell,
    "lru": OnlineLRULayer,
    "s5": StackedEncoderModel,
    "ltc": LTCCell,
    "lrc": LRCCell,
    "mlp": MLPCell,
    # "linear": StackedEncoderModel, # TODO: homogenize StackedEncoderModel and MultiLayerRNN
    # "gru": StackedEncoderModel,
    **ONLINE_CELL_TYPES,
}


def make_cell(cell_type, **kwargs):
    if cell_type in ONLINE_CELL_TYPES:
        return ONLINE_CELL_TYPES[cell_type](**kwargs)
    elif cell_type in CELL_TYPES:
        return CELL_TYPES[cell_type](**kwargs)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")
