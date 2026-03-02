# jax-rtrl

Fast JAX/Flax implementation of online learning rules for recurrent neural networks, including Real-Time Recurrent Learning (RTRL) [1], Random Feedback Local Online Learning (RFLO) [2], and SNAP-0, applied to Continuous-Time RNNs [3], Liquid Time-Constant networks, and Linear Recurrent Units [4].

> If you found this repository useful in your research, please consider citing:

```bibtex
@article{lemmel2024,
  title = {Real-Time Recurrent Reinforcement Learning},
  author = {Lemmel, Julian and Grosu, Radu},
  year = {2024},
  month = mar,
  url = {http://arxiv.org/abs/2311.04830},
  doi = {10.48550/arXiv.2311.04830},
}
```

## Overview

This library provides biologically-plausible, online-capable gradient rules for RNNs. Unlike BPTT, these rules compute parameter gradients at every time step without storing the full history, making them suitable for real-time and continual learning settings.

### Supported Plasticity Rules

| Name | Description |
|---|---|
| `rtrl` | Full Real-Time Recurrent Learning — exact online Jacobian traces |
| `rflo` | Random Feedback Local Online — tractable RTRL approximation using random feedback |
| `snap0` | Immediate Jacobian only (no temporal propagation) — fastest, least accurate |
| `eprop` | E-prop learning rule for CTRNNs |
| `bptt` | Standard Backpropagation Through Time — baseline, not online |

### Supported RNN Cells

| Cell | Keys | Description |
|---|---|---|
| `CTRNNCell` / `OnlineCTRNNCell` | `rflo`, `rtrl`, `eprop`, `snap0`, `bptt` | Continuous-Time RNN |
| `LTCCell` / `OnlineLTCCell` | `ltc_rtrl`, `ltc_rflo`, `ltc_snap0` | Liquid Time-Constant network |
| `LRCCell` / `OnlineLRCCell` | `lrc_rtrl`, `lrc_snap0` | Linear Recurrent Cell |
| `OnlineLRULayer` | `lru_rtrl` | Linear Recurrent Unit with online learning |
| `StackedEncoderModel` | `s5_rtrl`, `s5` | S5 state-space model |

## Installation

Requires Python ≥ 3.10 and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/FranzKnut/jax_rtrl.git
cd jax_rtrl
poetry install
```

For GPU/CUDA support:
```bash
poetry run poe cuda
```

For supervised learning with PyTorch datasets:
```bash
poetry install --with torch_ds
```

## Usage

### Building an Online RNN

```python
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig

config = RNNEnsembleConfig(
    model_name="ltc_rtrl",   # cell type + plasticity rule
    layers=(32,),            # hidden sizes
    out_dist="Categorical",  # output distribution
    dist_eps=0.01,           # unimix epsilon for numerical stability
    rnn_kwargs={"dt": 1.0, "ode_type": "ltc", "plasticity": "rtrl"},
)
model = RNNEnsemble(out_size=2, config=config)
```

### Using a Policy Network

```python
from jax_rtrl.networks.policies import PolicyConfig, PolicyRNN

policy_config = PolicyConfig(
    model_name="rflo",
    layers=(64,),
    out_dist="NormalTanh",          # for continuous actions
    dist_scale_bounds=[0.001, 0.1], # log-std clamping
)
policy = PolicyRNN(a_dim=4, config=policy_config)
```

### Online Gradient Computation

Online cells maintain Jacobian traces in the carry state `(h, jp, jx)`. The carry is initialized with `initialize_carry` and traces are updated automatically each step via the custom VJP defined in `OnlineODECell.solve`.

```python
import jax
import jax.numpy as jnp
from jax_rtrl.models.cells.ltc import OnlineLTCCell

cell = OnlineLTCCell(num_units=32, plasticity="rtrl")
key = jax.random.PRNGKey(0)
carry = cell.initialize_carry(key, (4,))   # includes zero Jacobian traces
params = cell.init(key, carry, jnp.zeros(4))
carry, h = cell.apply(params, carry, x)   # traces updated in-place
```

### Supervised Learning

Scripts for offline and online supervised training are in `jax_rtrl/supervised/`:

```bash
# Online supervised training
python -m jax_rtrl.supervised.supervised_online

# Offline training with PyTorch / HuggingFace datasets
python -m jax_rtrl.supervised.train_torch_ds
```

## Module Reference

```
jax_rtrl/
  models/
    cells/
      ctrnn.py         # Continuous-Time RNN (RFLO, RTRL, e-prop, BPTT)
      ltc.py           # Liquid Time-Constant network
      lrc.py           # Linear Recurrent Cell
      lru.py           # Linear Recurrent Unit
      ode.py           # Base class: OnlineODECell, rtrl(), snap0()
    feedforward.py     # MLP, DistributionLayer, FADense, MLPEnsemble
    seq_models.py      # RNNEnsemble, RNNEnsembleConfig
    plasticity.py      # RTRL, RFLO, e-prop plasticity rule classes
    consolidation.py   # Synaptic Intelligence / weight consolidation
    regularization.py  # Sparsity penalties
  networks/
    policies.py        # PolicyRNN, PolicyConfig
    autoencoders.py    # ConvEncoder, Autoencoder
  supervised/
    supervised_online.py   # Online training loop
    supervised_offline.py  # Offline training loop
    train_torch_ds.py      # Training with HuggingFace/PyTorch datasets
  util/
    checkpointing.py   # Flax checkpoint save/restore
    jax_util.py        # JAX utility functions
```

## References

[1] R. J. Williams and D. Zipser, "A Learning Algorithm for Continually Running Fully Recurrent Neural Networks," *Neural Computation*, vol. 1, no. 2, pp. 270–280, 1989. doi:[10.1162/neco.1989.1.2.270](https://doi.org/10.1162/neco.1989.1.2.270)

[2] J. M. Murray, "Local online learning in recurrent networks with random feedback," *eLife*, vol. 8, p. e43299, 2019. doi:[10.7554/eLife.43299](https://doi.org/10.7554/eLife.43299)

[3] K. Funahashi and Y. Nakamura, "Approximation of dynamical systems by continuous time recurrent neural networks," *Neural Networks*, vol. 6, no. 6, pp. 801–806, 1993. doi:[10.1016/S0893-6080(05)80125-X](https://doi.org/10.1016/S0893-6080(05)80125-X)

[4] N. Zucchet et al., "Online learning of long-range dependencies," *NeurIPS 2023*. [arxiv](https://arxiv.org/abs/2305.15947)

