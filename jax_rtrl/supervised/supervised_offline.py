import os
import sys

import flax.linen as nn
import jax
import jax.random as jrand
import matplotlib.pyplot as plt
import optax

from jax_rtrl.models.cells.ltc import LTCCell
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from jax_rtrl.models.jax_util import mse_loss
from jax_rtrl.supervised.training_utils import train_rnn_offline as train
from models.cells import CTRNNCell
from models import FADense

from jax_rtrl.supervised.datasets import sine

# jax.config.update("jax_disable_jit", True)

key = jrand.PRNGKey(0)
key, key_model, key_data, key_train = jrand.split(key, 4)
x, y = sine()

# config = RNNEnsembleConfig(
#     model_name="bptt",
#     layers=(32,) * 2,
#     out_size=1,
#     num_modules=1,
#     num_blocks=1,
#     out_dist="Deterministic",
#     rnn_kwargs={},
#     output_layers=None,
#     fa_type="bp",
#     method="linear",
# )


model = nn.Sequential(
    [
        nn.RNN(LTCCell(8, dt=0.2)),
        FADense(1),
    ]
)

params = model.init(key_model, x)


def loss(p, __x, __y):
    # MSE loss
    y_hat = model.apply(p, __x)
    return mse_loss(y_hat, __y)


loss(params, x, y)

key, key_train = jrand.split(key_data)
optimizer = optax.adam(1e-3)
params, losses = train(loss, optimizer, params, (x, y), key_train)

plt.figure(figsize=(10, 5))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(losses)

# Plot the trained model output
plt.subplot(1, 2, 2)
y_hat = model.apply(params, x)
plt.plot(x, y, label="target")
plt.plot(x, y_hat, label="trained")
plt.legend()
plt.show()
