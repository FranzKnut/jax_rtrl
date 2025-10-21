import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import numpy as np
import optax

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from jax_rtrl.models.jax_util import mse_loss
from jax_rtrl.supervised.training_utils import train_rnn_offline as train
from models.cells import CTRNNCell
from models import FADense

from jax_rtrl.supervised.datasets import sine

key = jrand.PRNGKey(0)
key, key_model, key_data, key_train = jrand.split(key, 4)
x, y = sine()

model = nn.Sequential(
    [
        nn.RNN(CTRNNCell(32)),
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
