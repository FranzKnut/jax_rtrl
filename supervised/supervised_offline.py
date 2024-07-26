import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import flax.linen as nn

import matplotlib.pyplot as plt

from models.custom_grad import FADense
from models.neural_networks import CTRNNCell

key = jrand.PRNGKey(0)
key, key_model, key_data, key_train = jrand.split(key, 4)
x = jnp.linspace(0, 5 * np.pi, 100)[:, None]
y = jnp.sin(x) + 2

cell = CTRNNCell(16)
carry = cell.initialize_carry(jrand.PRNGKey(0), (1,))
params = cell.init(jrand.PRNGKey(0), carry, x[0])

model = nn.Sequential(
    [
        nn.RNN(cell),
        FADense(1),
    ]
)

params = model.init(key_model, x, mutable=True)


def loss(p, __x, __y):
    # MSE loss
    y_hat = model.apply(p, __x)
    return jax.numpy.sum((y_hat - __y) ** 2)


loss(params, x, y)


def print_progress(i, loss):
    if i % 1000 == 0:
        print(f"Iteration {i} | Loss: {loss:.3f}")


def train(_loss_fn, _params, data, _key, num_steps=300_000, lr=1e-4):
    # We use Stochastic Gradient Descent with a constant learning rate
    _x, _y = data
    mask = jax.tree.map(lambda x: True, _params)
    mask["params"]["layers_0"]["cell"]["tau"] = False
    optimizer = optax.adamaxw(lr, mask=mask)
    opt_state = optimizer.init(_params)

    def step(carry, n):
        __params, _opt_state, _key = carry
        # _key, key_batch = jrand.split(_key)
        current_loss, grads = jax.value_and_grad(_loss_fn)(__params, _x, _y)
        updates, _opt_state = optimizer.update(grads, _opt_state, __params)
        __params = optax.apply_updates(__params, updates)
        jax.debug.callback(print_progress, n, current_loss)
        return (__params, _opt_state, _key), current_loss

    (_params, *_), _losses = jax.lax.scan(step, (_params, opt_state, _key), jnp.arange(num_steps, dtype=np.int32))
    print(f"Final loss: {_losses[-1]:.3f}")
    return _params, _losses


key, key_train = jrand.split(key_data)
params, losses = train(loss, params, (x, y), key_train)

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
