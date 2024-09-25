import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import flax.linen as nn

import matplotlib.pyplot as plt

from jax_rtrl.models.ctrnn.ctrnn import OnlineCTRNNCell
from jax_rtrl.models.lru.online_lru import OnlineLRULayer
from jax_rtrl.optimizers import OptimizerConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from jax_rtrl.models import CELL_TYPES
from models.neural_networks import RNNEnsemble, RNNEnsembleConfig
from supervised.training_utils import predict, train_rnn_online as train

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)


class Model(nn.Module):
    outsize: int
    out_dist: str = None  # 'Normal'
    hidden_size: int = 32
    num_modules: int = 1
    dt: float = 1
    plasticity: str = "rflo"

    @nn.nowrap
    def _make_rnn(self):
        model_cls = CELL_TYPES[self.plasticity]
        if model_cls == OnlineCTRNNCell:
            kwargs = {"dt": self.dt, "plasticity": self.plasticity}
        elif model_cls == OnlineLRULayer:
            kwargs = {"d_output": self.outsize}
        return RNNEnsemble(
            RNNEnsembleConfig(
                layers=(self.hidden_size,) * 2,
                model=CELL_TYPES[self.plasticity],
                out_size=self.outsize,
                num_modules=self.num_modules,
                out_dist=self.out_dist,
                rnn_kwargs=kwargs,
                output_layers=None,
                fa_type="dfa"
            ),
            name="rnn",
        )

    @nn.compact
    def __call__(self, x, carry=None, key=None):
        cell = self._make_rnn()
        if carry is None:
            carry = cell.initialize_carry(key, x.shape)
            key = jrand.fold_in(key, key[0])
        carry, out = cell(carry, x)
        return carry, out

    @nn.nowrap
    def initialize_carry(self, key, input_shape):
        return self._make_rnn().initialize_carry(key, input_shape)


def make_model(initial_input, key, kwargs={}):
    key, key_model = jrand.split(key)
    model = Model(1, **kwargs)
    params = model.init(key_model, initial_input, key=key_model)
    h0 = model.initialize_carry(key_model, initial_input.shape[-1:])
    return model, params, h0


if __name__ == "__main__":
    key = jrand.PRNGKey(1)
    key, key_train = jrand.split(key)

    x = jnp.linspace(0, 5 * np.pi, 100)[:, None]
    y = jnp.sin(x) + 2

    model, params, h0 = make_model(x[0], key)

    def loss(p, __x, __y, rnn_state=None):
        # MSE loss
        rnn_state, y_hat = model.apply(p, __x, rnn_state)
        if model.out_dist is None:
            loss = jnp.mean((__y - y_hat) ** 2)
        else:
            loss = jnp.mean(-y_hat.log_prob(__y))
        return loss, rnn_state

    params, losses = train(
        loss,
        params,
        (x, y),
        key_train,
        h0,
        opt_config=OptimizerConfig(opt_name="adam", learning_rate=1e-5, gradient_clip=1),
    )

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)

    y_hat = predict(model, params, x)
    print(f"Final loss: {jnp.mean((y-y_hat)**2):.3f}")
    plt.plot(x, y, label="target")
    plt.plot(x, y_hat.squeeze(), label="trained")
    plt.legend()
    plt.show()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/sinewave.png")
