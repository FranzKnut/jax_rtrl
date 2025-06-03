import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import optax
from jax_rtrl.models.seq_models import RNNEnsembleConfig
from jax_rtrl.supervised.datasets import sine

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.seq_models import RNNEnsemble
from supervised.training_utils import predict
from supervised.training_utils import train_rnn_online as train


# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)


class Model(nn.Module):
    outsize: int
    out_dist: str = None  # 'Normal'
    hidden_size: int = 32
    num_modules: int = 1
    dt: float = 1
    model_name: str = "lru_rtrl"

    def setup(self):
        if self.model_name in ["rtrl", "rflo"]:
            kwargs = {"dt": self.dt, "plasticity": self.model_name}
        else:
            kwargs = {}
        self.rnn = RNNEnsemble(
            RNNEnsembleConfig(
                model_name=self.model_name,
                layers=(self.hidden_size,) * 2,
                out_size=self.outsize,
                num_modules=self.num_modules,
                out_dist=self.out_dist,
                rnn_kwargs=kwargs,
                output_layers=None,
                fa_type="bp",
            ),
            name="rnn",
        )

    @nn.compact
    def __call__(self, x, carry=None, key=None):
        if carry is None:
            carry = self.rnn.initialize_carry(key, x.shape)
            key = jrand.fold_in(key, key[0])
        carry, out = self.rnn(carry, x)
        return carry, out

    def initialize_carry(self, key, input_shape):
        return self.rnn.initialize_carry(key, input_shape)


def make_model(initial_input, key, kwargs={}):
    key, key_model = jrand.split(key)
    model = Model(1, **kwargs)
    params = model.init(key_model, initial_input, key=key_model)
    h0 = model.apply(
        params, key_model, initial_input.shape[-1:], method=model.initialize_carry
    )
    return model, params, h0


if __name__ == "__main__":
    key = jrand.PRNGKey(1)
    key, key_data, key_train = jrand.split(key, 3)

    x, y = sine()

    model, params, h0 = make_model(x[0], key)

    def loss(p, __x, __y, rnn_state=None):
        # MSE loss
        rnn_state, y_hat = model.apply(p, __x, rnn_state)
        if model.out_dist is None:
            loss = jnp.mean((__y - y_hat) ** 2)
        else:
            loss = jnp.mean(-y_hat.log_prob(__y))
        return loss, rnn_state

    optimizer = optax.adam(1e-4)

    params, losses = train(
        loss,
        optimizer,
        params,
        (x, y),
        key_train,
        h0,
    )

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)

    y_hat = predict(model, params, x)
    print(f"Final loss: {jnp.mean((y - y_hat) ** 2):.3f}")
    plt.plot(x, y, label="target")
    plt.plot(x, y_hat.squeeze(), label="trained")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/sinewave.png")
    plt.show()
