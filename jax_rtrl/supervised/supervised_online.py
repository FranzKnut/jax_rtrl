from dataclasses import dataclass, field, replace
import os
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import optax
import simple_parsing
from jax_rtrl.models.cells.ctrnn import clip_tau
from jax_rtrl.models.seq_models import RNNEnsembleConfig
from jax_rtrl.supervised.datasets import sine

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.seq_models import RNNEnsemble
from supervised.training_utils import predict
from supervised.training_utils import train_rnn_online as train


# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)


@dataclass
class TrainingConfig:
    rnn_config: RNNEnsembleConfig = field(
        default_factory=lambda: RNNEnsembleConfig(
            model_name="rtrl",
            layers=(32,),
            num_modules=1,
            num_blocks=1,
            out_dist="Deterministic",
            rnn_kwargs={"dt": 0.2},
            output_layers=None,
            fa_type="bp",
            method="linear",
        )
    )


cfg = simple_parsing.parse(TrainingConfig)


class Model(nn.Module):
    out_size: int
    rnn_config: RNNEnsembleConfig

    def setup(self):
        _config = replace(self.rnn_config, out_size=self.out_size)
        self.rnn = RNNEnsemble(_config, name="rnn")

    @nn.compact
    def __call__(self, x, carry=None, key=None):
        carry, out = self.rnn(carry, x)
        return carry, out

    def initialize_carry(self, key, input_shape):
        return self.rnn.initialize_carry(key, input_shape)


def make_model(initial_input, key, kwargs: RNNEnsembleConfig):
    key, key_model = jrand.split(key)
    model = Model(1, kwargs)
    params = model.init(key_model, initial_input, key=key_model)
    h0 = model.apply(
        params, key_model, initial_input.shape[-1:], method=model.initialize_carry
    )
    return model, params, h0


if __name__ == "__main__":
    key = jrand.PRNGKey(1)
    key, key_data, key_train = jrand.split(key, 3)

    x, y = sine()

    model, params, h0 = make_model(x[0], key, cfg.rnn_config)

    def loss(p, __x, __y, rnn_state=None):
        # MSE loss
        rnn_state, y_hat = model.apply(p, __x, rnn_state)
        if model.ensemble_method is not None:
            y_hat = y_hat[0]
        if model.out_dist == "Deterministic":
            loss = jnp.mean((__y - y_hat.mode()) ** 2)
        else:
            loss = jnp.mean(-y_hat.log_prob(__y))
        return loss, rnn_state

    optimizer = optax.adam(LR)

    params, losses = train(
        loss, optimizer, params, (x, y), key_train, h0, param_post_update_fn=clip_tau
    )

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)

    y_hat = predict(model, params, x)
    if model.ensemble_method is not None:
        y_hat = y_hat[0]
    y_hat = y_hat.mode()
    print(f"Final loss: {jnp.mean((y - y_hat) ** 2):.3f}")
    plt.plot(x, y, label="target")
    plt.plot(x, y_hat.squeeze(), label="trained")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/sinewave.png")
    plt.show()
