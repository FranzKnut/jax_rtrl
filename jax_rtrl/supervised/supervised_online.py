from dataclasses import dataclass, field
import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib.pyplot as plt
import optax
import simple_parsing
from jax_rtrl.models.cells.ctrnn import clip_tau
from jax_rtrl.util.jax_util import mse_loss
from jax_rtrl.models.seq_models import RNNEnsembleConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from jax_rtrl.supervised.training_utils import make_model
from models.seq_models import SequenceLayerConfig
from supervised.training_utils import get_data, predict
from supervised.training_utils import train_rnn_online as train


jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)


@dataclass
class TrainingConfig:
    # dataset: str = "legacy_rollouts"
    dataset: str = "sine"
    # dataset: str = "spirals"
    learning_rate: float = 1e-4
    gradient_clip: float | None = None
    num_steps: int = 10000

    rnn_config: RNNEnsembleConfig = field(
        default_factory=lambda: RNNEnsembleConfig(
            # model_name="rflo",
            # model_name="snap0",
            # model_name="rtrl",
            model_name="lrc_snap0",
            # model_name="ltc_snap0",
            # model_name="ltc_rtrl",
            # model_name="lrc_rtrl",
            # model_name="ltc_rflo",
            layers=(32,),
            num_modules=1,
            num_blocks=1,
            layer_config=SequenceLayerConfig(
                norm=None,
                glu=False,
                skip_connection=False,
            ),
            out_dist="Deterministic",
            rnn_kwargs={
                "dt": 1.0,
                # "ode_type": "murray",
            },
            output_layers=None,
            fa_type="bp",
            # method="linear",
        )
    )


cfg = simple_parsing.parse(TrainingConfig)

key = jrand.PRNGKey(0)
key, key_data, key_train = jrand.split(key, 3)

x_train, y_train, x_test, y_test = get_data(cfg.dataset)

# Transpose to time dim first
x_train = x_train.transpose(1, 0, 2)
y_train = y_train.transpose(1, 0, 2)

model, params, h0 = make_model(x_train[0], key, y_train.shape[-1], cfg.rnn_config)

# Compute initial loss
y_hat = predict(model, params, x_test[None] if x_test.ndim == 2 else x_test)
if cfg.rnn_config.ensemble_method is not None:
    y_hat = y_hat[0]
y_hat = y_hat.mode().squeeze()
test_loss = mse_loss(y_hat, y_test)
print(f"Initial loss: {test_loss:.3f}")


# @jax.vmap
def loss(p, __x, __y, rnn_state=None):
    # MSE loss
    rnn_state, y_hat = model.apply(p, rnn_state, __x)
    if cfg.rnn_config.ensemble_method is not None:
        y_hat = y_hat[0]
    if cfg.rnn_config.out_dist == "Deterministic":
        loss = mse_loss(y_hat.mode().reshape(__y.shape), __y)
    else:
        loss = jnp.mean(-y_hat.log_prob(__y))
    return loss, rnn_state


# Make optimizer
optimizer = optax.chain(
    # Gradient clipping
    optax.clip_by_block_rms(cfg.gradient_clip)
    if cfg.gradient_clip
    else optax.identity(),
    optax.adam(cfg.learning_rate),
)

params, losses = train(
    loss,
    optimizer,
    params,
    (x_train, y_train),
    key_train,
    h0,
    param_post_update_fn=clip_tau,
    num_steps=cfg.num_steps,
)


plt.figure(figsize=(10, 5))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(losses)

# Plot the trained model output
plt.subplot(1, 2, 2)

y_hat = predict(model, params, x_test[None] if x_test.ndim == 2 else x_test)
if cfg.rnn_config.ensemble_method is not None:
    y_hat = y_hat[0]
y_hat = y_hat.mode()
test_loss = mse_loss(y_hat.squeeze(), y_test)
print(f"Final loss: {test_loss:.3f}")

if cfg.dataset == "spirals":
    for _x, _y in zip(x_test, y_test):
        plt.plot(
            _x[..., 0].T,
            _x[..., 1].T,
            c="y" if _y[..., 0, 0] else "darkblue",
            alpha=0.5,
        )
    plt.scatter(x_test[..., 0], x_test[..., 1], c=y_hat[..., 0])
    plt.title(f"Spirals, loss:{test_loss:.3f}")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/spirals.png")
    plt.show()
elif cfg.dataset == "sine":
    plt.plot(x_test.squeeze(), y_test.squeeze(), label="target")
    plt.plot(x_test.squeeze(), y_hat.squeeze(), label="trained")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/sinewave.png")
    plt.show()
else:
    plt.plot(y_test[:, 0, ..., 0], label="target")
    plt.plot(y_hat[:, 0, ..., 0], label="trained")
    plt.legend()
    os.makedirs("plots/supervised", exist_ok=True)
    plt.savefig(f"plots/supervised/{cfg.dataset}_{cfg.rnn_config.model_name}.png")
    plt.show()
