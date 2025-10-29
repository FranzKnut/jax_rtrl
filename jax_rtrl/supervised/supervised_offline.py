from dataclasses import dataclass, field, replace
import os
import sys

import jax.random as jrand
import matplotlib.pyplot as plt
import optax
import simple_parsing

from jax_rtrl.models.seq_models import (
    RNNEnsemble,
    RNNEnsembleConfig,
    SequenceLayerConfig,
    make_batched_model,
    scan_rnn,
)


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from jax_rtrl.models.jax_util import mse_loss
from jax_rtrl.supervised.training_utils import (
    get_data,
    make_model,
    predict,
    train_rnn_offline as train,
)

# jax.config.update("jax_disable_jit", True)


@dataclass
class TrainingConfig:
    dataset: str = "legacy_rollouts"
    # dataset: str = "sine"
    # dataset: str = "spirals"
    num_steps: int = 10_000
    learning_rate: float = 3e-4
    gradient_clip: float | None = None

    rnn_config: RNNEnsembleConfig = field(
        default_factory=lambda: RNNEnsembleConfig(
            model_name="bptt",
            # model_name="lrc_snap0",
            # model_name="ltc_rtrl",
            # model_name="lrc_rtrl",
            layers=(32,),
            num_modules=1,
            num_blocks=1,
            layer_config=SequenceLayerConfig(
                norm="layer",
                glu=True,
                # skip_connection=True,
            ),
            out_dist="Deterministic",
            rnn_kwargs={
                # "dt": 1.0,
                # "ode_type": "murray",
            },
            output_layers=None,
            fa_type="bp",
        )
    )


cfg = simple_parsing.parse(TrainingConfig)

key = jrand.PRNGKey(0)
key, key_data, key_train = jrand.split(key, 3)

x_train, y_train, x_test, y_test = get_data(cfg.dataset)

rnn_config = replace(cfg.rnn_config, out_size=y_train.shape[-1])
model, params, h0 = make_model(x_train[0], key, rnn_config)

model = make_batched_model(RNNEnsemble)(rnn_config)
params = model.init(key, None, x_train[0])


def loss(p, __x, __y):
    # MSE loss
    _, y_hat = scan_rnn(model, p, None, __x)
    return mse_loss(y_hat.mode().squeeze(), __y)


key, key_train = jrand.split(key_data)
optimizer = optax.adam(1e-3)
params, losses = train(
    loss,
    optimizer,
    params,
    (x_train, y_train),
    key_train,
    num_steps=cfg.num_steps,
)

plt.figure(figsize=(10, 5))

# Plot the training loss
plt.subplot(1, 2, 1)
plt.plot(losses)

# Plot the trained model output
plt.subplot(1, 2, 2)

y_hat = predict(model, params, x_test[:, None] if x_test.ndim == 2 else x_test)
y_hat = y_hat.mode().squeeze()
test_loss = mse_loss(y_hat, y_test)
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
    plt.plot(y_test[0, ..., 0], label="target")
    plt.plot(y_hat[0, ..., 0], label="trained")
    plt.legend()
    os.makedirs("plots/supervised", exist_ok=True)
    plt.savefig(f"plots/supervised/{cfg.dataset}_{cfg.rnn_config.model_name}.png")
    plt.show()
