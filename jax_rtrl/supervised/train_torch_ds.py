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
from jax_rtrl.supervised import example_datasets

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from jax_rtrl.supervised.dataloading import Datasets
from models.seq_models import RNNEnsemble, SequenceLayerConfig, make_batched_model
from supervised.training_utils import predict
from supervised.training_utils import train_rnn_online as train


# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_nans", True)


@dataclass
class TrainingConfig:
    dataset: str = "imdb-classification"
    # dataset: str = "spirals"
    learning_rate: float = 1e-3
    gradient_clip: float | None = None

    rnn_config: RNNEnsembleConfig = field(
        default_factory=lambda: RNNEnsembleConfig(
            # model_name="rtrl",
            model_name="rflo",
            # model_name="ltc_rtrl",
            # model_name="lrc_rtrl",
            layers=(8,),
            num_modules=1,
            num_blocks=1,
            layer_config=SequenceLayerConfig(
                # norm="layer",
                # glu=True,
                # skip_connection=True,
            ),
            out_dist="Deterministic",
            rnn_kwargs={
                "dt": 1.0,
                # "ode_type": "murray",
            },
            output_layers=None,
            fa_type="bp",
            ensemble_method="linear",
        )
    )


class Model(nn.Module):
    rnn_config: RNNEnsembleConfig

    def setup(self):
        self.rnn = RNNEnsemble(self.rnn_config, name="rnn")

    @nn.compact
    def __call__(self, x, carry=None, key=None):
        if carry is None:
            carry = self.rnn.initialize_carry(key, x.shape)
            key = jrand.fold_in(key, key[0])
        carry, out = self.rnn(carry, x)
        return carry, out

    def initialize_carry(self, key, input_shape):
        return self.rnn.initialize_carry(key, input_shape)


def make_model(initial_input, key, kwargs: RNNEnsembleConfig):
    key_model = jrand.split(key, initial_input.shape[0])
    model = make_batched_model(Model, methods=["initialize_carry"])(kwargs)
    params = model.init(key_model[0], initial_input, None, key_model)
    h0 = model.apply(
        params, key_model[0], initial_input.shape, method=model.initialize_carry
    )
    return model, params, h0


if __name__ == "__main__":
    cfg = simple_parsing.parse(TrainingConfig)

    key = jrand.PRNGKey(1)
    key, key_data, key_train = jrand.split(key, 3)

    # x, y = getattr(datasets, cfg.dataset)()
    train_loader, val_loader, test_loader, *_ = Datasets[cfg.dataset]()

    # add missing time and feature dims
    t = x.shape[-2]
    if x.ndim <= 2:
        # Assume single batch
        y = y * jnp.ones((1, 1, 1))
    elif y.ndim == 2:
        assert y.shape[0] == x.shape[0]
        # Assume one label per sequence
        # Repeat along batch and time dimension
        y = y[:, None] * jnp.ones((x.shape[0], t, 1))

    if x.ndim == 2:
        x = x[None]

    batch_size = x.shape[0]
    if batch_size > 1:
        (
            (x_train, y_train),
            (x_test, y_test),
        ) = example_datasets.split_train_test(
            (x, y),
            percent_eval=0.05,
            shuffle=True,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test = x[0], y[0]

    # Transpose to time dim first
    x_train = x_train.transpose(1, 0, 2)
    y_train = y_train.transpose(1, 0, 2)

    rnn_config = replace(cfg.rnn_config, out_size=y.shape[-1])
    model, params, h0 = make_model(x_train[0], key, rnn_config)

    # @jax.vmap
    def loss(p, __x, __y, rnn_state=None):
        # MSE loss
        rnn_state, y_hat = model.apply(p, __x, rnn_state)
        if cfg.rnn_config.ensemble_method is not None:
            y_hat = y_hat[0]
        if cfg.rnn_config.out_dist == "Deterministic":
            loss = jnp.mean((__y - y_hat.mode()) ** 2)
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
    )

    plt.figure(figsize=(10, 5))

    # Plot the training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)

    # Plot the trained model output
    plt.subplot(1, 2, 2)

    y_hat = predict(model, params, x_test[:, None] if x_test.ndim == 2 else x_test)
    if cfg.rnn_config.ensemble_method is not None:
        y_hat = y_hat[0]
    y_hat = y_hat.mode()
    test_loss = jnp.mean((y_test - y_hat) ** 2)
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
