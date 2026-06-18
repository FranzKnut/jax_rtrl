from dataclasses import dataclass, field
import os

import jax
import jax.random as jrand
import optax
import simple_parsing

from jax_rtrl.models.seq_models import (
    RNNEnsemble,
    RNNEnsembleConfig,
    SequenceLayerConfig,
    scan_rnn,
)
from jax_rtrl.supervised.training_utils import (
    get_data,
    make_model,
    predict,
    train_rnn_offline as train,
)
from jax_rtrl.util.jax_util import mse_loss

# jax.config.update("jax_disable_jit", True)


@dataclass
class TrainingConfig:
    # dataset: str = "legacy_rollouts"
    dataset: str = "sine"
    # dataset: str = "spirals"
    num_steps: int = 10000
    learning_rate: float = 1e-3
    gradient_clip: float | None = None

    rnn_config: RNNEnsembleConfig = field(
        default_factory=lambda: RNNEnsembleConfig(
            # model_name="bptt",
            # model_name="ltc",
            model_name="lrc",
            layers=(32, 4),
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


class OfflineSupervisedExperiment:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    def _prepare_data(self, data=None):
        if data is None:
            return get_data(self.cfg.dataset)
        return data

    def _make_loss(self, model):
        def loss(p, __x, __y):
            _, y_hat = scan_rnn(model, p, __x)
            if self.cfg.rnn_config.ensemble_method is not None:
                y_hat = y_hat[0]
            return mse_loss(y_hat.mode().mean(axis=-1), __y)

        return loss

    def gradient_smoke_test(self, data=None, key=None):
        key = jrand.PRNGKey(0) if key is None else key
        x_train, y_train, *_ = self._prepare_data(data)
        x_seq = x_train[0]
        y_seq = y_train[0]
        model = RNNEnsemble(self.cfg.rnn_config, y_seq.shape[-1])
        params = model.init(key, None, x_seq[0])

        def loss_fn(p):
            _, y_hat = scan_rnn(model, p, x_seq)
            if self.cfg.rnn_config.ensemble_method is not None:
                y_hat = y_hat[0]
            return mse_loss(y_hat.mode().mean(axis=-1), y_seq)

        grads = jax.grad(loss_fn)(params)
        return grads

    def forward_smoke_test(self, data=None, key=None):
        key = jrand.PRNGKey(0) if key is None else key
        x_train, y_train, x_test, _ = self._prepare_data(data)
        model, params, _ = make_model(
            x_train[:, 0], key, y_train.shape[-1], self.cfg.rnn_config
        )
        y_hat = predict(model, params, None, x_test[None] if x_test.ndim == 2 else x_test)
        if self.cfg.rnn_config.ensemble_method is not None:
            y_hat = y_hat[0]
        return y_hat.mode()

    def run(self, data=None, key=None, plot=False):
        key = jrand.PRNGKey(0) if key is None else key
        key_model, key_train = jrand.split(key)

        x_train, y_train, x_test, y_test = self._prepare_data(data)
        model, params, _ = make_model(
            x_train[:, 0], key_model, y_train.shape[-1], self.cfg.rnn_config
        )

        y_hat = predict(model, params, None, x_test[None] if x_test.ndim == 2 else x_test)
        if self.cfg.rnn_config.ensemble_method is not None:
            y_hat = y_hat[0]
        initial_loss = mse_loss(y_hat.mode().squeeze(), y_test)
        print(f"Initial loss: {initial_loss:.3f}")

        optimizer = optax.adam(self.cfg.learning_rate)
        params, losses = train(
            self._make_loss(model),
            optimizer,
            params,
            (x_train, y_train),
            key_train,
            num_steps=self.cfg.num_steps,
        )

        y_hat = predict(model, params, x_test[None] if x_test.ndim == 2 else x_test)
        if self.cfg.rnn_config.ensemble_method is not None:
            y_hat = y_hat[0]
        y_hat = y_hat.mode()
        final_loss = mse_loss(y_hat.squeeze(), y_test)
        print(f"Final loss: {final_loss:.3f}")

        if plot:
            self._plot_results(losses, x_test, y_test, y_hat, final_loss)
        return {
            "params": params,
            "losses": losses,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        }

    def _plot_results(self, losses, x_test, y_test, y_hat, final_loss):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.subplot(1, 2, 2)

        if self.cfg.dataset == "spirals":
            for _x, _y in zip(x_test, y_test):
                plt.plot(
                    _x[..., 0].T,
                    _x[..., 1].T,
                    c="y" if _y[..., 0, 0] else "darkblue",
                    alpha=0.5,
                )
            plt.scatter(x_test[..., 0], x_test[..., 1], c=y_hat[..., 0])
            plt.title(f"Spirals, loss:{final_loss:.3f}")
            plt.legend()
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/spirals.png")
        elif self.cfg.dataset == "sine":
            plt.plot(x_test.squeeze(), y_test.squeeze(), label="target")
            plt.plot(x_test.squeeze(), y_hat.squeeze(), label="trained")
            plt.legend()
            os.makedirs("plots", exist_ok=True)
            plt.savefig("plots/sinewave.png")
        else:
            plt.plot(y_test[:, 0, ..., 0], label="target")
            plt.plot(y_hat[:, 0, ..., 0], label="trained")
            plt.legend()
            os.makedirs("plots/supervised", exist_ok=True)
            plt.savefig(
                f"plots/supervised/{self.cfg.dataset}_{self.cfg.rnn_config.model_name}.png"
            )
        plt.show()


def main():
    cfg = simple_parsing.parse(TrainingConfig)
    OfflineSupervisedExperiment(cfg).run(plot=True)


if __name__ == "__main__":
    main()
