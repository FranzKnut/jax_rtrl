from dataclasses import dataclass, field
import os

import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
import simple_parsing

from jax_rtrl.models.cells.ctrnn import clip_tau
from jax_rtrl.models.seq_models import (
    RNNEnsemble,
    RNNEnsembleConfig,
    SequenceLayerConfig,
)
from jax_rtrl.supervised.training_utils import (
    get_data,
    make_model,
    predict,
    train_rnn_online as train,
)
from jax_rtrl.util.jax_util import mse_loss


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


class OnlineSupervisedExperiment:
    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg

    def _prepare_data(self, data=None):
        if data is None:
            x_train, y_train, x_test, y_test = get_data(self.cfg.dataset)
        else:
            x_train, y_train, x_test, y_test = data
        x_train = x_train.transpose(1, 0, 2)
        y_train = y_train.transpose(1, 0, 2)
        return x_train, y_train, x_test, y_test

    def _make_loss(self, model):
        def loss(p, __x, __y, rnn_state=None):
            rnn_state, y_hat = model.apply(p, rnn_state, __x)
            if self.cfg.rnn_config.ensemble_method is not None:
                y_hat = y_hat[0]
            if self.cfg.rnn_config.out_dist == "Deterministic":
                value = mse_loss(y_hat.mode().reshape(__y.shape), __y)
            else:
                value = jnp.mean(-y_hat.log_prob(__y))
            return value, rnn_state

        return loss

    def gradient_smoke_test(self, data=None, key=None):
        key = jrand.PRNGKey(0) if key is None else key
        key_model, key_state = jrand.split(key)
        x_train, y_train, *_ = self._prepare_data(data)
        x_seq = x_train[:, 0]
        y_seq = y_train[:, 0]

        model = RNNEnsemble(self.cfg.rnn_config, y_seq.shape[-1])
        params = model.init(key_model, None, x_seq[0])
        h0 = model.apply(params, key_state, x_seq[0].shape, method=model.initialize_carry)

        def loss_fn(p):
            def step(carry, _data):
                __x, __y = _data
                carry, y_hat = model.apply(p, carry, __x)
                if self.cfg.rnn_config.ensemble_method is not None:
                    y_hat = y_hat[0]
                if self.cfg.rnn_config.out_dist == "Deterministic":
                    value = mse_loss(y_hat.mode().reshape(__y.shape), __y)
                else:
                    value = jnp.mean(-y_hat.log_prob(__y))
                return carry, value

            _, losses = jax.lax.scan(step, h0, (x_seq, y_seq))
            return losses.mean()

        grads = jax.grad(loss_fn)(params)
        return grads

    def forward_smoke_test(self, data=None, key=None):
        key = jrand.PRNGKey(0) if key is None else key
        x_train, y_train, x_test, _ = self._prepare_data(data)
        model, params, _ = make_model(
            x_train[0], key, y_train.shape[-1], self.cfg.rnn_config
        )
        y_hat = predict(model, params, x_test[None] if x_test.ndim == 2 else x_test)
        if self.cfg.rnn_config.ensemble_method is not None:
            y_hat = y_hat[0]
        return y_hat.mode()

    def run(self, data=None, key=None, plot=False):
        key = jrand.PRNGKey(0) if key is None else key
        key_model, key_train = jrand.split(key)
        x_train, y_train, x_test, y_test = self._prepare_data(data)
        model, params, h0 = make_model(
            x_train[0], key_model, y_train.shape[-1], self.cfg.rnn_config
        )

        y_hat = predict(model, params, x_test[None] if x_test.ndim == 2 else x_test)
        if self.cfg.rnn_config.ensemble_method is not None:
            y_hat = y_hat[0]
        initial_loss = mse_loss(y_hat.mode().squeeze(), y_test)
        print(f"Initial loss: {initial_loss:.3f}")

        optimizer = optax.chain(
            optax.clip_by_block_rms(self.cfg.gradient_clip)
            if self.cfg.gradient_clip
            else optax.identity(),
            optax.adam(self.cfg.learning_rate),
        )

        params, losses = train(
            self._make_loss(model),
            optimizer,
            params,
            (x_train, y_train),
            key_train,
            h0,
            param_post_update_fn=clip_tau,
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
    OnlineSupervisedExperiment(cfg).run(plot=True)


if __name__ == "__main__":
    main()
