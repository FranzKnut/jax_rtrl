from dataclasses import dataclass, field
from chex import PRNGKey
from jax import numpy as jnp

from flax import linen as nn
import jax
from simple_parsing import Serializable

from jax_rtrl.models import make_rnn_ensemble_config
from jax_rtrl.models.autoencoders import ConvEncoder
from jax_rtrl.models.feedforward import MLPEnsemble
from jax_rtrl.models.seq_models import RNNEnsemble


@dataclass(unsafe_hash=True)
class PolicyConfig(Serializable):
    """Config for Policy.

    TODO: this repeats many parameters found in the RNNEnsemble config.
    """

    agent_type: str = "mlp"
    hidden_size: int = 256
    num_modules: int = 5
    num_blocks: int = 1
    num_layers: int = 1
    stochastic: bool = False
    output_layers: tuple[int] | None = (128,)
    skip_connection: bool = False
    norm: str | None = "layer"  # e.g. "layer", "batch", "group", None

    # RNN specific
    model_config: dict = field(default_factory=dict, hash=False)
    glu: bool = False
    dropout: float = 0.0

    # CNN specific
    use_cnn: bool = False
    latent_size: int = 16
    c_hid: int = 4


class Policy(nn.Module):
    """Generic Policy Base Class."""

    a_dim: int
    config: PolicyConfig = field(default_factory=PolicyConfig)
    use_rnn: bool = False

    def __call__(self, x):
        """Compute action or action distribution for given observation."""
        raise NotImplementedError()

    def call_and_sample(self, rng: PRNGKey, *args, **kwargs):
        """Compute action or action distribution for given observation."""
        out = self(*args, **kwargs)
        if self.use_rnn:
            *rest, out = out
        action = out.sample(seed=rng)
        if self.use_rnn:
            return *rest, action
        else:
            return action


class PolicyMLP(Policy):
    """MLP Policy."""

    use_rnn: bool = False  # Do not alter!

    @nn.compact
    def __call__(self, x, training: bool = False):
        """Compute Action from observation."""
        if self.config.use_cnn:
            x = ConvEncoder(latent_size=self.config.latent_size, c_hid=self.config.c_hid)(x)
        layers = [self.config.hidden_size] * self.config.num_layers
        if self.config.output_layers:
            layers += list(self.config.output_layers)
        layers += [self.a_dim]
        x = MLPEnsemble(
            out_size=self.a_dim,
            num_modules=self.config.num_modules,
            out_dist="Normal" if self.config.stochastic else None,
            kwargs={"layers": layers, "norm": self.config.norm},
            name="mlp",
            skip_connection=self.config.skip_connection,
        )(x, training)
        return x


class PolicyRNN(nn.RNNCellBase, Policy):
    """RNN Policy."""

    use_rnn: bool = True  # Do not alter!
    num_submodule_extra_args: int = 0

    def setup(self):
        """Initialize and set up the RNN configuration."""
        self.rnn = RNNEnsemble(
            make_rnn_ensemble_config(
                model_name=self.config.agent_type,
                hidden_size=self.config.hidden_size,
                out_size=self.a_dim,
                num_modules=self.config.num_modules,
                num_blocks=self.config.num_blocks,
                num_layers=self.config.num_layers,
                output_layers=self.config.output_layers,
                stochastic=self.config.stochastic,
                model_kwargs=self.config.model_config,
                skip_connection=self.config.skip_connection,
                glu=self.config.glu,
                dropout=self.config.dropout,
                norm=self.config.norm,
            ),
            num_submodule_extra_args=self.num_submodule_extra_args,
            name="rnn",
        )

    @nn.compact
    def __call__(
        self,
        carry: jax.Array | None = None,
        x: jax.Array = None,
        img: jax.Array = None,
        training: bool = False,
        *args,
        **kwargs,
    ):  # type: ignore
        """Compute Action from observation."""
        if self.config.use_cnn:
            if img is None:
                img = x
            img_enc = ConvEncoder(
                latent_size=self.config.latent_size,
                c_hid=self.config.c_hid,
                name="cnn",
            )(img)
            if x is None:
                input_shape = img_enc.shape[:-1] + (0,)
                x = img_enc
            else:
                input_shape = x.shape
                x = jnp.concatenate([x, img_enc], axis=-1)
        else:
            input_shape = x.shape

        # Step RNN
        if carry is None:
            carry = self.rnn.initialize_carry(jax.random.key(0), input_shape)
        carry, x = self.rnn(carry, x, training, *args, **kwargs)
        return carry, x

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the RNN cell carry."""
        if self.config.use_cnn:
            input_shape = input_shape[:-1] + (self.config.latent_size + input_shape[-1],)
        return self.rnn.initialize_carry(rng, input_shape)


class PolicyRTRL(PolicyRNN):
    """Ensemble RNN Policy."""

    def loss_and_grad_asnyc(self, loss_fn, carry, hidden, x, target):
        """Compute loss and gradients asynchronously.

        Parameters
        ----------
        params : dict
            Parameters for the RNN.
        loss_fn : callable
            Loss function to compute the loss.
        carry : Any
            Carry state for the RNN.
        hidden : Any
            Hidden state for the RNN.
        x : Any
            Input batch.
        target : Any
            Target batch.

        Returns
        -------
        tuple: A tuple containing:
        - loss (Any): Computed loss.
        - gradients (dict): Dictionary containing gradients for the RNN parameters and zeroed gradients for other parameters.
        """
        loss, rnn_grads = self.rnn.loss_and_grad_async(loss_fn, carry, hidden, x, target)
        return loss, {
            "params": {"rnn": rnn_grads["params"]},
            **{k: jax.tree.map(lambda x: jnp.zeros_like(x), v) for k, v in self.variables.items() if k != "params"},
        }
