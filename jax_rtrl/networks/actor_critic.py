"""Actor-critic model with optional encoder and shared RNN: ``obs -> [enc] -> [rnn] -> actor + critic``.

The parameters of `RNNActorCritic` are laid out as ``{"enc"?, "rnn"?, "actor", "critic"}``. This is the
shared checkpoint layout of all trainers, see `MODEL_FORMAT` and `jax_rtrl.networks.legacy` for loading
checkpoints saved in older layouts.
"""

import dataclasses
from dataclasses import dataclass, field

import distrax
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import linen as nn
from simple_parsing import Serializable
from jax_rtrl.networks.autoencoders import ConvEncoder, ConvConfig
from jax_rtrl.models.feedforward import DistributionLayer
from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig
from jax_rtrl.networks.policies import PolicyConfig, PolicyRNN


Actor = DistributionLayer


class Critic(DistributionLayer):
    """Value critic head. Same as DistributionLayer with out_size=1 default."""

    out_size: int = 1
    distribution: str = "Deterministic"


MODEL_FORMAT = "enc_rnn_ac/1"
"""Version tag of the `RNNActorCritic` checkpoint layout, saved as ``model_format`` in ``hparams.json``."""


@dataclass
class ActorCriticConfig(Serializable):
    """Architecture of an `RNNActorCritic`, i.e. everything needed to rebuild it from a checkpoint."""

    use_cnn: bool = False  # Encode image observations with a ConvEncoder ("enc")
    cnn_config: ConvConfig = field(default_factory=ConvConfig)
    rnn_config: RNNEnsembleConfig | None = None  # Shared RNN ("rnn"), None for none
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    critic_config: PolicyConfig = field(default_factory=PolicyConfig)
    skip_actor: bool = False  # Concatenate the observation to the actor input
    skip_critic: bool = False  # Concatenate the observation to the critic input
    critic_subset_start_idx: int | None = None
    actor_subset_stop_idx: int | None = None

    @classmethod
    def from_fields(cls, obj) -> "ActorCriticConfig":
        """Collect the architecture from a trainer config with the same (flat) field names.

        Parameters
        ----------
        obj : object
            E.g. an RTRRLConfig. Fields it lacks keep their defaults.

        Returns
        -------
        ActorCriticConfig
            The architecture part of `obj`.
        """
        names = [f.name for f in dataclasses.fields(cls)]
        return cls(**{n: getattr(obj, n) for n in names if hasattr(obj, n)})


class RNNActorCritic(nn.RNNCellBase):
    """RTRRL cell with shared RNN and linear actor and critic networks."""

    a_dim: int
    discrete: bool
    # obs_dim: int = None
    # img_dim: int = None
    rnn_config: RNNEnsembleConfig | None = field(default_factory=RNNEnsembleConfig)
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    critic_config: RNNEnsembleConfig = field(default_factory=RNNEnsembleConfig)
    use_cnn: bool = False
    cnn_config: ConvConfig = field(default_factory=ConvConfig)
    # act_log_bounds: tuple[float, float] | float | None = -1
    act_bounds: tuple[float] | None = None
    # pass_obs: bool = False
    pred_obs: bool = False
    critic_subset_start_idx: int | None = None
    """Index of the start of the rnn hidden state subset to be used for the critic. 
    
    The critic input has input shape (..., hidden_size-critic_subset_start_idx).
    If None, the full hidden state is used.
    """
    actor_subset_stop_idx: int | None = None
    """Index of the stop of the rnn hidden state subset to be used for the actor. 

    If set, the actor input has input shape (..., actor_subset_stop_idx).
    If None, the full hidden state is used.
    """
    skip_critic: bool = False
    """Concatenate the observation to the critic input"""
    skip_actor: bool = False
    """Concatenate the observation to the actor input"""

    @classmethod
    def from_config(cls, a_dim: int, discrete: bool, config: ActorCriticConfig, **kwargs) -> "RNNActorCritic":
        """Build the model described by `config`.

        Parameters
        ----------
        a_dim : int
            Action dimension.
        discrete : bool
            Whether the action space is discrete.
        config : ActorCriticConfig
            Architecture.
        **kwargs
            Further attributes, e.g. ``act_bounds`` or ``name``.

        Returns
        -------
        RNNActorCritic
            The unbound model.
        """
        fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
        return cls(a_dim=a_dim, discrete=discrete, **fields, **kwargs)

    @property
    def use_shared_rnn(self) -> bool:
        """Whether to use a shared RNN for actor and critic."""
        return self.rnn_config is not None and self.rnn_config.model_name is not None

    def _skip(self, encoded, obs):
        """Concatenate the observation to every ensemble member's latent.

        Without a shared RNN the latent already is the observation, so nothing is added.
        """
        if obs is None or not self.use_shared_rnn:
            return encoded
        obs = jnp.broadcast_to(obs, encoded.shape[:-1] + obs.shape[-1:])
        return jnp.concatenate([encoded, obs], axis=-1)

    def setup(self) -> None:
        """Initialize components."""
        if self.use_cnn and (self.skip_critic or self.skip_actor):
            raise NotImplementedError(
                "skip_critic and skip_actor are not supported with a CNN encoder."
            )

        if self.rnn_config is not None:
            assert self.policy_config.num_modules % self.rnn_config.num_modules == 0, (
                "The number of modules of the policy have to be multiples of the RNN modules."
            )
            assert self.critic_config.num_modules % self.rnn_config.num_modules == 0, (
                "The number of modules of the critic have to be multiples of the RNN modules."
            )
        self.rnn = RNNEnsemble(self.rnn_config, out_size=None, name="rnn")

        self.actor = PolicyRNN(
            self.a_dim,
            self.policy_config,
            # TODO: this config for splitting inputs to ensemble modules is confusing
            split_input=self.use_shared_rnn,
            # and (self.rnn_config.num_modules == self.policy_config.num_modules),
            name="actor",
        )
        self.critic = RNNEnsemble(
            self.critic_config,
            out_size=1,
            split_input=self.use_shared_rnn,
            # and (self.rnn_config.num_modules == self.critic_config.num_modules),
            name="critic",
        )

        if self.pred_obs:
            raise NotImplementedError("Observation prediction not implemented yet.")
            self.obs = RNNEnsemble(
                self.obs_dim + 1,  # Predict obs and reward
                # f_align=self.f_align,
                #    kernel_init=nn.initializers.zeros_init(),
                #  use_bias=False,
                name="obs",
            )

        if self.use_cnn:
            self.enc = ConvEncoder(
                self.policy_config.latent_size,
                self.cnn_config,
                name="enc",
            )

    def encode(self, carry, obs, reset=False, img=None, training=True, **kwargs):
        """Step RNN."""

        if self.use_cnn:
            if img is None:
                img = obs
                obs = None
            encoded_img = self.enc(img)
            # If given, concatenate encoded image with vector observations
            if obs is not None:
                obs = jnp.concatenate([obs, encoded_img], axis=-1)
            else:
                obs = encoded_img

        if not self.use_shared_rnn:
            return obs, carry

        if self.rnn_config is not None and not self.rnn_config.model_name:
            # No RNN, just return repeated obs as hidden state
            obs = obs[None] * jnp.ones(
                (self.rnn_config.num_modules,) + (1,) * (obs.ndim)
            )
            return obs, carry

        carry, hidden = self.rnn(carry, obs, training, **kwargs)
        return hidden, carry

    def value(self, encoded, img=None, v_hidden=None, training=True, obs=None):
        """Compute value from latent, with ``obs`` appended if ``skip_critic``."""
        # if not self.shared:
        #     # hidden = jnp.concatenate([jax.lax.stop_gradient(hidden[0]), hidden[1]], axis=-1)
        #     hidden = hidden[..., 1:, :]
        # if self.pass_obs:
        #     if len(x.shape) < len(encoded.shape):
        #         x = jnp.expand_dims(x, -2)
        #     encoded = jnp.concatenate([encoded, x], axis=-1)

        # Slice the subset of the hidden state for the critic
        encoded = encoded[..., self.critic_subset_start_idx :]

        if self.use_shared_rnn:
            # Tile the hidden state for the critic ensemble
            if encoded.ndim == 1:
                encoded = encoded[None]
            _num_module_factor = self.critic_config.num_modules // (
                self.rnn_config.num_modules if self.rnn_config is not None else 1
            )  # Compute how many times to tile the hidden state for the critic ensemble
            encoded = encoded * jnp.ones((_num_module_factor,) + (1,) * (encoded.ndim))
            enc_shape = encoded.shape
            # Collapse the first two dimensions to get total number of modules
            encoded = encoded.reshape(enc_shape[0] * enc_shape[1], *enc_shape[2:])

        if self.skip_critic:
            encoded = self._skip(encoded, obs)
        return self.critic(v_hidden, encoded, training=training)

    def obs_prediction(self, hidden, a, x=None):
        """Compute observation prediction from latent."""
        hidden = jnp.concatenate([hidden, a.reshape(*hidden.shape[:-1], -1)], axis=-1)
        # if self.pass_obs:
        #     if len(x.shape) < len(hidden.shape):
        #         x = jnp.expand_dims(x, -2)
        #     hidden = jnp.concatenate([hidden, x], axis=-1)
        return self.obs(hidden)

    def policy(
        self,
        encoded,
        img=None,
        pi_state=None,
        sample_act: bool = False,
        training: bool = True,
        selected_act=None,
        epsilon: float = 0.0,
        obs=None,
    ) -> tuple[jax.Array, distrax.Distribution] | jax.Array:
        """Compute the action distribution from latent, with ``obs`` appended if ``skip_actor``."""
        # if not self.shared:
        #     # hidden = jnp.concatenate([hidden[0], jax.lax.stop_gradient(hidden[1])], axis=-1)
        #     hidden = hidden[..., :1, :]
        # if self.pass_obs:
        #     if len(x.shape) < len(hidden.shape):
        #         x = jnp.expand_dims(x, -2)
        #     hidden = jnp.concatenate([hidden, x], axis=-1)

        if self.actor_subset_stop_idx is not None:
            encoded = encoded[..., : self.actor_subset_stop_idx]

        if self.use_shared_rnn:
            # Tile the hidden state for the policy ensemble
            if encoded.ndim == 1:
                encoded = encoded[None]
            _num_module_factor = self.policy_config.num_modules // (
                self.rnn_config.num_modules
            )  # Compute how many times to tile the hidden state for the policy ensemble
            encoded = encoded * jnp.ones((_num_module_factor,) + (1,) * (encoded.ndim))
            enc_shape = encoded.shape
            # Collapse the first two dimensions to get total number of modules
            encoded = encoded.reshape(enc_shape[0] * enc_shape[1], *enc_shape[2:])

        if self.skip_actor:
            encoded = self._skip(encoded, obs)
        encoded, (combined_dist, dists) = self.actor(pi_state, encoded, training, img)
        if sample_act:
            greedy_action = dists.mode()
            if not training:
                action = greedy_action
            else:
                _rng = jax.random.uniform(
                    self.make_rng("default"), shape=greedy_action.shape
                )
                # Epsilon-greedy action selection
                action = jnp.where(
                    _rng < epsilon,
                    greedy_action,
                    combined_dist.sample(seed=self.make_rng("default")),
                )
            if self.act_bounds is not None:
                action = jnp.clip(
                    action, jnp.array(self.act_bounds[0]), jnp.array(self.act_bounds[1])
                )
            return encoded, (action, (combined_dist, dists))
        return encoded, (combined_dist, dists)

    @nn.compact
    def __call__(
        self, carry, x, img=None, reset=False, training=True, epsilon: float = 0.0
    ):
        """Step RNN and compute actor and critic."""
        h0 = self.initialize_carry(self.make_rng("default"), x.shape)
        if carry is None:
            # Initialize seed and the carry
            carry = h0
        else:
            carry = jax.tree.map(lambda a, b: jnp.where(reset, a, b), h0, carry)

        # RNN
        encoded, rnn_state = self.encode(
            carry[0], x, reset=reset, img=img, training=training
        )

        # Critic
        v_state, v_hat = self.value(encoded, x, carry[1], training=training, obs=x)
        # selected_act = v_hat.argmax()

        # Actor
        pi_state, (action, _) = self.policy(
            encoded, img, carry[2], True, training=training, epsilon=epsilon, obs=x
        )

        if self.pred_obs:
            prediction = self.obs_prediction(encoded, action, img)
            out = (action, v_hat, prediction, encoded)
        else:
            out = (action, v_hat, encoded)
        return (rnn_state, v_state, pi_state), out

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    def _init_shared_rnn(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the shared RNN cell."""
        if self.use_cnn:
            input_shape = input_shape[:-3] + (self.policy_config.latent_size,)

        return self.rnn.initialize_carry(rng, input_shape)

    def initialize_carry(self, rng: PRNGKey, input_shape: tuple[int, ...]):
        """Initialize the submodule states."""

        if self.use_cnn:
            input_shape = input_shape[:-3] + (self.policy_config.latent_size,)
        obs_dim = input_shape[-1]
        if self.rnn_config is not None and self.rnn_config.model_name:
            rnn_state = self._init_shared_rnn(rng, input_shape)
            input_shape = input_shape[:-1] + self.rnn_config.layers[-1:]
        else:
            rnn_state = None

        v_shape = input_shape[:-1] + (
            input_shape[-1]
            if self.critic_subset_start_idx is None
            else input_shape[-1] - self.critic_subset_start_idx,
        )

        pi_shape = input_shape[:-1] + (
            input_shape[-1]
            if self.actor_subset_stop_idx is None
            else self.actor_subset_stop_idx,
        )
        if self.use_shared_rnn:
            if self.skip_critic:
                v_shape = input_shape[:-1] + (v_shape[-1] + obs_dim,)
            if self.skip_actor:
                pi_shape = input_shape[:-1] + (pi_shape[-1] + obs_dim,)

        v_state = self.critic.initialize_carry(
            rng,
            (self.critic_config.num_modules,) + v_shape
            if self.critic.split_input
            else v_shape,
        )
        pi_state = self.actor.initialize_carry(
            rng,
            (self.policy_config.num_modules,) + pi_shape
            if self.actor.split_input
            else pi_shape,
        )
        return (rnn_state, v_state, pi_state)


def actor_critic_hparams(config: ActorCriticConfig) -> dict:
    """Make the ``hparams.json`` entries that mark a checkpoint as `MODEL_FORMAT`.

    Trainers merge these into the hparams they save, so that `jax_rtrl.networks.legacy.restore_actor_critic`
    can rebuild the model without knowing the trainer's config class.

    Parameters
    ----------
    config : ActorCriticConfig
        Architecture of the saved model.

    Returns
    -------
    dict
        ``{"model_format": MODEL_FORMAT, "actor_critic_config": config.to_dict()}``.
    """
    return {"model_format": MODEL_FORMAT, "actor_critic_config": config.to_dict()}
