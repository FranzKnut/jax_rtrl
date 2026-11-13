"""Tests for loading actor-critic checkpoints of every trainer with `restore_actor_critic`."""

import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from orbax import checkpoint

from jax_rtrl.models.consolidation import WeightConsolidationState, init_weight_consolidation_state
from jax_rtrl.models.seq_models import RNNEnsembleConfig
from jax_rtrl.networks.actor_critic import (
    MODEL_FORMAT,
    ActorCriticConfig,
    RNNActorCritic,
    actor_critic_hparams,
)
from jax_rtrl.networks.autoencoders import Autoencoder, AutoencoderConfig
from jax_rtrl.networks.legacy import actor_variables, detect_format, restore_actor_critic
from jax_rtrl.networks.policies import PolicyConfig, PolicyRNN
from jax_rtrl.util.checkpointing import save_config

A_DIM = 2
X = jnp.zeros(3)
IMG = jnp.zeros((16, 16, 1))
SHARED = ActorCriticConfig(rnn_config=RNNEnsembleConfig(model_name="bptt", hidden_size=8))


def _save(path, tree, hparams):
    save_config(path, hparams)
    ckptr = checkpoint.StandardCheckpointer()
    ckptr.save(os.path.join(path, "ckpt"), tree)
    ckptr.wait_until_finished()


def _init_ac(config, x=X, img=None, seed=0):
    model = RNNActorCritic.from_config(A_DIM, False, config)
    return model.init({"params": jax.random.PRNGKey(seed), "default": jax.random.PRNGKey(seed)}, None, x, img=img)


def _assert_equal(a, b):
    jax.tree.map(np.testing.assert_array_equal, a, b)


def _assert_same_structure(restored, fresh, keys):
    """The restored parts have exactly the shapes a freshly built model expects."""
    for key in keys:
        shapes = lambda t: jax.tree.map(jnp.shape, t)  # noqa: E731
        assert shapes(restored["params"][key]) == shapes(fresh["params"][key]), key


def test_new_format_roundtrip(tmp_path):
    variables = _init_ac(SHARED)
    _save(str(tmp_path), variables, {"seed": 1, **actor_critic_hparams(SHARED)})

    restored = restore_actor_critic(str(tmp_path))

    assert restored.source_format == MODEL_FORMAT
    assert restored.config == SHARED
    _assert_equal(restored.variables, variables)
    assert restored.cons_state is None


def test_new_format_drops_decoder_unless_asked(tmp_path):
    variables = _init_ac(SHARED)
    variables = {"params": {**variables["params"], "dec": {"w": jnp.ones(2)}}}
    _save(str(tmp_path), variables, actor_critic_hparams(SHARED))

    assert "dec" not in restore_actor_critic(str(tmp_path)).variables["params"]
    assert "dec" in restore_actor_critic(str(tmp_path), with_decoder=True).variables["params"]


def test_rtrrl_checkpoint(tmp_path):
    variables = _init_ac(SHARED)
    hparams = {
        "lambda_pi": 0.95,
        "rnn_config": SHARED.rnn_config.to_dict(),
        "policy_config": SHARED.policy_config.to_dict(),
        "critic_config": RNNEnsembleConfig().to_dict(),
        "use_cnn": False,
        "skip_critic": False,
        "env_params": {"env_name": "CartPole-v1"},
    }
    _save(str(tmp_path), variables, hparams)

    restored = restore_actor_critic(str(tmp_path))

    assert restored.source_format == "rtrrl"
    assert restored.config.rnn_config == SHARED.rnn_config
    _assert_equal(restored.variables, variables)
    _assert_same_structure(restored.variables, _init_ac(restored.config), ["rnn", "actor", "critic"])


def test_ppo_ensemble_checkpoint(tmp_path):
    config = ActorCriticConfig()
    variables = _init_ac(config)
    saved = {col: {"cell": tree} for col, tree in variables.items()}
    hparams = {"model_config": PolicyConfig().to_dict(), "train_encoder": False, "num_units": 32, "model": "lru"}
    _save(str(tmp_path), saved, hparams)

    restored = restore_actor_critic(str(tmp_path))

    assert restored.source_format == "ppo_ensemble"
    assert restored.config.critic_config.out_dist == "Deterministic"
    _assert_equal(restored.variables, variables)


def test_pretrain_policy_critic_and_cons_state(tmp_path):
    policy_cfg = PolicyConfig(hidden_size=8)
    policy = PolicyRNN(a_dim=A_DIM, config=policy_cfg).init(jax.random.PRNGKey(1), None, x=X)
    critic = PolicyRNN(a_dim=1, config=policy_cfg).init(jax.random.PRNGKey(2), None, x=X)
    bundle = (policy, critic)
    hparams = {
        "policy_config": policy_cfg.to_dict(),
        "critic_config": policy_cfg.to_dict(),
        "train_critic": "V",
        "data_folder": "data/brax/mjx/ant",
    }
    _save(str(tmp_path), (bundle, init_weight_consolidation_state(bundle)), hparams)

    restored = restore_actor_critic(str(tmp_path))

    assert restored.source_format == "pretrain"
    _assert_equal(restored.variables["params"]["actor"], policy["params"])
    _assert_equal(restored.variables["params"]["critic"], critic["params"]["rnn"])
    assert isinstance(restored.cons_state, WeightConsolidationState)
    _assert_equal(restored.cons_state.theta_ref["params"]["critic"], critic["params"]["rnn"])
    _assert_same_structure(restored.variables, _init_ac(restored.config), ["actor", "critic"])


def test_pretrain_autoencoder(tmp_path):
    ae_cfg = AutoencoderConfig(latent_size=8)
    policy_cfg = PolicyConfig(hidden_size=8)
    autoencoder = Autoencoder(IMG.shape, config=ae_cfg).init(jax.random.PRNGKey(1), IMG)
    policy = PolicyRNN(a_dim=A_DIM, config=policy_cfg).init(jax.random.PRNGKey(2), None, x=jnp.zeros(8))
    hparams = {
        "policy_config": policy_cfg.to_dict(),
        "use_autoencoder": True,
        "autoencoder_cfg": ae_cfg.to_dict(),
    }
    _save(str(tmp_path), (policy, autoencoder), hparams)

    restored = restore_actor_critic(str(tmp_path))
    with_dec = restore_actor_critic(str(tmp_path), with_decoder=True)

    assert restored.config.use_cnn and restored.config.policy_config.latent_size == 8
    assert set(restored.variables["params"]) == {"actor", "enc"}
    assert "dec" in with_dec.variables["params"]
    fresh = _init_ac(replace(restored.config), x=IMG)
    _assert_same_structure(restored.variables, fresh, ["actor", "enc"])


def test_classic_ppo_is_refused(tmp_path):
    _save(str(tmp_path), {"params": {"emb": {"kernel": jnp.ones((3, 4))}}}, {"num_units": 32, "model": "CTRNN"})

    assert detect_format({"num_units": 32, "model": "CTRNN"}) == "ppo"
    with pytest.raises(ValueError, match="classic PPO"):
        restore_actor_critic(str(tmp_path))


def test_ppo_ensemble_with_shared_rnn(tmp_path):
    variables = _init_ac(replace(SHARED, critic_config=replace(PolicyConfig(), out_dist="Deterministic")))
    saved = {col: {"cell": tree} for col, tree in variables.items()}
    hparams = {
        "model_config": PolicyConfig().to_dict(),
        "rnn_config": SHARED.rnn_config.to_dict(),
        "train_encoder": True,
    }
    _save(str(tmp_path), saved, hparams)

    restored = restore_actor_critic(str(tmp_path))

    assert restored.config.rnn_config == SHARED.rnn_config
    _assert_equal(restored.variables, variables)


def test_actor_variables_for_imitation(tmp_path):
    variables = _init_ac(ActorCriticConfig())
    _save(str(tmp_path), variables, actor_critic_hparams(ActorCriticConfig()))
    restored = restore_actor_critic(str(tmp_path))

    actor = actor_variables(restored)
    policy = PolicyRNN(a_dim=A_DIM, config=restored.config.policy_config)
    fresh = policy.init(jax.random.PRNGKey(3), None, x=X)

    assert jax.tree.map(jnp.shape, actor) == jax.tree.map(jnp.shape, fresh)
    policy.apply(actor, None, X)


def test_actor_variables_refuses_shared_rnn(tmp_path):
    _save(str(tmp_path), _init_ac(SHARED), actor_critic_hparams(SHARED))

    with pytest.raises(ValueError, match="has rnn"):
        actor_variables(restore_actor_critic(str(tmp_path)))
