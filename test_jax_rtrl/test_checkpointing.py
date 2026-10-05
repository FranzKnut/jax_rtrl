"""Tests for restoring policy checkpoints."""

import os

import jax
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint

from jax_rtrl.models.consolidation import WeightConsolidationState, init_weight_consolidation_state
from jax_rtrl.networks.policies import PolicyConfig, PolicyRNN, restore_policy_from_ckpt
from jax_rtrl.util.checkpointing import save_config

A_DIM = 2
X = jnp.zeros(3)


def _save(path, tree, hparams):
    save_config(path, hparams)
    ckptr = checkpoint.StandardCheckpointer()
    ckptr.save(os.path.join(path, "ckpt"), tree)
    ckptr.wait_until_finished()


def _init(a_dim, x, seed):
    model = PolicyRNN(a_dim=a_dim, config=PolicyConfig())
    return model.init(jax.random.PRNGKey(seed), None, x=x)


def test_restore_policy_critic_and_cons_state(tmp_path):
    """Regression: (policy, critic) bundles saved with cons_state were indexed wrongly on restore."""
    policy_params = _init(A_DIM, X, 1)
    critic_params = _init(1, jnp.zeros(X.shape[0] + A_DIM), 2)
    params = (policy_params, critic_params)
    hparams = {
        "policy_config": PolicyConfig().to_dict(),
        "critic_config": PolicyConfig().to_dict(),
        "train_critic": True,
        "cons_config": {"cons_type": "si", "decay": 0.95},
    }
    _save(str(tmp_path), (params, init_weight_consolidation_state(params)), hparams)

    (policy, critic), cons_state = restore_policy_from_ckpt(
        str(tmp_path), A_DIM, return_cons_state=True, x=X
    )

    assert isinstance(cons_state, WeightConsolidationState)
    for restored, saved in [(policy, policy_params), (critic, critic_params)]:
        jax.tree.map(np.testing.assert_array_equal, restored.variables, saved)


def test_restore_params_only_ignores_cons_config(tmp_path):
    """Regression: params-only checkpoints with a (legacy) cons_config in hparams failed to restore."""
    policy_params = _init(A_DIM, X, 1)
    hparams = {
        "policy_config": PolicyConfig().to_dict(),
        "cons_config": {"cons_type": "si", "apply_when": "step", "decay": 0.95},
    }
    _save(str(tmp_path), policy_params, hparams)

    policy, cons_state = restore_policy_from_ckpt(str(tmp_path), A_DIM, return_cons_state=True, x=X)

    assert cons_state is None
    jax.tree.map(np.testing.assert_array_equal, policy.variables, policy_params)
