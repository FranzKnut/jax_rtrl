"""Load actor-critic checkpoints of any trainer into the `RNNActorCritic` layout.

Checkpoints saved with `actor_critic_hparams` carry ``model_format`` in their ``hparams.json``. Older
checkpoints are recognized by the fields of the config they were saved with, and converted on load:

========================  ===========================================  ==========================================
Format                    Recognized by                                Saved parameter tree
========================  ===========================================  ==========================================
``enc_rnn_ac/1``          ``model_format``                             ``{enc?, rnn?, actor, critic?, dec?}``
``rtrrl``                 ``lambda_pi`` (RTRRLConfig and subclasses)   ``{enc?, rnn?, actor, critic}``
``ppo_ensemble``          ``model_config`` and ``train_encoder``       ``{cell: {enc?, actor, critic}}``
``pretrain``              ``policy_config`` (RTR-IIL PreTrainConfig)   ``(policy, [autoencoder], [critic])``
``ppo``                   ``num_units`` and ``model``                  ``ActorCriticRNN``, cannot be converted
========================  ===========================================  ==========================================

The checkpoint files themselves are never changed.
"""

import dataclasses
import os
import warnings
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
from orbax import checkpoint

from jax_rtrl.models.consolidation import WeightConsolidationState
from jax_rtrl.networks.actor_critic import MODEL_FORMAT, ActorCriticConfig
from jax_rtrl.networks.autoencoders import AutoencoderConfig, ConvConfig
from jax_rtrl.networks.policies import PolicyConfig, _from_legacy_dict
from jax_rtrl.models.seq_models import RNNEnsembleConfig
from jax_rtrl.util.checkpointing import restore_config, restore_remote

CONS_STATE_KEYS = {"omega", "reg_strength", "theta_ref"}


@dataclass
class RestoredActorCritic:
    """An actor-critic checkpoint converted to the `RNNActorCritic` layout.

    Attributes
    ----------
    variables : dict
        Flax variables ``{collection: {"enc"?, "rnn"?, "actor", "critic"?, "dec"?}}``. Parts the
        checkpoint doesn't have are missing, so merge them into freshly initialized variables.
    config : ActorCriticConfig
        Architecture of the saved model.
    cons_state : WeightConsolidationState or None
        Consolidation state in the same layout as ``variables``, None if none was saved.
    source_format : str
        Format the checkpoint was saved in, see the module docstring.
    hparams : dict
        The full saved ``hparams.json``.
    """

    variables: dict
    config: ActorCriticConfig
    cons_state: WeightConsolidationState | None
    source_format: str
    hparams: dict


def detect_format(hparams: dict) -> str:
    """Tell the checkpoint format from its saved hparams.

    Parameters
    ----------
    hparams : dict
        Contents of ``hparams.json``.

    Returns
    -------
    str
        One of ``MODEL_FORMAT``, ``"rtrrl"``, ``"ppo_ensemble"``, ``"pretrain"`` or ``"ppo"``.

    Raises
    ------
    ValueError
        If the format can't be recognized.
    """
    if "model_format" in hparams:
        if hparams["model_format"] != MODEL_FORMAT:
            raise ValueError(f"Unknown model_format {hparams['model_format']}, expected {MODEL_FORMAT}.")
        return MODEL_FORMAT
    if "lambda_pi" in hparams:
        return "rtrrl"
    if "model_config" in hparams and "train_encoder" in hparams:
        return "ppo_ensemble"
    if "policy_config" in hparams:
        return "pretrain"
    if "num_units" in hparams and "model" in hparams:
        return "ppo"
    raise ValueError(f"Unrecognized checkpoint config with keys {sorted(hparams)}.")


def _restore_raw(path: str):
    """Restore the saved tree without a target; tuples come back as lists."""
    tree = checkpoint.StandardCheckpointer().restore(os.path.join(os.path.abspath(path), "ckpt"))
    return jax.tree.map(jnp.asarray, tree)


def _split_cons_state(tree):
    """Split a ``(params, cons_state)`` tree, as saved by consolidating trainers."""
    if (
        isinstance(tree, (list, tuple))
        and len(tree) == 2
        and isinstance(tree[1], dict)
        and set(tree[1]) == CONS_STATE_KEYS
    ):
        return tree[0], tree[1]
    return tree, None


def _map_cons_state(cons_state: dict | None, convert) -> WeightConsolidationState | None:
    """Apply the parameter conversion `convert` to every tree of a raw consolidation state."""
    if cons_state is None:
        return None
    return WeightConsolidationState(**{k: convert(v) for k, v in cons_state.items()})


def _config_from(hparams: dict, cls: type, **overrides):
    """Build dataclass `cls` from the keys of `hparams` that are fields of it."""
    names = {f.name for f in dataclasses.fields(cls)} - set(overrides)
    return cls(**{k: v for k, v in hparams.items() if k in names}, **overrides)


def _policy_config(d: dict | None) -> PolicyConfig:
    return PolicyConfig() if d is None else _from_legacy_dict(dict(d))


def _rtrrl_config(hparams: dict) -> ActorCriticConfig:
    """Read the architecture fields of an RTRRLConfig."""
    rnn = hparams.get("rnn_config")
    return _config_from(
        hparams,
        ActorCriticConfig,
        cnn_config=ConvConfig.from_dict(hparams.get("cnn_config") or {}),
        rnn_config=None if rnn is None else RNNEnsembleConfig.from_dict(rnn),
        policy_config=_policy_config(hparams.get("policy_config")),
        critic_config=_policy_config(hparams.get("critic_config")),
    )


def _ppo_ensemble_config(hparams: dict) -> ActorCriticConfig:
    """Rebuild the architecture `EnsembleActorCritic` derives from a PPOEnsembleParams."""
    policy = _policy_config(hparams["model_config"])
    use_cnn = policy.use_cnn
    policy = replace(policy, use_cnn=False)
    rnn = hparams.get("rnn_config")
    return ActorCriticConfig(
        rnn_config=None if rnn is None else RNNEnsembleConfig.from_dict(rnn),
        use_cnn=use_cnn,
        cnn_config=policy.cnn_config,
        policy_config=policy,
        critic_config=replace(policy, out_dist="Deterministic"),
    )


def _strip_cell(params: dict) -> dict:
    """Move the `RNNActorCritic` variables out of `EnsembleActorCritic`'s ``cell`` scope."""
    out = {}
    for col, tree in params.items():
        tree = tree.get("scan_step", tree)
        out[col] = tree.get("cell", tree)
    return out


def _identity(params: dict) -> dict:
    return params


def _drop_decoder(params: dict) -> dict:
    """Remove the decoder ``dec`` from every collection."""
    return {col: {k: v for k, v in tree.items() if k != "dec"} for col, tree in params.items()}


def _convert_pretrain(params, cons_state, hparams: dict, with_decoder: bool):
    """Convert an RTR-IIL ``(policy, [autoencoder], [critic])`` bundle."""
    use_autoencoder = hparams.get("use_autoencoder", False)
    train_critic = hparams.get("train_critic")
    if train_critic == "Q":
        warnings.warn("Dropping the pretrained Q critic, RNNActorCritic has a state-value critic.")

    def convert(bundle):
        parts = list(bundle) if isinstance(bundle, (list, tuple)) else [bundle]
        policy = parts.pop(0)
        autoencoder = parts.pop(0) if use_autoencoder else {}
        critic = parts.pop(0) if train_critic else {}
        out = {}
        for col in set(policy) | set(autoencoder) | set(critic):
            tree = {}
            if col in policy:
                tree["actor"] = policy[col]
            if "enc" in autoencoder.get(col, {}):
                tree["enc"] = autoencoder[col]["enc"]
            if with_decoder and "dec" in autoencoder.get(col, {}):
                tree["dec"] = autoencoder[col]["dec"]
            if col in critic and train_critic != "Q":
                tree["critic"] = critic[col].get("rnn", critic[col])
            out[col] = tree
        return out

    policy = _policy_config(hparams.get("policy_config"))
    config = ActorCriticConfig(policy_config=policy, critic_config=_policy_config(hparams.get("critic_config")))
    if use_autoencoder:
        ae = AutoencoderConfig.from_dict(hparams["autoencoder_cfg"])
        config = replace(
            config,
            use_cnn=True,
            cnn_config=ae.encoder_cfg,
            policy_config=replace(policy, latent_size=ae.latent_size, use_cnn=False),
        )
    return convert(params), _map_cons_state(cons_state, convert), config


def restore_actor_critic_config(hparams: dict) -> ActorCriticConfig:
    """Read the architecture of a checkpoint from its saved hparams, without loading parameters.

    Parameters
    ----------
    hparams : dict
        Contents of ``hparams.json``.

    Returns
    -------
    ActorCriticConfig
        Architecture of the saved model.
    """
    fmt = detect_format(hparams)
    if fmt == MODEL_FORMAT:
        return ActorCriticConfig.from_dict(hparams["actor_critic_config"])
    if fmt == "rtrrl":
        return _rtrrl_config(hparams)
    if fmt == "ppo_ensemble":
        return _ppo_ensemble_config(hparams)
    if fmt == "pretrain":
        return _convert_pretrain({}, None, hparams, with_decoder=False)[2]
    raise ValueError("Classic PPO (ActorCriticRNN) checkpoints have no RNNActorCritic architecture.")


def restore_actor_critic(path: str, with_decoder: bool = False) -> RestoredActorCritic:
    """Load an actor-critic checkpoint of any trainer in the `RNNActorCritic` layout.

    Parameters
    ----------
    path : str
        Checkpoint directory, or a wandb artifact prefixed with ``wandb:``.
    with_decoder : bool, optional
        Also return the decoder ``dec`` of a pretrained autoencoder, e.g. to resume its training.

    Returns
    -------
    RestoredActorCritic
        Variables, architecture and consolidation state.

    Raises
    ------
    ValueError
        For classic PPO checkpoints (`ActorCriticRNN`), whose architecture has no `RNNActorCritic`
        counterpart. These still load with ``jax_rl_util.baselines.ppo_rnn`` itself.
    """
    if path.startswith("wandb:"):
        path = restore_remote(path)
    hparams = restore_config(path)
    fmt = detect_format(hparams)
    if fmt == "ppo":
        raise ValueError(
            f"{path} is a classic PPO (ActorCriticRNN) checkpoint, which can't be converted to RNNActorCritic."
        )
    params, cons_state = _split_cons_state(_restore_raw(path))

    if fmt == "pretrain":
        variables, cons, config = _convert_pretrain(params, cons_state, hparams, with_decoder)
        return RestoredActorCritic(variables, config, cons, fmt, hparams)

    if fmt == MODEL_FORMAT:
        config = ActorCriticConfig.from_dict(hparams["actor_critic_config"])
        convert = _identity if with_decoder else _drop_decoder
    elif fmt == "rtrrl":
        config = _rtrrl_config(hparams)
        convert = _identity
    else:  # ppo_ensemble
        config = _ppo_ensemble_config(hparams)
        convert = _strip_cell
    return RestoredActorCritic(convert(params), config, _map_cons_state(cons_state, convert), fmt, hparams)


def actor_variables(restored: RestoredActorCritic) -> dict:
    """Get the actor of a restored checkpoint as the variables of a standalone `PolicyRNN`.

    The imitation trainers train a bare `PolicyRNN` on observations, which is the ``actor`` of an
    `RNNActorCritic` without encoder and shared RNN.

    Parameters
    ----------
    restored : RestoredActorCritic
        Result of `restore_actor_critic`.

    Returns
    -------
    dict
        Flax variables ``{collection: actor_tree}`` for ``PolicyRNN(a_dim, restored.config.policy_config)``.

    Raises
    ------
    ValueError
        If the actor reads the output of an encoder or shared RNN, or the observation is appended
        to its input (``skip_actor``), so it can't run on its own.
    """
    params = restored.variables.get("params", {})
    shared = [k for k in ("enc", "rnn") if k in params]
    if shared or restored.config.skip_actor:
        reason = f"has {' and '.join(shared)}" if shared else "uses skip_actor"
        raise ValueError(
            f"The {restored.source_format} checkpoint {reason}, so its actor can't be used as a standalone policy."
        )
    return {col: tree["actor"] for col, tree in restored.variables.items() if "actor" in tree}
