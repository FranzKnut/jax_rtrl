"""Checkpointing utilities using Orbax."""

import json
from dataclasses import asdict
import os
import jax
from orbax import checkpoint


def restore_remote(artifact_id: str):
    """Try to download wandb artifact if needed. Returns local path.

    Also checks if the path exists locally."""
    import wandb

    # Get from wandb
    path = artifact_id.replace("wandb:", "")
    if wandb.run:
        print(f"Linked {path} to current wandb run.")
        ancestor = wandb.run.use_artifact(path)
    else:
        api = wandb.Api()
        ancestor = api.artifact(path)
    restore_path = os.path.join("artifacts/restored", path)
    if not os.path.exists(restore_path):
        restore_path = ancestor.download(root=restore_path)
    if not os.path.exists(restore_path):
        raise FileNotFoundError(f"Checkpoint not found: {restore_path}")
    return restore_path


def restore_config(path):
    """Restore config from checkpoint."""
    path = os.path.abspath(path)
    hparams_file_path = os.path.join(path, "hparams.json")

    if os.path.exists(hparams_file_path):
        with open(hparams_file_path) as f:
            restored_hparams = json.load(f)
    else:
        restored_hparams = {}
    return restored_hparams


def restore_params(path, tree=None):
    """Restore parameters from orbax checkpoint."""
    path = os.path.abspath(path)
    orbax_path = os.path.join(path, "ckpt")

    checkpointer = checkpoint.StandardCheckpointer()
    try:
        params = checkpointer.restore(
            orbax_path,
            jax.tree_util.tree_map(checkpoint.utils.to_shape_dtype_struct, tree),
        )
    except FileNotFoundError:
        print(f"Checkpoint not found at {orbax_path}. Returning None.")
        params = None
    return params


def restore_params_and_config(path, tree=None):
    """Restore params and config from checkpoint."""
    params = restore_params(path, tree)
    config = restore_config(path)
    return params, config


def checkpointing(path, fresh=False, hparams: dict = None, tree=None):
    """Set up checkpointing at given path.

    Parameters
    ----------
    path : str
        Path to the checkpoint directory.
    fresh : bool, optional
        If True, overwrite existing checkpoint. Default is False.
    hparams : dict, optional
        Hyper-parameters to be saved alongside model params.
    tree : PyTree, optional
        A PyTree structure that matches the parameters to be restored.
        See `orbax.checkpoint.PyTreeCheckpointer.restore` for details.

    Returns
    -------
    tuple
        A tuple containing:
            - params : PyTree or None
                Restored parameters, or None if no checkpoint found or fresh is True.
            - hparams : dict
                Restored or provided hyper-parameters.
        save_model : Callable
            Function (PyTree -> None) for saving given PyTree.
    """
    path = os.path.abspath(path)
    hparams_file_path = os.path.join(path, "hparams.json")

    checkpointer = checkpoint.StandardCheckpointer()
    orbax_path = os.path.join(path, "ckpt")

    def save_model(_params):
        _params = jax.tree.map(
            lambda x: jax.device_put(x, jax.devices("cpu")[0]), _params
        )
        out = checkpointer.save(orbax_path, _params, force=True)
        checkpointer.wait_until_finished()
        return out

    restored_params = None
    restored_hparams = {}
    print(path, end=": ")
    exists = os.path.exists(path)
    if not exists:
        print("No checkpoint found")
    else:
        if fresh:
            print("Overwriting existing checkpoint")
        else:
            restored_params, restored_hparams = restore_params_and_config(path, tree)
            print("Restored checkpoint")

    if (not exists or fresh) and hparams is not None:
        os.makedirs(path, exist_ok=True)
        if not isinstance(hparams, dict):
            # Try to convert to dict
            hparams = asdict(hparams)
        with open(hparams_file_path, "w") as f:
            json.dump(hparams, f)

    return (restored_params, restored_hparams), save_model
