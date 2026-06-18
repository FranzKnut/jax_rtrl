import unittest

import jax
import jax.numpy as jnp

from jax_rtrl.models.cells import CELL_TYPES
from jax_rtrl.models.seq_models import RNNEnsembleConfig, SequenceLayerConfig
from jax_rtrl.supervised.supervised_offline import (
    OfflineSupervisedExperiment,
    TrainingConfig as OfflineTrainingConfig,
)
from jax_rtrl.supervised.supervised_online import (
    OnlineSupervisedExperiment,
    TrainingConfig as OnlineTrainingConfig,
)


def _toy_data():
    key = jax.random.PRNGKey(0)
    key_x, key_y = jax.random.split(key)
    x = jax.random.normal(key_x, (1, 4, 3))
    y = jax.random.normal(key_y, (1, 4, 1))
    return x, y, x[0], y[0]


def _rnn_config(model_name: str):
    return RNNEnsembleConfig(
        model_name=model_name,
        layers=(4,),
        num_modules=1,
        num_blocks=1,
        layer_config=SequenceLayerConfig(
            norm=None,
            glu=False,
            skip_connection=False,
        ),
        out_dist="Deterministic",
        output_layers=None,
        fa_type="bp",
    )


class TestSupervised(unittest.TestCase):
    def test_online_gradient_smoke(self):
        data = _toy_data()
        cfg = OnlineTrainingConfig(dataset="sine", num_steps=1)
        grads = OnlineSupervisedExperiment(cfg).gradient_smoke_test(data=data)
        leaves = jax.tree.leaves(grads)
        self.assertGreater(len(leaves), 0)
        self.assertTrue(all(jnp.all(jnp.isfinite(g)) for g in leaves))

    def test_offline_gradient_smoke(self):
        data = _toy_data()
        cfg = OfflineTrainingConfig(dataset="sine", num_steps=1)
        grads = OfflineSupervisedExperiment(cfg).gradient_smoke_test(data=data)
        leaves = jax.tree.leaves(grads)
        self.assertGreater(len(leaves), 0)
        self.assertTrue(all(jnp.all(jnp.isfinite(g)) for g in leaves))

    def test_online_forward_smoke_all_cell_types(self):
        data = _toy_data()
        for model_name in CELL_TYPES:
            with self.subTest(model_name=model_name):
                cfg = OnlineTrainingConfig(
                    dataset="sine",
                    num_steps=1,
                    rnn_config=_rnn_config(model_name),
                )
                preds = OnlineSupervisedExperiment(cfg).forward_smoke_test(data=data)
                self.assertTrue(jnp.all(jnp.isfinite(preds)))

    def test_offline_forward_smoke_all_cell_types(self):
        data = _toy_data()
        for model_name in CELL_TYPES:
            with self.subTest(model_name=model_name):
                cfg = OfflineTrainingConfig(
                    dataset="sine",
                    num_steps=1,
                    rnn_config=_rnn_config(model_name),
                )
                preds = OfflineSupervisedExperiment(cfg).forward_smoke_test(data=data)
                self.assertTrue(jnp.all(jnp.isfinite(preds)))
