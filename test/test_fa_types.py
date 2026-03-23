import unittest

import jax
import jax.numpy as jnp

from jax_rtrl.models.feedforward import MLPCell
from jax_rtrl.models.seq_models import FAMultiLayerRNN


class TestFAMultiLayerRNNTypes(unittest.TestCase):
    def setUp(self):
        self.x = jnp.ones((4,))
        self.sizes = [7, 6, 5]

    def _init_model(self, fa_type: str):
        model = FAMultiLayerRNN(sizes=self.sizes, rnn_cls=MLPCell, fa_type=fa_type)
        variables = model.init(jax.random.PRNGKey(0), None, self.x)
        return model, variables

    def test_bp_does_not_create_feedback_variables(self):
        model, variables = self._init_model("bp")

        self.assertIn("params", variables)
        self.assertNotIn("falign", variables)

        carry = model.apply(
            variables,
            jax.random.PRNGKey(1),
            self.x.shape,
            method=model.initialize_carry,
        )
        _, y = model.apply(variables, carry, self.x)
        self.assertEqual(y.shape, (self.sizes[-1],))

    def test_fa_creates_expected_feedback_shapes(self):
        _, variables = self._init_model("fa")

        self.assertIn("falign", variables)
        self.assertEqual(variables["falign"]["B0"].shape, (4, 6))
        self.assertEqual(variables["falign"]["B1"].shape, (7, 5))
        self.assertEqual(variables["falign"]["B2"].shape, (6, 5))

    def test_dfa_creates_expected_feedback_shapes(self):
        _, variables = self._init_model("dfa")

        self.assertIn("falign", variables)
        self.assertEqual(variables["falign"]["B0"].shape, (4, 5))
        self.assertEqual(variables["falign"]["B1"].shape, (7, 5))
        self.assertEqual(variables["falign"]["B2"].shape, (6, 5))

    def test_unknown_fa_type_raises(self):
        model = FAMultiLayerRNN(sizes=self.sizes, rnn_cls=MLPCell, fa_type="invalid")
        with self.assertRaisesRegex(ValueError, "unknown fa_type: invalid"):
            model.init(jax.random.PRNGKey(0), None, self.x)

    def test_gradients_compute_for_all_fa_types(self):
        grad_sizes = [5, 5, 5]

        for fa_type in ("bp", "fa", "dfa"):
            with self.subTest(fa_type=fa_type):
                model = FAMultiLayerRNN(
                    sizes=grad_sizes,
                    rnn_cls=MLPCell,
                    fa_type=fa_type,
                )
                variables = model.init(jax.random.PRNGKey(0), None, self.x)
                carry = model.apply(
                    variables,
                    jax.random.PRNGKey(1),
                    self.x.shape,
                    method=model.initialize_carry,
                )

                def loss_fn(v):
                    _, y = model.apply(v, carry, self.x)
                    return jnp.sum(y**2)

                grads = jax.grad(loss_fn)(variables)
                grad_norms = [jnp.linalg.norm(g) for g in jax.tree.leaves(grads["params"])]

                self.assertTrue(all(jnp.isfinite(g) for g in grad_norms))
                self.assertTrue(any(g > 0 for g in grad_norms))

                if fa_type in ("fa", "dfa"):
                    for g in jax.tree.leaves(grads["falign"]):
                        self.assertTrue(jnp.allclose(g, 0.0))


if __name__ == "__main__":
    unittest.main()