"""Tests for the distribution heads used by DistributionLayer."""

import unittest

import distrax
import jax
import jax.numpy as jnp
import numpy as np

from jax_rtrl.models import distributions
from jax_rtrl.models.feedforward import DistributionLayer

OUT_SIZE = 3
X = jax.random.normal(jax.random.key(0), (4, 5))  # Batch of 4 features

# name -> (DistributionLayer kwargs, bounds that samples and modes must respect)
CASES = {
    "Deterministic": ({}, None),
    "Normal": ({"loc_bounds": (-2.0, 3.0), "scale_bounds": (0.01, 1.0)}, None),
    "LogStddevNormal": ({"scale_bounds": -1.0}, None),
    "NormalTanh": ({}, (-1.0, 1.0)),
    "Beta": ({}, (0.0, 1.0)),
    "ScaledNormal": ({"loc_bounds": (-1.0, 2.0)}, None),
    "ScaledLogStddevNormal": ({"loc_bounds": (-1.0, 2.0)}, None),
    "ScaledNormalTanh": (
        {"loc_bounds": ((-1.0, 0.0, -3.0), (2.0, 1.0, 3.0))},
        ((-1.0, 0.0, -3.0), (2.0, 1.0, 3.0)),
    ),
    "ScaledBeta": ({"loc_bounds": 2.0}, (-2.0, 2.0)),
    "Categorical": ({"eps_unimix": 0.01}, None),
    "Bernoulli": ({"eps_unimix": 0.01}, None),
    "TwoHot": ({"loc_bounds": (-5.0, 5.0), "num_bins": 11}, (-5.0, 5.0)),
}


def _make(name, out_size=OUT_SIZE, x=X, **kwargs):
    """Init and apply a DistributionLayer, returning the distribution."""
    layer = DistributionLayer(out_size, distribution=name, **kwargs)
    params = layer.init(jax.random.key(1), x)
    return layer.apply(params, x)


class TestDistributionHeads(unittest.TestCase):
    """Every head builds a usable distribution with the expected shapes."""

    def test_all_heads_sample_and_score(self):
        """Protects against heads with wrong param counts, shapes or out-of-range samples."""
        for name, (kwargs, bounds) in CASES.items():
            with self.subTest(name):
                dist = _make(name, **kwargs)
                self.assertIsInstance(dist, distributions.get_distribution(name))
                sample = dist.sample(seed=jax.random.key(2))
                mode = dist.mode()
                expected = (4,) if name == "Categorical" else (4, OUT_SIZE)
                self.assertEqual(sample.shape, expected)
                self.assertEqual(mode.shape, expected)
                self.assertTrue(jnp.all(jnp.isfinite(dist.log_prob(sample))))
                if bounds is not None:
                    low, high = (np.asarray(b) for b in bounds)
                    for v in (sample, mode):
                        self.assertTrue(np.all((v >= low - 1e-5) & (v <= high + 1e-5)))

    def test_learned_bounds_for_scaled_heads(self):
        """Protects the learned min/max params used when Scaled heads have no loc_bounds."""
        layer = DistributionLayer(OUT_SIZE, distribution="ScaledNormalTanh")
        params = layer.init(jax.random.key(1), X)
        np.testing.assert_allclose(params["params"]["min_val"], -1.0)
        np.testing.assert_allclose(params["params"]["max_val"], 1.0)
        mode = layer.apply(params, X).mode()
        self.assertTrue(jnp.all(jnp.abs(mode) <= 1.0))

    def test_scaled_variance_matches_affine(self):
        """Protects Scaled.variance, which scales the base variance by the squared factor."""
        dist = _make("ScaledNormal", loc_bounds=(-1.0, 3.0))
        np.testing.assert_allclose(
            dist.variance(), 16.0 * dist.distribution.variance(), rtol=1e-5
        )


class TestDiscreteHeads(unittest.TestCase):
    """Unimix, tuple event shapes and KL for Categorical and Bernoulli."""

    def test_unimix_floors_probabilities(self):
        """Protects unimix blending toward the uniform distribution (1/K, or 1/2 for Bernoulli)."""
        eps = 0.1
        x = 100.0 * X  # Saturated logits
        cat = _make("Categorical", x=x, eps_unimix=eps)
        self.assertGreaterEqual(float(cat.probs.min()), eps / OUT_SIZE - 1e-6)
        bern = _make("Bernoulli", x=x, eps_unimix=eps)
        self.assertGreaterEqual(float(bern.probs.min()), eps / 2 - 1e-6)
        self.assertLessEqual(float(bern.probs.max()), 1 - eps / 2 + 1e-6)

    def test_tuple_out_size_keeps_batch_axes(self):
        """Protects the reshape to (batch, *out_size) for factored categorical latents."""
        dist = _make("Categorical", out_size=(2, 4))
        self.assertEqual(dist.logits.shape, (4, 2, 4))
        self.assertEqual(dist.sample(seed=jax.random.key(2)).shape, (4, 2))

    def test_kl_between_heads(self):
        """Protects KL dispatch for subclasses of distrax distributions (world model losses)."""
        for name in ["Normal", "Categorical", "Bernoulli"]:
            with self.subTest(name):
                p = _make(name)
                q = _make(name, x=X + 1.0)
                kl = p.kl_divergence(q)
                self.assertTrue(jnp.all(jnp.isfinite(kl)) and jnp.all(kl >= -1e-6))
                self.assertIsInstance(p, getattr(distrax, name))


class TestTwoHot(unittest.TestCase):
    """Two-hot encoding and the TwoHot head."""

    def test_two_hot_expectation_recovers_value(self):
        """Protects two_hot weights summing to one with expectation equal to the input."""
        bins = jnp.linspace(-5.0, 5.0, 11)
        x = jnp.array([-7.0, -5.0, -1.3, 0.0, 2.25, 5.0, 9.0])
        enc = distributions.two_hot(x, bins)
        np.testing.assert_allclose(enc.sum(-1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(enc @ bins, jnp.clip(x, -5.0, 5.0), atol=1e-5)

    def test_log_prob_peaks_at_mean_of_target(self):
        """Protects TwoHot.log_prob as negative cross entropy against two-hot targets."""
        bins = jnp.linspace(-5.0, 5.0, 11)
        target = jnp.array([1.5])
        logits = jnp.log(distributions.two_hot(target, bins) + 1e-8).reshape(1, -1)
        dist = distributions.TwoHot(logits, 11, -5.0, 5.0)
        np.testing.assert_allclose(dist.mean()[0], target, atol=1e-4)
        self.assertGreater(
            float(dist.log_prob(target).sum()),
            float(dist.log_prob(jnp.array([-3.0])).sum()),
        )

    def test_bounds_stay_static_under_jit(self):
        """Protects against tracers in TwoHot's static fields, which break tree ops on outputs."""
        layer = DistributionLayer(
            OUT_SIZE, distribution="TwoHot", loc_bounds=(-5.0, 5.0), num_bins=11
        )
        params = layer.init(jax.random.key(1), X)
        dist = jax.jit(layer.apply)(params, X)
        self.assertIsInstance(dist.low, float)
        self.assertIsInstance(dist.high, float)


class TestGetDistribution(unittest.TestCase):
    """Name lookup used by DistributionLayer."""

    def test_lookup(self):
        """Protects name resolution: local heads first, distrax fallback, aliases and None."""
        self.assertIs(distributions.get_distribution("Normal"), distributions.Normal)
        self.assertIs(distributions.get_distribution("beta"), distributions.ScaledBeta)
        self.assertIs(distributions.get_distribution("Softmax"), distrax.Softmax)
        self.assertIsNone(distributions.get_distribution(None))
        self.assertIsInstance(_make("Softmax"), distrax.Softmax)
        self.assertEqual(_make(None).shape, (4, OUT_SIZE))


if __name__ == "__main__":
    unittest.main()
