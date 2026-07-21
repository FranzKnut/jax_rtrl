"""Simple tests for RNNEnsemble features."""

import unittest
import jax.random as jrandom

from jax_rtrl.models.seq_models import RNNEnsemble, RNNEnsembleConfig, SequenceLayerConfig


class TestRNNEnsembleConfig(unittest.TestCase):
    """Test RNNEnsembleConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RNNEnsembleConfig()
        self.assertIsNone(config.model_name)
        self.assertEqual(config.num_modules, 1)
        self.assertEqual(config.num_layers, 1)
        self.assertIsNone(config.hidden_size)
        self.assertEqual(config.ensemble_method, "mean")
        self.assertEqual(config.num_blocks, 1)

    def test_layers_property_with_hidden_size(self):
        """Test layers property when hidden_size is specified."""
        config = RNNEnsembleConfig(
            hidden_size=64,
            num_layers=3,
            model_name="bptt",
        )
        # layers should be None since _layers is None
        self.assertIsNone(config.layers)
        # But num_layers should be used
        self.assertEqual(config.num_layers, 3)

    def test_layers_property_with_explicit_layers(self):
        """Test layers property when _layers is explicitly specified."""
        config = RNNEnsembleConfig(
            _layers=(32, 64, 128),
            model_name="bptt",
        )
        self.assertEqual(config.layers, (32, 64, 128))
        self.assertEqual(config.num_layers, 3)
        self.assertIsNone(config.hidden_size)

    def test_invalid_model_name_raises(self):
        """Test that invalid model_name raises an error."""
        with self.assertRaises(AssertionError):
            RNNEnsembleConfig(model_name="invalid_model")

    def test_kalman_fusion_requires_normal_dist(self):
        """Test that kalman fusion requires Normal output distribution."""
        with self.assertRaises(AssertionError):
            RNNEnsembleConfig(
                model_name="bptt",
                hidden_size=32,
                ensemble_method="kalman",
                out_dist="Categorical",
            )

    def test_s5_config_handling(self):
        """Test S5 config handling in rnn_kwargs."""
        config = RNNEnsembleConfig(
            model_name="s5",
            hidden_size=32,
            rnn_kwargs={"d_model": 32, "n_layers": 2},
        )
        # rnn_kwargs should be converted to S5Config
        self.assertIn("config", config.rnn_kwargs)


class TestRNNEnsembleInitialization(unittest.TestCase):
    """Test RNNEnsemble initialization."""

    def test_basic_initialization(self):
        """Test basic RNNEnsemble initialization with bptt model."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        # Initialize parameters
        input_shape = (8,)  # input size
        params = model.init(rng, input_shape)
        
        self.assertIn("params", params)
        self.assertIn("ensemble", params["params"])

    def test_initialization_with_mlp_output_layers(self):
        """Test initialization with MLP output layers."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            output_layers=(32, 16),
            num_modules=1,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        input_shape = (8,)
        params = model.init(rng, input_shape)
        
        # Should have mlps_out in params
        self.assertIn("mlps_out", params["params"])

    def test_initialization_without_model_name(self):
        """Test initialization without model_name (MLP mode)."""
        config = RNNEnsembleConfig(
            model_name=None,
            num_modules=2,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        input_shape = (8,)
        params = model.init(rng, input_shape)
        
        # Should still work, just without RNN ensembles
        self.assertIn("params", params)

    def test_carry_initialization(self):
        """Test carry (hidden state) initialization."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        # Initialize parameters
        params = model.init(rng, (8,))
        
        # Initialize carry
        input_shape = (8,)  # input size
        carry = model.apply(
            params,
            rng,
            input_shape,
            method=model.initialize_carry,
        )
        
        # Carry should be a nested structure
        self.assertIsNotNone(carry)


class TestRNNEnsembleForward(unittest.TestCase):
    """Test RNNEnsemble forward pass."""

    def test_single_step_forward(self):
        """Test single step forward pass."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=1,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        # Create input
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        # Forward pass
        carry, output = model.apply(params, carry, x, training=True)
        
        # Check output shape
        self.assertEqual(output.loc.shape, (1, 4))  # batch=1, out_size=4

    def test_multi_module_forward(self):
        """Test forward pass with multiple modules."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=3,
            ensemble_method="mean",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Output should be combined from 3 modules
        self.assertEqual(output.loc.shape, (1, 4))

    def test_ensemble_with_output_layers(self):
        """Test ensemble with output MLP layers."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
            output_layers=(32,),
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        self.assertEqual(output.loc.shape, (1, 4))

    def test_split_input_mode(self):
        """Test split_input mode."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=3,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4, split_input=True)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        # Input must match num_modules
        x = jrandom.normal(jrandom.PRNGKey(1), (3, 8))  # batch=3, input=8
        
        carry, output = model.apply(params, carry, x, training=True)
        
        self.assertEqual(output.loc.shape, (1, 4))


class TestRNNEnsembleMethods(unittest.TestCase):
    """Test different ensemble methods."""

    def test_mean_ensemble_method(self):
        """Test mean ensemble method."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=3,
            ensemble_method="mean",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Output should be the mean of all modules
        self.assertEqual(output.loc.shape, (1, 4))

    def test_linear_ensemble_method(self):
        """Test linear ensemble method."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
            ensemble_method="linear",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Output should be linearly combined
        self.assertEqual(output.loc.shape, (1, 4))
        # Should have combine_layer in params
        self.assertIn("combine_layer", params["params"])

    def test_dist_ensemble_method(self):
        """Test dist (UniformMixture) ensemble method."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
            ensemble_method="dist",
            out_dist="Normal",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Output should be a UniformMixture distribution
        self.assertEqual(output.loc.shape[0], 2)  # num_modules

    def test_no_ensemble_method(self):
        """Test with ensemble_method=None (return all distributions)."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
            ensemble_method=None,
            out_dist="Normal",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Should return all distributions without combining
        # Output shape should be (num_modules, batch, out_size)
        self.assertEqual(output.loc.shape[0], 2)


class TestRNNEnsembleInputMasking(unittest.TestCase):
    """Test input masking for ensemble modules."""

    def test_full_visibility(self):
        """Test with full input visibility."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=3,
            ensemble_visible_obs_prob=1.0,
            ensemble_first_full_obs=True,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Should work normally
        self.assertEqual(output.loc.shape, (1, 4))

    def test_partial_visibility(self):
        """Test with partial input visibility."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=3,
            ensemble_visible_obs_prob=0.5,
            ensemble_first_full_obs=True,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Should still work, with some inputs masked
        self.assertEqual(output.loc.shape, (1, 4))


class TestRNNEnsembleOutputDistributions(unittest.TestCase):
    """Test different output distributions."""

    def test_deterministic_output(self):
        """Test deterministic output (no distribution)."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=1,
            out_dist="Deterministic",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # For Deterministic, output should not be a distribution
        # The _postprocessing returns the raw output when out_size is None
        # But we set out_size=4, so it should return distributions
        self.assertIsNotNone(output)

    def test_normal_output(self):
        """Test Normal output distribution."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=2,
            out_dist="Normal",
            ensemble_method="mean",
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Output should be a Normal distribution
        self.assertTrue(hasattr(output, 'loc'))
        self.assertTrue(hasattr(output, 'scale'))
        self.assertEqual(output.loc.shape, (1, 4))

    def test_no_output_size(self):
        """Test RNNEnsemble without output size (no distribution layer)."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=1,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=None)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        # Output should be raw RNN output
        self.assertIsNotNone(output)


class TestRNNEnsembleSequenceProcessing(unittest.TestCase):
    """Test sequence processing with RNNEnsemble."""

    def test_sequence_forward(self):
        """Test forward pass with sequence input."""
        from jax_rtrl.models.seq_models import scan_rnn
        
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=1,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        # Create sequence of inputs
        seq_length = 5
        x_seq = jrandom.normal(jrandom.PRNGKey(1), (seq_length, 8))
        
        # Process sequence
        final_carry, outputs = scan_rnn(
            model, params, x_seq, init_carry=carry, batched=False
        )
        
        # Check outputs shape
        self.assertEqual(outputs.loc.shape[0], seq_length)
        self.assertEqual(outputs.loc.shape[1], 4)

    def test_num_blocks_functionality(self):
        """Test num_blocks parameter for input chunking."""
        config = RNNEnsembleConfig(
            model_name="bptt",
            hidden_size=16,
            num_modules=1,
            num_blocks=2,
        )
        rng = jrandom.PRNGKey(0)
        model = RNNEnsemble(config, out_size=4)
        
        # Input size must be divisible by num_blocks
        params = model.init(rng, (8,))
        input_shape = (8,)
        carry = model.apply(params, rng, input_shape, method=model.initialize_carry)
        
        x = jrandom.normal(jrandom.PRNGKey(1), (8,))
        
        carry, output = model.apply(params, carry, x, training=True)
        
        self.assertEqual(output.loc.shape, (1, 4))


class TestSequenceLayerConfig(unittest.TestCase):
    """Test SequenceLayerConfig."""

    def test_default_sequence_layer_config(self):
        """Test default SequenceLayerConfig values."""
        config = SequenceLayerConfig()
        self.assertEqual(config.dropout, 0.0)
        self.assertIsNone(config.norm)
        self.assertFalse(config.glu)
        self.assertFalse(config.skip_connection)
        self.assertFalse(config.learnable_scale_rnn_out)

    def test_skip_connection_config(self):
        """Test skip connection configuration."""
        config = SequenceLayerConfig(skip_connection=True)
        self.assertTrue(config.skip_connection)


if __name__ == "__main__":
    unittest.main()