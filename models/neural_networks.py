"""Neural networks using the flax package."""
from dataclasses import field
from typing import Tuple
from chex import PRNGKey
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom

from models.ctrnn import CTRNNCell, FADense, OnlineCTRNNCell


"""
Neural network structure.
"""


class MLP(nn.Module):
    """MLP built with equinox."""

    layers: list
    f_align: bool = True

    @nn.compact
    def __call__(self, x):
        """Call MLP."""
        for i, size in enumerate(self.layers[:-1]):
            x = jax.nn.elu(FADense(size, f_align=self.f_align)(x))
        x = FADense(self.layers[-1], f_align=self.f_align)(x)
        return x


class RBFLayer(nn.Module):
    """Gaussian Radial Basis Function Layer."""
    output_size: int
    c_initializer: nn.initializers.Initializer = nn.initializers.normal(1)

    @nn.compact
    def __call__(self, x):
        """Compute the distance to centers."""
        c = self.param('centers', self.c_initializer, (self.output_size, x.shape[-1]))
        beta = self.param('beta', nn.initializers.ones_init(), (self.output_size, 1))
        x = x.reshape(x.shape[:-1]+(1, x.shape[-1]))
        z = jnp.exp(-beta * (x - c)**2)
        return jnp.sum(z, axis=-1)


class Ensemble(nn.RNNCellBase):
    """Ensemble of CTRNN cells."""
    out_size: int
    num_modules: int
    model = OnlineCTRNNCell
    out_dist: str | None = None
    kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, h, x, training=False):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters
        ----------
        h : List 
            of rnn submodule states
        x : Array 
            input
        training : bool, optional, by default False
            If true, returns one value per submodule in order to train them independently,
            If false, mean of submodules or a Mixed Distribution is returned.

        Returns
        -------
        _type_
            _description_
        """
        outs = []
        carry_out = []
        for i in range(self.num_modules):
            # Loop over rnn submodules
            carry, out = self.model(**self.kwargs)(h[i], x)
            carry_out.append(carry)
            # Make distribution for each submodule
            out = DistributionLayer(self.out_size, self.out_dist)(out)
            outs.append(out)
        
        outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=0), *outs)
        if not training:
            if not self.out_dist:
                outs = jnp.mean(outs, axis=0)
            else:
                outs = distrax.MixtureSameFamily(distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs)

        return carry_out, outs

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return [self.model(**self.kwargs).initialize_carry(rng, input_shape)] * self.num_modules

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1


class DistributionLayer(nn.Module):
    out_size: int
    distribution: str = 'Normal'
    eps: float = 0.01
    f_align: bool = False

    @nn.compact
    def __call__(self, x):
        if self.distribution == 'Normal':
            x = FADense(2*self.out_size, f_align=self.f_align)(x)
            loc, scale = jnp.split(x, 2, axis=-1)
            return distrax.Normal(loc, jax.nn.softplus(scale)+self.eps)
        elif self.distribution == 'Categorical':
            x = FADense(self.out_size, f_align=self.f_align)(x)
            return distrax.Categorical(logits=x)
        else:
            # Becomes deterministic
            return FADense(self.out_size, f_align=self.f_align)(x)


class ConvEncoder(nn.Module):
    """2D-Convolutional Encoder."""
    c_hid: int = 8
    latent_size: int = 128

    @nn.compact
    def __call__(self, x):
        """Encode given Image."""
        # Encode observation using CNN
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=2*self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=4*self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = x.flatten()  # Image grid to single feature vector
        return nn.Dense(features=self.latent_size)(x)


class ConvDecoder(nn.Module):
    """2D-Convolutional Decoder."""
    c_out: int = 1
    c_hid: int = 32

    @nn.compact
    def __call__(self, x):
        """Decode Image from latent vector."""
        x = nn.Dense(features=12*12*self.c_hid)(x)
        x = nn.relu(x)
        x = x.reshape(12, 12, -1)
        x = nn.ConvTranspose(features=self.c_hid//2, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.c_hid//4, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.c_out, kernel_size=(3, 3), strides=2)(x)
        x = nn.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    """Deterministic 2D-Autoencoder for dimension reduction."""
    latent_size: int = 128

    def setup(self) -> None:
        """Initialize submodules."""
        super().setup()
        self.enc = ConvEncoder(self.latent_size)
        self.dec = ConvDecoder()

    def encode(self, params, x, *_):
        """Encode given Image."""
        return self.bind(params).enc.apply({'params': params['params']['enc']}, x)

    def decode(self, params, x):
        """Decode Image from latent vector."""
        return self.bind(params).dec.apply({'params': params['params']['dec']}, x)

    def __call__(self, x, *_):
        """Encode then decode. Returns prediction and latent vector."""
        latent = self.enc(x)
        pred = self.dec(latent)
        return pred, latent


class Autoencoder_RNN(nn.Module):
    """Deterministic 2D-Autoencoder that also contains an RNN."""
    latent_size: int = 128
    num_units: int = 32
    use_cnn: bool = True

    @nn.compact
    def __call__(self, x, a, carry=None):
        """Takes in an image, action and hidden state to predict next image."""
        # Encode observation using CNN
        obs_size = x.shape[-1]

        if self.use_cnn:
            x = ConvEncoder()(x)

        # Action FC
        a = nn.Dense(features=self.latent_size)(a)
        a = nn.tanh(a)

        # Encoded observation FC
        e = nn.Dense(features=self.latent_size)(x)
        x = nn.tanh(e)

        # Concatenate
        x = jnp.concatenate([x, a], axis=-1)

        # Step LSTM
        rnn = CTRNNCell(self.num_units)
        if carry is None:
            carry = rnn.initialize_carry(jax.random.key(0), x.shape)
        carry, x = rnn(carry, x)

        # Concatenate last LSTM state with latent
        x = jnp.concatenate([x, e], axis=-1)
        latent = nn.Dense(features=self.latent_size)(x)
        if self.use_cnn:
            pred = ConvDecoder()(latent)
        else:
            pred = nn.Dense(features=obs_size)(latent)
        return carry, pred, latent


@jax.value_and_grad
def bptt_loss(params, model: Autoencoder_RNN, trajectories, initial_hidden):
    """Loss for a sequence of Observations."""
    # Transpose batch and sequence dimensions
    trajectories = jax.jax.tree.map(lambda x: x.swapaxes(1, 0), trajectories)

    def predict_loop(hidden, batch):
        hidden, pred, _ = model.apply(params, batch.experience['obs'], batch.experience['act'], hidden)
        return hidden, pred

    _, predictions = jax.lax.scan(jax.vmap(predict_loop), initial_hidden, trajectories)
    loss = ((predictions[1:-1] - trajectories.experience['obs'][2:]) ** 2).sum()
    return loss


if __name__ == "__main__":
    # Test encoder implementation
    model = Autoencoder_RNN()

    # Random key for initialization
    rng = jrandom.PRNGKey(0)
    # Example images as input
    imgs = jrandom.normal(rng, (32, 32, 1))
    # Example actions as input
    a = jrandom.normal(rng, (3,))
    print("Input shape: ", imgs.shape)

    # Initialize parameters of encoder with random key and images
    out, params = model.init_with_output(rng, imgs, a)
    print("Output shapes: ")
    for k in range(len(out)):
        try:
            print(k, out[k].shape)
        except AttributeError:
            # Assuming its a tuple
            print(k, [o.shape for o in out[k]])
