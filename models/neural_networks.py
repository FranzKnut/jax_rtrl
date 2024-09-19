"""Neural networks using the flax package."""

from dataclasses import dataclass, field
from typing import Callable, Tuple
from chex import PRNGKey
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


from .ctrnn.ctrnn import CTRNNCell, OnlineCTRNNCell, rtrl_gradient


"""
Neural network structure.
"""


class FADense(nn.Dense):
    """Dense Layer with feedback alignment."""

    f_align: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.glorot_normal(in_axis=-1, out_axis=-2)

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        if self.f_align:
            B = self.variable(
                "falign",
                "B",
                self.kernel_init,
                self.make_rng() if self.has_rng("params") else None,
                (jnp.shape(x)[-1], self.features),
                self.param_dtype,
            ).value
        else:
            B = self.param(
                "kernel",
                self.kernel_init,
                (jnp.shape(x)[-1], self.features),
                self.param_dtype,
            )

        def f(mdl, x, B):
            return nn.Dense.__call__(mdl, x)

        def fwd(mdl, x, B):
            """Forward pass with tmp for backward pass."""
            return nn.Dense.__call__(mdl, x), (x, B)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(tmp, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _B = tmp
            grads = {"params": {"kernel": jnp.einsum("...X,...Y->YX", y_bar, _x)}}
            if self.use_bias:
                grads["params"]["bias"] = jnp.einsum("...X->X", y_bar)
            # if self.f_align:
            #     grads['params']['B'] = jnp.zeros_like(B)
            x_grad = jnp.einsum("YX,...X->...Y", _B, y_bar)
            return (grads, x_grad, jnp.zeros_like(_B))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)

        return fa_grad(self, x, B)


class MLP(nn.Module):
    """MLP built with Flax.

    activation_fn is applied after every layer except the last one.
    If f_align is true, each layer uses feedback alignment instead of backpropagation."""

    layers: list
    activation_fn: Callable = jax.nn.relu
    f_align: bool = False

    @nn.compact
    def __call__(self, x):
        """Call MLP."""
        for size in self.layers[:-1]:
            x = self.activation_fn(FADense(size, f_align=self.f_align)(x))
        x = FADense(self.layers[-1], f_align=self.f_align)(x)
        return x


class RBFLayer(nn.Module):
    """Gaussian Radial Basis Function Layer."""

    output_size: int
    c_initializer: nn.initializers.Initializer = nn.initializers.normal(1)

    @nn.compact
    def __call__(self, x):
        """Compute the distance to centers."""
        c = self.param("centers", self.c_initializer, (self.output_size, x.shape[-1]))
        beta = self.param("beta", nn.initializers.ones_init(), (self.output_size, 1))
        x = x.reshape(x.shape[:-1] + (1, x.shape[-1]))
        z = jnp.exp(-beta * (x - c) ** 2)
        return jnp.sum(z, axis=-1)


class MLPEnsemble(nn.Module):
    """Ensemble of CTRNN cells."""

    num_modules: int
    model: type = MLP
    out_size: int | None = None
    out_dist: str | None = None
    kwargs: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, rng=None):  # noqa
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
        for i in range(self.num_modules):
            # Loop over rnn submodules
            out = self.model(**self.kwargs, name=f"mlp{i}")(x)
            # Make distribution for each submodule
            if self.out_size is not None:
                out = DistributionLayer(self.out_size, self.out_dist)(out)
            outs.append(out)

        if not self.out_dist:
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-2), *outs)
            if rng is not None:
                outs = jnp.mean(outs, axis=0)
        else:
            # Last dim is batch in distrax
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-1), *outs)
            outs = distrax.MixtureSameFamily(distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs)
            if rng is not None:
                outs = outs.sample(seed=rng)

        return outs


class BPMultiLayerRNN(nn.RNNCellBase):
    """Multilayer RNN."""

    layers: list
    rnn_cls: nn.RNNCellBase = CTRNNCell
    rnn_kwargs: dict = field(default_factory=dict)

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        shapes = zip(self.layers, [input_shape, self.layers[:-1]])
        return [self.rnn_cls(size, **self.rnn_kwargs).initializes_carry(rng, in_size) for size, in_size in shapes]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @nn.compact
    def __call__(self, carries, x):
        """Call MLP."""
        for i, size in enumerate(self.layers):
            carries[i], x = self.rnn_cls(size, **self.rnn_kwargs)(carries[i], x)
        return carries, x


class DFAMultiLayerRNN(nn.RNNCellBase):
    layers: list
    rnn_cls: nn.RNNCellBase = OnlineCTRNNCell
    rnn_kwargs: dict = field(default_factory=dict)


class RNNEnsemble(nn.RNNCellBase):
    """Ensemble of RNN cells."""

    num_modules: int
    model: type = OnlineCTRNNCell
    out_size: int | None = None
    out_dist: str | None = None
    output_layers: tuple[int] | None = None
    kwargs: dict = field(default_factory=dict)

    def setup(self):
        self.rnns = [self.model(**self.kwargs, name=f"rnns_{i}") for i in range(self.num_modules)]
        if self.output_layers:
            self.mlps = [
                MLP(self.output_layers, self.kwargs.get("f_align", False), name=f"mlps_{i}")
                for i in range(self.num_modules)
            ]
        if self.out_size is not None:
            # Make distribution for each submodule
            self.dists = [
                DistributionLayer(self.out_size, self.out_dist, name=f"dists_{i}") for i in range(self.num_modules)
            ]

    def _map_outputs(self, outs):
        if not self.out_dist:
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=0), *outs)
        else:
            # Last dim is batch in distrax
            outs = jax.tree.map(lambda *_x: jnp.stack(_x, axis=-1), *outs)
            outs = distrax.MixtureSameFamily(distrax.Categorical(logits=jnp.zeros(outs.loc.shape)), outs)
        return outs

    def __call__(self, h: jax.Array | None = None, x: jax.Array = None, rng=None, **call_args):  # noqa
        """Call submodules and concatenate output.

        If out_dist is not None, the output will be distribution(s),

        Parameters
        ----------
        h : List
            of rnn submodule states
        x : Array
            input
        rng : PRNGKey, optional,
            if given, returns one value per submodule in order to train them independently,
            If None, mean of submodules or a Mixed Distribution is returned.

        Returns
        -------
        _type_
            _description_
        """
        if h is None:
            h = self.initialize_carry(jax.random.key(0), x.shape)
        outs = []
        carry_out = []
        for i in range(self.num_modules):
            # Loop over rnn submodules
            carry, out = self.rnns[i](h[i], x, **call_args)
            carry_out.append(carry)
            if self.output_layers:
                out = self.mlps[i](out)
            if self.out_size is not None:
                # Make distribution for each submodule
                out = self.dists[i](out)
            outs.append(out)
        # Aggregate outputs
        outs = self._map_outputs(outs)

        if rng is not None:
            outs = outs.sample(seed=rng) if self.out_dist else outs.mean(axis=0)
        return carry_out, outs

    def _loss(self, hidden, loss_fn, target):
        outs = []
        for i in range(self.num_modules):
            out = hidden[i]
            # Loop over rnn submodules
            if self.output_layers:
                out = self.mlps[i](out)
            if self.out_size is not None:
                # Make distribution for each submodule
                out = self.dists[i](out)
            outs.append(out)

        outs = self._map_outputs(outs)
        return loss_fn(outs, target)

    @nn.nowrap
    def loss_and_grad(self, params, loss_fn, carry, target):
        if self.model != OnlineCTRNNCell:
            raise NotImplementedError("Only OnlineCTRNNCell is supported for asynchronuous gradient computation.")
        loss, (grads, df_dh) = jax.value_and_grad(self.apply, argnums=[0, 1])(
            params, [c[0] for c in carry], loss_fn, target, method=self._loss
        )

        for i, (c, d) in enumerate(zip(carry, df_dh)):
            grads["params"][f"rnns_{i}"] = rtrl_gradient(c, d, plasticity=self.kwargs["plasticity"])[0]
        return loss, grads

    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize neuron states."""
        return [self.model(**self.kwargs).initialize_carry(rng, input_shape) for _ in range(self.num_modules)]

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        return 1

    @staticmethod
    def clip_tau(params):
        """HACK: clip tau to > 1.0"""
        for k in params["params"]["rnn"]:
            params["params"]["rnn"][k]["tau"] = jnp.clip(params["params"]["rnn"][k]["tau"], min=1.0)
        return params


class DistributionLayer(nn.Module):
    """Parameterized distribution output layer."""

    out_size: int
    distribution: str = "Normal"
    eps: float = 0.001
    f_align: bool = False

    @nn.compact
    def __call__(self, x):
        """Make the distribution from given vector."""
        if self.distribution == "Normal":
            x = FADense(2 * self.out_size, f_align=self.f_align)(x)
            loc, scale = jnp.split(x, 2, axis=-1)
            return distrax.Normal(loc, jax.nn.sigmoid(scale) + self.eps)
        elif self.distribution == "Categorical":
            out_size = np.prod(self.out_size) if isinstance(self.out_size, tuple) else self.out_size
            x = FADense(out_size, f_align=self.f_align)(x)
            if isinstance(self.out_size, tuple):
                x = x.reshape(self.out_size)
            return distrax.Categorical(logits=x)
        else:
            # Becomes deterministic
            return FADense(self.out_size, f_align=self.f_align)(x)


class ConvEncoder(nn.Module):
    """2D-Convolutional Encoder.

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html"""

    latent_size: int
    c_hid: int = 8

    @nn.compact
    def __call__(self, x):
        """Encode given Image."""
        # Encode observation using CNN
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.gelu(x)
        x = nn.Conv(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(features=4 * self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.gelu(x)
        x = x.flatten()  # Image grid to single feature vector
        return nn.Dense(features=self.latent_size)(x)


class ConvDecoder(nn.Module):
    """2D-Convolutional Decoder.

    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial9/AE_CIFAR10.html"""

    img_shape: tuple[int]
    c_hid: int = 8
    tanh_output: bool = False  # sigmoid otherwise

    @nn.compact
    def __call__(self, x):
        """Decode Image from latent vector."""
        xy_shape = np.array(self.img_shape[:2]) / (2 * 2)  # initial img shape depends on number of layers
        if any(xy_shape != xy_shape.astype(int)):
            raise ValueError("The img x- and y-shapes must be divisible by number of expanding layers.")

        x = nn.Dense(features=int(xy_shape.prod()) * self.c_hid)(x)
        x = nn.relu(x)
        x = x.reshape(*[int(n) for n in xy_shape], -1)
        x = nn.ConvTranspose(features=4 * self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=2 * self.c_hid, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.c_hid, kernel_size=(3, 3), strides=2)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.img_shape[-1], kernel_size=(3, 3))(x)
        x = nn.tanh(x) if self.tanh_output else nn.sigmoid(x)
        return x


@dataclass(frozen=True)
class AutoencoderParams:
    latent_size: int = 128
    c_hid: int = 32


class Autoencoder(nn.Module):
    """Deterministic 2D-Autoencoder for dimension reduction."""

    img_shape: tuple[int]
    config: AutoencoderParams = field(default_factory=AutoencoderParams)

    def setup(self) -> None:
        """Initialize submodules."""
        super().setup()
        self.enc = ConvEncoder(self.config.latent_size, self.config.c_hid)
        self.dec = ConvDecoder(self.img_shape, self.config.c_hid)

    def encode(self, x, *_):
        """Encode given Image."""
        return self.enc(x)

    def decode(self, latent):
        """Decode Image from latent vector."""
        return self.dec(latent)

    def __call__(self, x, *_):
        """Encode then decode. Returns prediction and latent vector."""
        latent = self.enc(x)
        pred = self.dec(latent)
        return pred, latent


class Autoencoder_RNN(nn.Module):
    """Deterministic 2D-Autoencoder that also contains an RNN."""

    img_shape: tuple[int]
    latent_size: int = 128
    num_units: int = 32
    use_cnn: bool = True

    @nn.compact
    def __call__(self, x, a, carry=None):
        """Take in an image, action and hidden state to predict next image."""
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
            pred = ConvDecoder(self.img_shape)(latent)
        else:
            pred = nn.Dense(features=obs_size)(latent)
        return carry, pred, latent


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


class FAAffine(nn.Module):
    """Affine Layer with feedback alignment."""

    features: int
    f_align: bool = True
    offset: int = 0

    @nn.compact
    def __call__(self, x):
        """Make use of randomly initialized Feedback Matrix B when f_align is True."""
        a = self.param("a", nn.initializers.normal(), (self.features,))
        b = self.param("b", nn.initializers.zeros, (self.features,))

        def s(x):
            return x[..., self.offset : self.features + self.offset]

        def f(mdl, x, a, b):
            return a * s(x) + b

        def fwd(mdl, x, a, b):
            """Forward pass with tmp for backward pass."""
            return a * s(x) + b, (x, a)

        # f_bwd :: (c, CT b) -> CT a
        def bwd(res, y_bar):
            """Backward pass that may use feedback alignment."""
            _x, _a = res
            grads = {"params": {"a": s(_x) * y_bar, "b": y_bar}}
            x_bar = jnp.zeros_like(_x)
            x_bar = x_bar.at[..., self.offset : self.features + self.offset].set(
                y_bar if not self.f_align else y_bar * _a
            )
            return (grads, x_bar, jnp.zeros_like(a), jnp.zeros_like(b))

        fa_grad = nn.custom_vjp(f, forward_fn=fwd, backward_fn=bwd)
        return fa_grad(self, x, a, b)
