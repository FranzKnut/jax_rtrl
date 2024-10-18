from dataclasses import dataclass, field, replace
from functools import partial
from chex import ArrayDevice
import jax
import jax.numpy as np
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax import random
from jax.numpy.linalg import eigh
from jax.scipy.linalg import block_diag
from models.jax_util import ModelConfig, map_nested_fn
from optimizers import OptimizerConfig, make_multi_transform


@dataclass
class S5Config(ModelConfig):
    """S5 Configuration

    Parameters
    ----------
    activation  (string):   Type of activation function to use
    prenorm     (bool):     apply prenorm if true or postnorm if false
    step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                            e.g. after training on a different resolution for
                            the speech commands benchmark
    """

    # Model Parameters
    state_size: int = 256
    n_layers: int = 2
    blocks: int = 8
    C_init: str = "trunc_standard_normal"
    discretization: str = "zoh"
    # mode: str = "pool"
    activation_fn: str = "half_glu1"
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    kwargs: dict = field(default_factory=dict)
    do_norm: bool = False
    prenorm: bool = False
    do_gtrxl_norm: bool = False
    step_rescale: float = 1.0


class SequenceLayer(nn.Module):
    """Defines a single S5 layer, with S5 SSM, nonlinearity, etc.
    Args:
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                we usually refer to this size as H
        config      (S5Config): the SSM config to be used


    """

    d_model: int
    config: S5Config

    def setup(self):
        """Initializes the ssm, layer norm and dense layers"""
        self.seq = init_S5SSM(self.d_model, self.config)(step_rescale=self.config.step_rescale)

        if self.config.activation_fn in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.config.activation_fn in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        self.norm = nn.LayerNorm()

    def __call__(self, hidden, x, d):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
             d (bool): reset signal (L,)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.config.prenorm and self.config.do_norm:
            x = self.norm(x)
        hidden, x = self.seq(hidden, x, d)
        # hidden, x = jax.vmap(self.seq, in_axes=1, out_axes=1)(hidden, x, d)
        # hidden = jnp.swapaxes(hidden, 1, 0)
        if self.config.do_gtrxl_norm:
            x = self.norm(x)

        if self.config.activation_fn in ["full_glu"]:
            x = nn.gelu(x)
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        elif self.config.activation_fn in ["half_glu1"]:
            x = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x))
        elif self.config.activation_fn in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x1))
        elif self.config.activation_fn in ["gelu"]:
            x = nn.gelu(x)
        else:
            raise NotImplementedError("Activation: {} not implemented".format(self.config.activation_fn))

        x = skip + x
        if not self.config.prenorm and self.config.do_norm:
            x = self.norm(x)
        return hidden, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jnp.zeros((1, batch_size, hidden_size), dtype=jnp.complex64)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initialize the learnable timescale Delta by sampling
    uniformly between dt_min and dt_max.
    Args:
        dt_min (float32): minimum value
        dt_max (float32): maximum value
    Returns:
        init function
    """

    def init(key, shape):
        """Init function
        Args:
            key: jax random key
            shape tuple: desired shape
        Returns:
            sampled log_step (float32)
        """
        return random.uniform(key, shape) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """Initialize an array of learnable timescale parameters
    Args:
        key: jax random key
        input: tuple containing the array shape H and
               dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
    """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax random key to be used with init function.
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H,2)
    """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """Sample C with a truncated normal distribution with standard deviation 1.
    Args:
        key: jax random key
        shape (tuple): desired shape, of length 3, (H,P,_)
    Returns:
        sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax random key to be used with init function.
         shape (tuple): desired shape  (H,P)
         V: (complex64)     the eigenvectors used for initialization
     Returns:
         C_tilde (complex64) of shape (H,P,2)
    """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def binary_operator_diag_spatial(q_i, q_j):
    """Same as above but stop the gradient for the recurrent connection"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, jax.lax.stop_gradient(A_j * b_i) + b_j


@jax.vmap
def binary_operator_reset(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i) * (1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )


class S5SSM(nn.Module):
    Lambda_re_init: ArrayDevice
    Lambda_im_init: ArrayDevice
    V: ArrayDevice
    Vinv: ArrayDevice

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
        the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2 * self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B", lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv), B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError("C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                self.C2 = self.param("C2", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, hidden, input_sequence, resets):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
             resets (bool): input sequence (L,)
        Returns:
            output sequence (float32): (L, H)
        """
        Lambda_elements = self.Lambda_bar * jnp.ones((*input_sequence.shape[:-1], self.Lambda_bar.shape[0])).reshape(
            -1, self.Lambda_bar.shape[0]
        )
        Bu_elements = jax.vmap(lambda u: self.B_bar @ u)(input_sequence.reshape(-1, input_sequence.shape[-1]))

        Lambda_elements = jnp.concatenate(
            [
                jnp.ones((1, self.Lambda_bar.shape[0])),
                Lambda_elements,
            ]
        )

        Bu_elements = jnp.concatenate(
            [
                hidden.reshape(-1, hidden.shape[-1]),
                Bu_elements,
            ]
        )

        if resets is None:
            _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
        else:
            resets = jnp.concatenate(
                [
                    jnp.zeros(1),
                    resets,
                ]
            )
            _, xs, _ = jax.lax.associative_scan(binary_operator_reset, (Lambda_elements, Bu_elements, resets))
        xs = xs[1:]

        if self.conj_sym:
            hidden = xs[..., -1, :]
            ys = jax.vmap(lambda x: 2 * (self.C_tilde @ x).real)(xs).reshape(input_sequence.shape)
        else:
            hidden = xs[..., -1, :]
            ys = jax.vmap(lambda x: (self.C_tilde @ x).real)(xs).reshape(input_sequence.shape)

        # Add feedthrough matrix output Du;
        input_time_dim = input_sequence.reshape(-1, input_sequence.shape[-1])
        Du = jax.vmap(lambda u: self.D * u)(input_time_dim).reshape(input_sequence.shape)
        return hidden, ys + Du


def init_S5SSM(d_model: int, config: S5Config):
    """Convenience function that will be used to initialize the SSM."""
    P = config.state_size
    H = d_model
    block_size = int(P / config.blocks)

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if config.conj_sym:
        block_size = block_size // 2
        P = P // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((config.blocks, block_size))).ravel()
    V = block_diag(*([V] * config.blocks))
    Vinv = block_diag(*([Vc] * config.blocks))

    # print("Lambda.shape={}".format(Lambda.shape))
    # print("V.shape={}".format(V.shape))
    # print("Vinv.shape={}".format(Vinv.shape))

    return partial(
        S5SSM,
        H=H,
        P=P,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=config.C_init,
        discretization=config.discretization,
        dt_min=config.dt_min,
        dt_max=config.dt_max,
        conj_sym=config.conj_sym,
        clip_eigs=config.clip_eigs,
        bidirectional=config.bidirectional,
        **config.kwargs,
    )


def make_HiPPO(N):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


class StackedEncoderModel(nn.Module):
    """Defines a stack of S5 layers to be used as an encoder.
    Args:
        config         (nn.Module): the SSM to be used (i.e. S5 ssm)
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                 we usually refer to this size as H
        n_layers    (int32):    the number of S5 layers to stack
        activation  (string):   Type of activation function to use
        prenorm     (bool):     apply prenorm if true or postnorm if false
    """

    d_model: int
    config: S5Config

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                config=self.config,
                d_model=self.d_model,
            )
            for _ in range(self.config.n_layers)
        ]

    def __call__(self, hidden, x, d=None):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        x = self.encoder(x)
        new_hiddens = []
        for i, layer in enumerate(self.layers):
            new_h, x = layer(hidden[i], x, d)
            new_hiddens.append(new_h)

        return new_hiddens, x

    @nn.nowrap
    def initialize_carry(self, rng, input_shape):
        # Use a dummy key since the default state init fn is just zeros.
        local_P = self.config.state_size // 2 if self.config.conj_sym else self.config.state_size
        return [jnp.zeros((*input_shape[:-1], local_P), dtype=jnp.complex64) for _ in range(self.config.n_layers)]
