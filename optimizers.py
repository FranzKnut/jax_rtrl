"""Util for creating optax optimizers."""


from dataclasses import dataclass, field
import optax


@dataclass(frozen=True, eq=True)
class OptimizerParams:
    """Class representing the parameters for an optimizer.

    Attributes:
        opt_name (str): The name of the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        kwargs (dict): Additional keyword arguments for the optimizer.
        decay_type (str): The type of decay for the learning rate.
        lr_kwargs (dict): Additional keyword arguments for the learning rate decay.
        weight_decay (float): The weight decay for the optimizer.
        gradient_clip (float): The value to clip the gradients.
    """

    opt_name: str = 'sgd'
    learning_rate: float = 1
    kwargs: dict = field(default_factory=dict, hash=False)
    decay_type: str | None = None
    lr_kwargs: dict = field(default_factory=dict, hash=False)
    weight_decay: float = 0.0
    gradient_clip: float | None = None


def make_optimizer(direction="min", optimizer_params=OptimizerParams()) -> optax.GradientTransformation:
    """Make optax optimizer.

    The decorator allows reading scheduled lr from the optimizer state.

    Parameters
    ----------
    learning_rate : float
        initial learning rate
    direction : str, optional
        min or max. Defaults to "min", by default "min"
    opt_name : str, optional
        Name of optimizer, by default 'sgd'
    gradient_clip : int, optional
        Clip gradient norm. Defaults to 0
    lr_decay : int, optional
         Exponential lr decay. Defaults to 1, by default 1
    optimizer_params : dict, optional
        Additional kwargs to the optimizer, by default {}

    Returns
    -------
        optax optimizer
    """
    learning_rate = optimizer_params.learning_rate
    weight_decay = optimizer_params.weight_decay
    if direction in ['max', 'maximize']:
        learning_rate = -learning_rate
    else:
        weight_decay = -weight_decay

    # def decay_mask(tree):
    #     mask = jax.tree.map(lambda _: False, tree)  # Initialize all False

    #     # Only set some leaves to true for RNNs
    #     if isinstance(tree, BaseRNNCell):
    #         mask = eqx.tree_at(lambda x: x.w,  # HERE the leaves for decay are selected
    #                            mask, True)
    #     # elif isinstance(tree, Linear):
    #     #     mask = eqx.tree_at(lambda x: x.W,  # HERE the leaves for decay are selected,
    #     #                        mask, True)
    #     return mask

    decay_type = optimizer_params.decay_type
    if decay_type == 'cosine_warmup':
        # Exponentially decaying learning rate, 100k transition steps since we update at each env step
        learning_rate = optax.warmup_cosine_decay_schedule(learning_rate, **optimizer_params.lr_kwargs)
    elif decay_type == 'exponential':
        """Args:
            init_value: the initial learning rate.
            transition_steps: must be positive. See the decay computation above.
            decay_rate: must not be zero. The decay rate.
            transition_begin: must be positive. After how many steps to start annealing
                (before this many steps the scalar value is held fixed at `init_value`).
            staircase: if `True`, decay the values at discrete intervals.
            end_value: the value at which the exponential decay stops. When
                `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as
                an upper bound. Has no effect when `decay_rate` = 0.
        """
        learning_rate = optax.exponential_decay(learning_rate, **optimizer_params.lr_kwargs)

    # Create optimizer from optax chain
    optimizer = optax.chain(
        # Weight decay
        optax.add_decayed_weights(weight_decay),  # , mask=decay_mask
        # Gradient clipping
        optax.clip_by_global_norm(optimizer_params.gradient_clip) if optimizer_params.gradient_clip else optax.identity(),
        # Optimizer
        getattr(optax, optimizer_params.opt_name)(learning_rate, **optimizer_params.kwargs),
    )
    return optimizer
