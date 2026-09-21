"""This test case's own learning-rate schedule for ``adam_variable_lr``."""


import optax

from src import optimizer_config

SCHEDULER_APPLY_EPOCHS = (500, 1000, 2500)
SCHEDULER_GAMMA = 0.5

# +1: the reference steps its scheduler after the update, optax's schedule before it.
_SCHEDULE_BOUNDARIES = {epoch + 1: SCHEDULER_GAMMA for epoch in SCHEDULER_APPLY_EPOCHS}


def build_adam_variable_lr(learning_rate):
    """Plain Adam with the paper's exact three-step exponential learning-rate decay."""

    adam_scaling = optax.scale_by_adam(
        b1=optimizer_config.ADAM_CONFIG.beta_1,
        b2=optimizer_config.ADAM_CONFIG.beta_2,
        eps=optimizer_config.ADAM_CONFIG.epsilon,
        nesterov=False,
    )
    schedule = optax.piecewise_constant_schedule(
        init_value=learning_rate,
        boundaries_and_scales=_SCHEDULE_BOUNDARIES,
    )
    return optax.chain(adam_scaling, optax.scale_by_learning_rate(schedule))
