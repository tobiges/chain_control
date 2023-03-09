import jax.numpy as jnp

from cc.core import AbstractController


class MultipleControllerWrapper(AbstractController):
    controllers: tuple[AbstractController]

    def __init__(self, *controllers: AbstractController):
        self.controllers = controllers

    def step(self, x):
        actions = [
            {
                "obs": {"xpos_of_segment_end": s_obs},
                "ref": {"xpos_of_segment_end": s_ref},
            }
            for s_obs, s_ref in zip(
                x["obs"]["xpos_of_segment_end"], x["ref"]["xpos_of_segment_end"]
            )
        ]

        new_controllers, actions = zip(
            *[rgf.step(actions[index]) for index, rgf in enumerate(self.controllers)]
        )
        return (
            MultipleControllerWrapper(*new_controllers),
            jnp.asarray(actions).flatten(),
        )

    def reset(self):
        return MultipleControllerWrapper(*(cntl.reset() for cntl in self.controllers))

    def grad_filter_spec(self):
        return MultipleControllerWrapper(
            *(cntl.grad_filter_spec() for cntl in self.controllers)
        )
