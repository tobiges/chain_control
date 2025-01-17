from dataclasses import dataclass
from typing import Union

import dm_env
import numpy as np
from flax import struct
from tree_utils import PyTree


@dataclass
class ReplayElement:
    """Atomic transition tuple generated by actor"""

    timestep: dm_env.TimeStep
    action: np.ndarray
    next_timestep: dm_env.TimeStep
    prev: Union[None, "ReplayElement"]
    actor_id: Union[int, None]
    episode_id: int
    timestep_id: int
    extras: PyTree

    def __repr__(self) -> str:
        prev = "ReplayElement(...)" if self.prev else "None"
        return f"ReplayElement(timestep={self.timestep}, action={self.action},\
            next_timestep={self.next_timestep}, prev={prev}, actor_id={self.actor_id},\
                episode_id={self.episode_id}, timestep_id={self.timestep_id}"


@struct.dataclass
class ReplaySample:
    """Accumulated `ReplayElement`s. Can be generated from the buffer and used for
    learning.
    """

    obs: np.ndarray
    action: np.ndarray
    rew: np.ndarray
    done: np.ndarray
    extras: dict

    @property
    def bs(self):
        assert self.action.ndim == 3
        return self.action.shape[0]

    @property
    def n_timesteps(self):
        assert self.action.ndim == 3
        return self.action.shape[1]
