"""Copied and modified from https://github.com/deepmind/acme. v0.4.0"""

import abc
from typing import Any, Dict

import dm_env
import numpy as np


class EnvLoopObserver(abc.ABC):
    """An interface for collecting metrics/counters in EnvironmentLoop."""

    @abc.abstractmethod
    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep) -> None:
        """Observes the initial state."""

    @abc.abstractmethod
    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray
    ) -> None:
        """Records one environment step."""

    @abc.abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Returns metrics collected for the current episode."""
