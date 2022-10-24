from collections import OrderedDict
from typing import Any, Generic, NewType, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp

PyTree = Any
PRNGKey = NewType("PRNGKey", jnp.ndarray)
Module = eqx.Module
ModuleState = NewType("ModuleState", Any)

Observation = NewType("Observation", OrderedDict)
TimeSeriesOfObs = NewType("TimeSeriesOfObs", OrderedDict)
BatchedTimeSeriesOfObs = NewType("BatchedTimeSeriesOfObs", OrderedDict)

Reference = NewType("Reference", OrderedDict)
TimeSeriesOfRef = NewType("TimeSeriesOfRef", OrderedDict)
BatchedTimeSeriesOfRef = NewType("BatchedTimeSeriesOfRef", OrderedDict)

Action = NewType("Action", jnp.ndarray)
TimeSeriesOfAct = NewType("TimeSeriesOfAct", jnp.ndarray)
BatchedTimeSeriesOfAct = NewType("BatchedTimeSeriesOfAct", jnp.ndarray)


T = TypeVar("T")


class NotAParameter(eqx.Module, Generic[T]):
    _: T

    def __call__(self) -> T:
        return self._


class Parameter(eqx.Module, Generic[T]):
    _: T

    def __call__(self) -> T:
        return self._


PossibleParameter = Union[Parameter[T], NotAParameter[T]]