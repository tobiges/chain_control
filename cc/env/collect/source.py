from typing import Optional, Tuple

import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from cc.core.abstract import AbstractController

from ...core import AbstractObservationReferenceSource
from ...core.types import (
    BatchedTimeSeriesOfAct,
    BatchedTimeSeriesOfRef,
    PyTree,
    TimeSeriesOfRef,
)
from ...utils import to_jax, to_numpy, tree_slice


class ObservationReferenceSource(AbstractObservationReferenceSource):
    def __init__(
        self,
        yss: BatchedTimeSeriesOfRef,
        ts: Optional[np.ndarray] = None,
        uss: Optional[BatchedTimeSeriesOfAct] = None,
        i_actor: int = 0,
    ):
        self._i_actor = i_actor
        self._ts = to_numpy(ts) if ts is not None else None
        self._yss = to_numpy(yss)
        self._uss = to_numpy(uss) if uss is not None else None

    def get_references(self) -> BatchedTimeSeriesOfRef:
        return jtu.tree_map(np.atleast_3d, self._yss)

    def get_references_for_optimisation(self) -> BatchedTimeSeriesOfRef:
        return to_jax(self.get_references())

    def get_reference_actor(self) -> TimeSeriesOfRef:
        return tree_slice(self.get_references_for_optimisation(), start=self._i_actor)

    def change_reference_of_actor(self, i_actor: int) -> None:
        self._i_actor = i_actor

class SourceController(AbstractController):
    source: ObservationReferenceSource
    it: int

    def __init__(
        self,
        source: ObservationReferenceSource,
        it: int = 0,
    ):
        self.source = source
        self.it = it

    def step(
        self, x: PyTree[jnp.ndarray]
    ) -> Tuple["AbstractController", PyTree[jnp.ndarray]]:

        return SourceController(self.source, self.it + 1), jnp.asarray([self.source._uss[0][self.it][0]])


    def reset(self) -> "AbstractController":
        return SourceController(self.source, 0)

    def grad_filter_spec(self):
        return False
        


default_kernel = 0.15 * RBF(0.75)


def draw_u_from_gaussian_process(
    ts, kernel=default_kernel, seed=1, scale_u_by: float = 0.25
):
    ts = ts[:, np.newaxis]
    us = GaussianProcessRegressor(kernel=kernel).sample_y(ts, random_state=seed)
    us = (us - np.mean(us)) / np.std(us)
    us *= scale_u_by
    return us.astype(np.float32)


def draw_u_from_cosines(ts, seed):
    freq = seed
    ts = ts / ts[-1]
    omega = 2 * np.pi * freq
    return np.cos(omega * ts)[:, None] * np.sqrt(freq)


def constant_after_transform_source(
    source: ObservationReferenceSource,
    after_time: Optional[float] = None,
    new_time_limit: Optional[float] = None,
) -> ObservationReferenceSource:

    if after_time is None and new_time_limit is None:
        # do nothing
        return source

    if source._ts is None:
        raise Exception(
            """Explicitly specify the argument `ts` during `ObservationReferenceSource`
            creation else this function can not be used."""
        )

    old_time_limit = source._ts[-1] + source._ts[1]

    if new_time_limit is not None and new_time_limit < old_time_limit:
        raise Exception(
            f"""`new_time_limit` can not be smaller than the old time limit
            but got {new_time_limit} < {old_time_limit}."""
        )

    if new_time_limit is None:
        new_time_limit = old_time_limit

    control_timestep = source._ts[1]

    if after_time is None:
        # never becomes constant
        after_time = new_time_limit - control_timestep

    new_ts = np.arange(0.0, new_time_limit, step=control_timestep)
    # TODO
    # exact equality does not work for some reason..
    switch_idx = np.where(np.isclose(new_ts, after_time))[0][0]

    new_yss = jtu.tree_map(
        lambda arr: np.pad(
            arr[:, :switch_idx],
            ((0, 0), (0, len(new_ts) - switch_idx + 1), (0, 0)),
            mode="edge",
        ),
        source.get_references(),
    )

    return ObservationReferenceSource(new_yss, new_ts, i_actor=source._i_actor)
