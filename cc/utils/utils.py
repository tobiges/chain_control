from collections import defaultdict

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from dm_control.rl import control
from jax.flatten_util import ravel_pytree
from tree_utils import batch_concat

from .sample_from_spec import sample_from_tree_of_specs


def time_limit_from_env(env: control.Environment) -> float:
    return env.control_timestep() * env._step_limit


def timestep_array_from_env(env: control.Environment) -> np.ndarray:
    return np.arange(time_limit_from_env(env), step=env.control_timestep())


def to_jax(tree):
    return jtu.tree_map(jnp.asarray, tree)


def to_numpy(tree):
    return jtu.tree_map(np.asarray, tree)


def make_postprocess_fn(output_specs=None, toy_output=None):
    if output_specs is not None and toy_output is not None:
        raise Exception("Please specifiy either one or the other function argument.")
    if output_specs:
        return ravel_pytree(sample_from_tree_of_specs(output_specs))[1]
    else:
        return ravel_pytree(toy_output)[1]


def l2_norm(vector):
    assert vector.ndim == 1
    return jnp.sqrt(jnp.sum(vector**2))


def l1_norm(vector):
    assert vector.ndim == 1
    return jnp.sum(jnp.abs(vector))


def mae(y, yhat):
    y, yhat = batch_concat(y, 0), batch_concat(yhat, 0)
    return jnp.mean(jnp.abs(y - yhat))


def mse(y, yhat):
    y, yhat = batch_concat(y, 0), batch_concat(yhat, 0)
    return jnp.mean((y - yhat) ** 2)


def weighted_mse(y, yhat, weights):
    y, yhat = batch_concat(y, 0), batch_concat(yhat, 0)
    se = (y - yhat) ** 2
    # moves batchaxis to the right; multiply and sum over it
    sse = jnp.sum(weights * jnp.moveaxis(se, 0, -1), axis=-1)
    return jnp.mean(sse)


def rmse(y, yhat):
    return jnp.sqrt(mse(y, yhat))


def primes(n: int) -> list[int]:
    """Find factorization of integer. Slow implementation."""
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def split_filename(filename: str):
    splits = filename.split(".")
    if len(splits) == 1:
        return filename, None
    elif len(splits) == 2:
        return splits[0], splits[1]
    else:
        raise Exception(f"Extension could not be uniquely inferred from {filename}")


def default_dict():
    return defaultdict(lambda: default_dict())


def default_dict_to_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = default_dict_to_dict(v)
    return dict(d)
