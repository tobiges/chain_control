import matplotlib
import optax
import ray
from ray import air, tune
from cc.collect import *
from cc import save, load
from cc.controller import create_pi_controller, LinearController, LinearControllerOptions
from cc.model import LinearModel, LinearModelOptions, NonlinearModelOptions, NonlinearModel
from cc.env.model_based_env import ModelBasedEnv
from cc.env.wrappers import AddReferenceObservationWrapper, RecordVideoWrapper
from cc.env import make_env
from cc.train import TrainingOptionsController, train_controller
from cc.utils.utils import generate_ts, extract_timelimit_timestep_from_env
import jax.random as jrand
from cc.train import train_model, TrainingOptionsModel
from typing import TypeVar
import pprint
import numpy
import numpy as np
import cloudpickle

T = TypeVar("T")

time_limit = 10.0
control_timestep = 0.01
ts = generate_ts(time_limit, control_timestep)

env = make_env("two_segments_v1", random=1, time_limit=time_limit, control_timestep=control_timestep)
source = collect_reference_source(env, seeds=[20], constant_after=True, constant_after_T=3.0)

model = load("model_for_two_segments_v1.pkl")

def get_controller(p_gain = 0.01, i_gain = 0.0):
    # Training
    options = create_pi_controller(p_gain, i_gain, delta_t=control_timestep)
    controller = LinearController(options)

    training_options = TrainingOptionsController(
        optax.adam(3e-3), 0.0, 500, 1, models=[model]
    )

    controller, losses = train_controller(controller, source, training_options)
    return controller, f"p_gain: {p_gain} i_gain: {i_gain}"

def train_judge(config: dict[str, T]) -> float:
    controller, s = get_controller(config["p"], config["i"])

    env_with_ref = AddReferenceObservationWrapper(env, source)

    it = collect(env=env_with_ref, controller=controller, ts=ts)
    replay_sample = next(it)
    pp.pprint(replay_sample)
    aggr_rew = np.sum(np.abs(replay_sample.rew))
    return aggr_rew

def force_cpu_backend():
    from jax import config
    config.update("jax_platform_name", "cpu")
    config.update("jax_platforms", "cpu")


def objective(config):
    force_cpu_backend()
    # Otherwise the stdout will be messy
    disable_tqdm()
    disable_compile_warn()

    return train_judge(config)


ray.init()

search_space = {
    "p": tune.grid_search([0.1, 0.5, 0.9]),
    "i": tune.grid_search([0.1, 0.5, 0.9]),
}

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 2}),  # <- two cpus per job
    tune_config=tune.TuneConfig(
        num_samples=1,  # <- 1 sample *per* grid point
        time_budget_s=24 * 3600,  # maximum runtime in seconds
        mode="min",  # either `min` or `max`
    ),
    run_config=air.RunConfig(
        log_to_file=("my_stdout.log", "my_stderr.log"), local_dir="ray_results"
    ),
    param_space=search_space,
)

tuner.fit()