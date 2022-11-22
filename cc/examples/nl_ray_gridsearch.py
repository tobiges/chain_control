from typing import Any

import numpy as np
import ray
from ray import air, tune

from cc.config import disable_compile_warn, disable_tqdm


# from .neural_ode_controller import make_neural_ode_controller
import equinox as eqx
from cc.env import *
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import pprint
import numpy
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
import matplotlib.pyplot as plt 

def force_cpu_backend():
    from jax import config

    config.update("jax_platform_name", "cpu")
    config.update("jax_platforms", "cpu")


def train_and_judge(config) -> float:
    env = make_env("two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)

    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep,
        state_dim=12,
        f_integrate_method="EE",
        f_depth=2,
        f_width_size=25,
        g_depth=0,
        u_transform=jnp.arctan
    )

    model = eqx.tree_deserialise_leaves(
        "/data/ba54womo/chain_control/cc/examples/env1_model.eqx", model)

    source, _ = sample_feedforward_collect_and_make_source(env, seeds=[20])
    source = constant_after_transform_source(source, after_T=3.0)

    env_w_source = AddRefSignalRewardFnWrapper(env, source)

    controller = make_neural_ode_controller(
        env_w_source.observation_spec(),
        env.action_spec(),
        env.control_timestep,
        state_dim=config["state_dim"],
        f_width_size=config["f_width_size"],
        f_depth=config["f_depth"],
        g_width_size=config["g_width_size"],
        g_depth=config["g_depth"],
    )

    controller_dataloader = make_dataloader(
        source.get_references_for_optimisation(),
        jrand.PRNGKey(1,),
        n_minibatches=1
    )


    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))


    controller_train_options = TrainingOptionsController(
    controller_dataloader, optimizer, 
)

    controller_trainer = ModelControllerTrainer(
        model, controller, controller_train_options=controller_train_options, 
        trackers=[Tracker("train_mse")]
    )

    controller_trainer.run(500)

    fitted_controller = controller_trainer.trackers[0].best_model()
    controller_performance_sample = collect_exhaust_source(
        env_w_source, fitted_controller)

    plt.plot(controller_performance_sample.obs["obs"]["xpos_of_segment_end"][0], label=f"state size {config['state_dim']}")
    plt.legend()
    plt.savefig(f"out_{config['state_dim']}_{config['f_width_size']}_{config['f_depth']}_{config['g_width_size']}_{config['g_depth']}.png")
    plt.clf()
    return np.sum(np.abs(controller_performance_sample.rew))


def objective(config):
    force_cpu_backend()
    # Otherwise the stdout will be messy
    disable_tqdm()
    disable_compile_warn()

    return train_and_judge(config)


ray.init()

search_space = {
    "state_dim": tune.grid_search([5, 10, 15, 20, 30, 40, 50]),
    "f_width_size": tune.grid_search([1]),
    "f_depth": tune.grid_search([25]),
    "g_width_size": tune.grid_search([25]),
    "g_depth": tune.grid_search([10]),
}
#state_dim = 5
#f_width_size = 1
#f_depth = 25
#g_width_size = 25
#g_depth = 10

tuner = tune.Tuner(
    tune.with_resources(objective, {"cpu": 1}),
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
ray.shutdown()

# This will create a folder, let's call it `objective_dir` in `ray_results`
# and store all logs in there.
# To later transfer the gridsearch results to your local machine you could e.g. do
# >> tune.ExperimentAnalysis(`objective_dir`).dataframe().to_pickle("./results.pkl")
# Then transfer the `results.pkl` to your local machine and use
# >> pd.read_pickle("results.pkl")
