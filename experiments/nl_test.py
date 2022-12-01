from typing import Any
from experiments.helpers import calc_reward, get_eval_source, plot_analysis, plot_controller

import numpy as np
import ray
from ray import air, tune

from cc.config import disable_compile_warn, disable_tqdm


# from .neural_ode_controller import make_neural_ode_controller
import equinox as eqx
from cc.env import *
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source, append_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
import optax
import jax.numpy as jnp
import pprint
import numpy as np
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
from cc.env.collect.collect import collect_random_step_source
from controllers.nonlinear_ode import make_neural_ode_controller
#from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller

import matplotlib.pyplot as plt 


from cc.config import force_cpu_backend
force_cpu_backend()
disable_tqdm()
disable_compile_warn()

def train_and_judge(config) -> float:
    env = make_env("two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)

    model = make_neural_ode_model(
        env.action_spec(),
        env.observation_spec(),
        env.control_timestep,
        state_dim=50,
        f_depth=0, 
        u_transform=jnp.arctan
    )

    model = eqx.tree_deserialise_leaves(
        "/data/ba54womo/chain_control/experiments/models/good_env1_model.eqx", model)

    print("test1")

    #source = collect_random_step_source(env, seeds=list(range(25)), amplitude=5.0)
    #for i in range(1, 20):
    #    if i % 2 == 0:
    #        source = append_source(source, sample_feedforward_collect_and_make_source(env, seeds=[i])[0])
    #    else:
    #        source = append_source(source, collect_random_step_source(env, seeds=[i]))

    print("test2")
    # Append 25 easy sources

    #source = collect_random_step_source(env, seeds=list(range(50)))
    
    source, _ = sample_feedforward_collect_and_make_source(env, seeds=list(range(25)))
    for i in range(25):
        source = append_source(source, collect_random_step_source(env, seeds=[i]))

    # if config["constant_after"] > 0:
    #source = constant_after_transform_source(source, after_time=config["constant_after"], offset=10)
    print("test3")

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
        u_transform=config["u_transform"]
    )

    controller_dataloader = make_dataloader(
        source.get_references_for_optimisation(),
        jrand.PRNGKey(1,),
        n_minibatches=5
    )


    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))


    controller_train_options = TrainingOptionsController(
    controller_dataloader, optimizer, )

    controller_trainer = ModelControllerTrainer(
        model, controller, controller_train_options=controller_train_options, 
        trackers=[Tracker("train_mse")]
    )

    print("test4")
    controller_trainer.run(500)
    print("test5")

    fitted_controller = controller_trainer.trackers[0].best_model_or_controller()

    eqx.tree_serialise_leaves("/data/ba54womo/chain_control/experiments/controller_80_0_0_0_0_source_25_mixed.eqx", fitted_controller)

    
    plot_analysis(fitted_controller, env, model, f"source_comp/plot_mixed_source.png")
    eval_source = get_eval_source(5, 5, 5)
    controller_performance_sample = collect_exhaust_source(AddRefSignalRewardFnWrapper(env, eval_source), fitted_controller)
    
    print("test6")
    print(np.sum(np.abs(controller_performance_sample.rew)))
   
    return calc_reward(controller_performance_sample)

def load_and_judge(config) -> float:
    print("creating env and source")
    env = make_env("two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)
    source, _ = sample_feedforward_collect_and_make_source(env, seeds=list(range(5)))
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
        u_transform=config["u_transform"]
    )
    print("loading controller")
    controller = eqx.tree_deserialise_leaves("/data/ba54womo/chain_control/experiments/controller_80_90_2_50_2_sampled_5.eqx", controller)
    print("plotting")
    plot_analysis(controller, env)
    print("rewarding")
    print("dont use not updated")
    controller_performance_sample = collect_exhaust_source(env_w_source, controller)
    return calc_reward(controller_performance_sample)


print(train_and_judge({
    "state_dim": 80,
    "f_width_size": 0,
    "f_depth": 0,
    "g_width_size": 0,
    "g_depth": 0,
    "u_transform": lambda u: u,
}))
