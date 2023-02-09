import pprint
import sys
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from cc.core import AbstractController
from cc.env import make_env
from cc.env.collect import (
    sample_feedforward_collect_and_make_source,
)
import jax.random as jrand
from cc.env.envs.two_segments import CartParams, Color, JointParams, generate_duplicate_env_config, generate_env_config
from cc.env.sample_envs import TWO_SEGMENT_V1
from cc.env.wrappers import RecordVideoWrapper
from cc.env.wrappers.add_reference_and_reward import AddRefSignalRewardFnWrapper
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller
from cc.examples.neural_ode_model import make_neural_ode_model
from cc.examples.pid_controller import make_pid_controller
from cc.utils.multiple_controller_wrapper import MultipleControllerWrapper
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.collect import collect_exhaust_source, collect_random_step_source, sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
import optax

from experiments.helpers import plot_analysis

pp = pprint.PrettyPrinter(indent=4)

env = make_env(TWO_SEGMENT_V1, random=1, time_limit=10.0, control_timestep=0.01)

source, _, _ = sample_feedforward_collect_and_make_source(env, seeds=[100])
env_w_source = AddRefSignalRewardFnWrapper(env, source)


controller0 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep,
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

controller1 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep,
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

controller2 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep,
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

controller3 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep,
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

controller4 = make_neural_ode_controller(
    env_w_source.observation_spec(),
    env.action_spec(),
    env.control_timestep,
    state_dim=80,
    f_width_size=0,
    f_depth=0,
    g_width_size=0,
    g_depth=0,
)

controller_e0 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_env_all.eqx", controller0)
controller_e1 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_env_1.eqx", controller1)
controller_e2 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_env_2.eqx", controller2)
controller_e3 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_env_3.eqx", controller3)
controller_e4 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_env_4.eqx", controller4)


# assert at least one cli param
assert len(sys.argv) > 1

env_id = sys.argv[1]

if env_id == "e1":
    hinge = JointParams(
        damping=1e-1, springref=0, stiffness=10
    )
elif env_id == "e2":
    hinge = JointParams(
        damping=1e-1, springref=0, stiffness=2
    )
elif env_id == "e3":
    hinge = JointParams(
        damping=3e-2, springref=0, stiffness=10
    )
elif env_id == "e4":
    hinge = JointParams(
        damping=3e2, springref=0, stiffness=2
    )
elif env_id == "env_new":
    hinge = JointParams(
        damping=65e-3, springref=0, stiffness=6
    )


cart = CartParams(
        name="cart",
        slider_joint_params=JointParams(damping=1e-3),
        hinge_joint_params=hinge,
    )

env = generate_env_config(cart)

video_env_config = generate_duplicate_env_config(
    cart,
    6
)


video_env = make_env(video_env_config, random=1, time_limit=10.0, control_timestep=0.01)
video_source, _, _ = sample_feedforward_collect_and_make_source(video_env, seeds=[100])

source_controller = SourceController(video_source)
wrapper = MultipleControllerWrapper(
     source_controller, controller_e0, controller_e1, controller_e2, controller_e3, controller_e4)

video_env_w_source = AddRefSignalRewardFnWrapper(video_env, video_source)
video_wrapped_env = RecordVideoWrapper( 
    video_env_w_source, camera_id="skyview", width=1920, height=1080, cleanup_imgs=False, path_to_folder=f"./images/{env_id}/video/")

controller_performance_sample = collect_exhaust_source(video_wrapped_env, wrapper)

plot_analysis(controller_e0, None, f"images/{env_id}/combined.png",
              mkdir=True, f_env=env)
plot_analysis(controller_e1, None, f"images/{env_id}/single_e1.png",
              mkdir=True, f_env=env)
plot_analysis(controller_e2, None, f"images/{env_id}/single_e2.png",
              mkdir=True, f_env=env)
plot_analysis(controller_e3, None, f"images/{env_id}/single_e3.png",
              mkdir=True, f_env=env)
plot_analysis(controller_e4, None, f"images/{env_id}/single_e4.png",
              mkdir=True, f_env=env)
