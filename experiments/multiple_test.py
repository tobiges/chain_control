import equinox as eqx
import jax.numpy as jnp
import numpy as np

from cc.core import AbstractController
from cc.env import make_env
from cc.env.collect import (
    sample_feedforward_collect_and_make_source,
)
import jax.random as jrand
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

controller_e0 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_master_0.eqx", controller0)
controller_e1 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_master_1.eqx", controller1)
controller_e2 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_master_2.eqx", controller2)
controller_e3 = eqx.tree_deserialise_leaves(
    f"/home/tobi/uni/aibe/chain_control/experiments/models/controller_master_3.eqx", controller3)


wrapper = MultipleControllerWrapper(controller_e0, controller_e1, controller_e2,)

video_env = make_env(f"two_segments_mul", random=1,
                     time_limit=10.0, control_timestep=0.01)
video_source, _, _ = sample_feedforward_collect_and_make_source(video_env, seeds=[100])
video_env_w_source = AddRefSignalRewardFnWrapper(video_env, video_source)

video_wrapped_env = RecordVideoWrapper(
    video_env_w_source, width=1920, height=1080, cleanup_imgs=False, path_to_folder="./new_videos")
controller_performance_sample = collect_exhaust_source(video_wrapped_env, wrapper)