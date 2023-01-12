

import time
from cc.env import make_env
from cc.env.collect.collect import append_source, collect_exhaust_source, collect_random_step_source, sample_feedforward_collect_and_make_source
from matplotlib import pyplot as plt
import numpy as np
import equinox as eqx

from cc.env.wrappers.replace_physics_by_model import ReplacePhysicsByModelWrapper

from cc.env.wrappers.add_reference_and_reward import AddRefSignalRewardFnWrapper

from cc.env.collect.source import constant_after_transform_source
from cc.examples.neural_ode_controller_compact_example import make_neural_ode_controller

global_timelimit = 10.0

def get_eval_source(ff_count=4, caft_count=4, step_count=4, f_env="two_segments_v1"):
    env = make_env(f_env, random=1, time_limit=global_timelimit, control_timestep=0.01)

    eval_source, _ = sample_feedforward_collect_and_make_source(
        env, seeds=list(range(100, 100 + ff_count)))
    for i in range(100 + ff_count, 100 + ff_count + caft_count):
        src, _ = sample_feedforward_collect_and_make_source(env, seeds=[i])
        src = constant_after_transform_source(src, after_time=3)
        eval_source = append_source(eval_source, src)
    for i in range(100, 100 + step_count):
        eval_source = append_source(
            eval_source, collect_random_step_source(env, seeds=[i]))
    for i in range(100 + step_count, 100 + (2 * step_count)):
        eval_source = append_source(
            eval_source, collect_random_step_source(env, seeds=[i], amplitude=15.0))
    

    # Add two step sources which are not that close to zer
    
    return eval_source

def get_very_special():
    SPECIAL_ENV = make_env(f"two_segments_v1", random=1, time_limit=global_timelimit, control_timestep=0.01)
    SPECIAL_SOURCE, _ = sample_feedforward_collect_and_make_source(SPECIAL_ENV, seeds=[100])
    ENV_W_SPECIAL_SOURCE = AddRefSignalRewardFnWrapper(SPECIAL_ENV, SPECIAL_SOURCE)

    controller = make_neural_ode_controller(
        SPECIAL_SOURCE.observation_spec(),
        SPECIAL_ENV.action_spec(),
        SPECIAL_ENV.control_timestep,
        state_dim=100,
        f_width_size=0,
        f_depth=0,
        g_width_size=0,
        g_depth=0,
    )
    controller = eqx.tree_deserialise_leaves("controller_good.eqx", controller)
    
    print("!aqsdasd")
    SPECIAL_SAMPLE = collect_exhaust_source(ENV_W_SPECIAL_SOURCE, controller)
    return SPECIAL_SAMPLE

import os

def plot_analysis(controller, env, model, filename, mkdir = True, f_env="two_segments_v1"):  # IMPORTANT: change for ray
    if mkdir:
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)

    plt.clf()
    eval_source = get_eval_source(2, 2, 2, f_env)
    env_w_source = AddRefSignalRewardFnWrapper(make_env(f_env, random=1, time_limit=global_timelimit, control_timestep=0.01), eval_source)
    sample_env = collect_exhaust_source(env_w_source, controller)
    sample_env_w_model = None

    if model is not None:
        local_env = make_env(f_env, random=1,
                             time_limit=global_timelimit, control_timestep=0.01)
        env_w_model = ReplacePhysicsByModelWrapper(local_env, model)
        env_w_model_w_source = AddRefSignalRewardFnWrapper(env_w_model, eval_source)
        sample_env_w_model = collect_exhaust_source(env_w_model_w_source, controller)

    for i in range(8):
        plt.plot(sample_env.obs["obs"]["xpos_of_segment_end"][i], label=f"env obs {i}")
        if sample_env_w_model is not None:
            plt.plot(sample_env_w_model.obs["obs"]
                     ["xpos_of_segment_end"][i], label=f"model obs {i}")
        plt.plot(sample_env.obs["ref"]["xpos_of_segment_end"][i], label=f"ref {i}")

        plt.xlabel("time in seconds")
        plt.ylabel(f"x position, r: {calc_reward(sample_env)}")
        plt.legend()
        # image_file_path = f"images/plot_{i}_{time.strftime('%Y-%m-%d-_%H:%M:%S.png')}"
        # if ray is True:
        #     image_file_path = f"plot_{i}_{time.strftime('%Y-%m-%d-_%H:%M:%S.png')}"
        filename = filename.removesuffix(".png")
        plt.savefig(f"{filename}_source:{i}.png")
        plt.clf()


def plot_controller(controller, env, model, image_filepath=f"images/plot_{time.strftime('%Y-%m-%d-_%H:%M:%S.png')}", print_rewards=True, show=False, skip_clear=False):
    if not skip_clear:
        plt.clf()

    if env:
        eval_source = collect_random_step_source(env, seeds=list(range(5)))
        source = constant_after_transform_source(eval_source, after_time=0, offset=10)

        sample_env = collect_exhaust_source(
            AddRefSignalRewardFnWrapper(env, source), controller)

        plt.plot(sample_env.obs["obs"]["xpos_of_segment_end"][0], label="env obs")

        if print_rewards:
            print("Environment reward: ", calc_reward(sample_env))

    # if model:
    #     env = make_env("two_segments_v1", random=1, time_limit=global_timelimit, control_timestep=0.01)
    #     source = collect_random_step_source(env, seeds=list(range(1)))
    #     env_w_source = AddRefSignalRewardFnWrapper(env, source)
    #     model_w_source = ReplacePhysicsByModelWrapper(env_w_source, model)
    #     sample_model = collect_exhaust_source(model_w_source, controller)

    #     plt.plot(sample_model.obs["obs"]["xpos_of_segment_end"][0], label="model obs")

    #     if print_rewards:
    #         print("Model reward: ", calc_reward(sample_model))

    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(image_filepath)


def calc_reward(replay_sample):
    return np.mean(np.abs(replay_sample.rew))
