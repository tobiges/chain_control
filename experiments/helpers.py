import os
from cc.env import make_env
from cc.env.collect.collect import append_source, collect_exhaust_source, collect_random_step_source, sample_feedforward_collect_and_make_source
from matplotlib import pyplot as plt
import numpy as np
import equinox as eqx
from cc.env.sample_envs import TWO_SEGMENT_V1


from cc.env.wrappers.replace_physics_by_model import ReplacePhysicsByModelWrapper

from cc.env.wrappers.add_reference_and_reward import AddRefSignalRewardFnWrapper

from cc.env.collect.source import constant_after_transform_source

global_timelimit = 10.0


def get_eval_source(ff_count=4, caft_count=4, step_count=4, big_step_count=4, f_env=TWO_SEGMENT_V1):
    env = make_env(f_env, random=1, time_limit=global_timelimit, control_timestep=0.01)

    eval_source, _, _ = sample_feedforward_collect_and_make_source(
        env, seeds=list(range(100, 100 + ff_count)))

    for i in range(100 + ff_count, 100 + ff_count + caft_count):
        src, _, _ = sample_feedforward_collect_and_make_source(env, seeds=[i])
        src = constant_after_transform_source(src, after_time=3)
        eval_source = append_source(eval_source, src)

    for i in range(100, 100 + step_count):
        eval_source = append_source(
            eval_source, collect_random_step_source(env, seeds=[i]))

    for i in range(100 + step_count, 100 + step_count + big_step_count):
        eval_source = append_source(
            eval_source, collect_random_step_source(env, seeds=[i], amplitude=15.0))

    return eval_source


def plot_analysis(controller, model, filename, mkdir=True, f_env=TWO_SEGMENT_V1):
    if mkdir:
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)

    plt.clf()
    eval_source = get_eval_source(2, 2, 2, 2, f_env)
    env_w_source = AddRefSignalRewardFnWrapper(make_env(
        f_env, random=1, time_limit=global_timelimit, control_timestep=0.01), eval_source)

    sample_env, _ = collect_exhaust_source(env_w_source, controller)
    sample_env_w_model, _ = (None, None)

    if model is not None:
        local_env = make_env(f_env, random=1,
                             time_limit=global_timelimit, control_timestep=0.01)
        env_w_model = ReplacePhysicsByModelWrapper(local_env, model)
        env_w_model_w_source = AddRefSignalRewardFnWrapper(env_w_model, eval_source)
        sample_env_w_model, _ = collect_exhaust_source(env_w_model_w_source, controller)

    for i in range(8):
        plt.plot(sample_env.obs["obs"]["xpos_of_segment_end"][i], label=f"env obs {i}")
        if sample_env_w_model is not None:
            plt.plot(sample_env_w_model.obs["obs"]
                     ["xpos_of_segment_end"][i], label=f"model obs {i}")
        plt.plot(sample_env.obs["ref"]["xpos_of_segment_end"][i], label=f"ref {i}")

        plt.xlabel("timesteps")
        plt.ylabel(f"x position, r: {calc_reward(sample_env)}")
        plt.legend()
        filename = filename.removesuffix(".png")
        plt.savefig(f"{filename}_source:{i}.png")
        plt.clf()


def calc_reward(replay_sample):
    return np.mean(np.abs(replay_sample.rew))
