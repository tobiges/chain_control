import time
from cc.env import *
from cc import save, load
from cc.env.collect import collect, collect_exhaust_source,  sample_feedforward_collect_and_make_source
from cc.env.collect.source import *
from cc.env.wrappers import AddRefSignalRewardFnWrapper, RecordVideoWrapper
from cc.examples.pid_controller import make_pid_controller
from cc.train import *
import jax.random as jrand
from cc.utils import rmse, l2_norm
from matplotlib import pyplot as plt
import optax
import jax.numpy as jnp
from cc.examples.neural_ode_model_compact_example import make_neural_ode_model
import equinox as eqx
import pprint
import numpy
import equinox as eqx

from helpers import calc_reward, plot_analysis
pp = pprint.PrettyPrinter(indent=4)
numpy.set_printoptions(threshold=5)

################ PARAMS ###############

env_num = 1
eval_env = 1

# Reference
#p, i, d = 0.01, 0.0, 0.0

# p, i, d = 10.0, -4.0, 2.0
# p, i, d = 0.01, 0.0, 0.0
p, i, d = 0.4518621872932873, 0.3192994958481972, -0.7791261522365616
# p, i, d = -0.4, -0.8, -0.4

# Reference
#ref = list(range(20, 22))

ref = list(range(30))
# ref = list(range(100, 100 + 2))

constant_after = 3.0

iterations = 500

n_minibatches = 5

optimizer = optax.adam(3e-3)


########################################


env = make_env(f"two_segments_v{env_num}", random=1,
               time_limit=20.0, control_timestep=0.01)


model = make_neural_ode_model(
    env.action_spec(),
    env.observation_spec(),
    env.control_timestep,
    state_dim=50,
    f_depth=0,
    u_transform=jnp.arctan
)

model = eqx.tree_deserialise_leaves(
    f"/data/ba54womo/chain_control/experiments/models/good_env1_model.eqx", model)


# env2 = make_env(f"two_segments_v1", random=1, time_limit=10.0, control_timestep=0.01)

source, _ = sample_feedforward_collect_and_make_source(env, seeds=ref)
#source = constant_after_transform_source(source, after_time=constant_after)

env_w_source = AddRefSignalRewardFnWrapper(env, source)

controller = make_pid_controller(p, i, d, env.control_timestep)

controller_dataloader = make_dataloader(
    UnsupervisedDataset(source.get_references_for_optimisation()),
    jrand.PRNGKey(1,),
    n_minibatches=n_minibatches,
)


controller_train_options = TrainingOptionsController(
    controller_dataloader, optimizer,
)

controller_trainer = ModelControllerTrainer(
    model, controller, controller_train_options=controller_train_options,
    trackers=[Tracker("train_mse")]
)

#controller_trainer.run(iterations)
fitted_controller = controller

fitted_controller = eqx.tree_deserialise_leaves("pid_best_controller.eqx", controller)

#eqx.tree_serialise_leaves("pid_best_controller.eqx", fitted_controller)


SPECIAL_ENV = make_env(f"two_segments_v1", random=1, time_limit=20.0, control_timestep=0.01)
SPECIAL_SOURCE, FF_SAMPLE = sample_feedforward_collect_and_make_source(SPECIAL_ENV, seeds=[100])
#SPECIAL_SOURCE = collect.collect_random_step_source(SPECIAL_ENV, seeds=[102], amplitude=15.0)

ENV_W_SPECIAL_SOURCE = AddRefSignalRewardFnWrapper(SPECIAL_ENV, SPECIAL_SOURCE)
SPECIAL_SAMPLE = collect_exhaust_source(ENV_W_SPECIAL_SOURCE, fitted_controller)

env_video = make_env(f"two_segments_v1_ref", random=1, time_limit=20.0, control_timestep=0.01, cps=FF_SAMPLE)

env_w_video_source = AddRefSignalRewardFnWrapper(env_video, SPECIAL_SOURCE)

env1_w_video = RecordVideoWrapper(env_w_video_source, width=1920, height=1080, cleanup_imgs=False, camera_id="lookatchain", path_to_folder="./pidvideo2")
controller_performance_sample = collect_exhaust_source(env1_w_video, fitted_controller)

plot_analysis(fitted_controller, env, model, f"images/PID_CONFIRM_trainenv{env_num}_eval{eval_env}_ref{len(ref)}.png",
                                            False, f"two_segments_v1")

# env_w_video = RecordVideoWrapper(env_w_source, width=1280, height=720, cleanup_imgs=False)
# controller_performance_sample = collect_exhaust_source(env_w_video, fitted_controller)

# pp.pprint(controller_performance_sample)

# import matplotlib.pyplot as plt

# plt.plot(controller_performance_sample.obs["obs"]["xpos_of_segment_end"][0], label="observation")
# plt.plot(controller_performance_sample.obs["ref"]["xpos_of_segment_end"][0], label="reference")
# plt.legend()
# plt.savefig(f"images/pid_{p}_{i}_{d}_ref{ref}_v{env_num}_ca{constant_after}_it{iterations}.png")


# print(np.sum(np.abs(controller_performance_sample.rew)))
# print(np.sum(np.abs(controller_performance_sample.rew)) / len(controller_performance_sample.rew[0]))



# env_longer = make_env(f"two_segments_v{env_num}",
#                       random=1, time_limit=20.0, control_timestep=0.01)
# # source, _ = sample_feedforward_collect_and_make_source(env_longer, seeds=ref)
# source = constant_after_transform_source(source, after_time=constant_after, new_time_limit=20.0)
# env_longer_w_source = AddRefSignalRewardFnWrapper(env_longer, source)

# # plot_analysis(fitted_controller, env)
# # plot_analysis(fitted_controller, env_longer)

# sample_env = collect_exhaust_source(env_w_source, controller)
# i = 0
# plt.plot(sample_env.obs["obs"]["xpos_of_segment_end"][i], label=f"env obs {i}")

# plt.plot(sample_env.obs["ref"]["xpos_of_segment_end"][i], label=f"ref {i}")

# plt.xlabel("time in seconds")
# plt.ylabel(f"x position, r: {calc_reward(sample_env)}")
# plt.legend()
# image_file_path = f"images/plot_{i}_{time.strftime('%Y-%m-%d-_%H:%M:%S.png')}"
# plt.savefig(image_file_path)
# plt.clf()

# sample_env_longer = collect_exhaust_source(env_longer_w_source, controller)
# plt.plot(sample_env_longer.obs["obs"]["xpos_of_segment_end"][i], label=f"env obs {i}")

# plt.plot(sample_env_longer.obs["ref"]["xpos_of_segment_end"][i], label=f"ref {i}")

# plt.xlabel("time in seconds")
# plt.ylabel(f"x position, r: {calc_reward(sample_env_longer)}")
# plt.legend()
# image_file_path = f"images/plot_{i}_{time.strftime('%Y-%m-%d-_%H:%M:%S.png')}"
# plt.savefig(image_file_path)
# plt.clf()
