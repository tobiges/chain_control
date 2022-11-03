import matplotlib.pyplot as plt
import optax

from cc.collect import *
from cc import save, load
from cc.controller import create_pi_controller, LinearController, LinearControllerOptions
from cc.env.model_based_env import ModelBasedEnv
from cc.env.wrappers import AddReferenceObservationWrapper, RecordVideoWrapper
from cc.env import make_env
from cc.train import TrainingOptionsController, train_controller
from cc.utils.utils import generate_ts
import numpy as np
import pprint

time_limit = 10.0
control_timestep = 0.01
ts = generate_ts(time_limit, control_timestep)

# Environment creation
env = make_env("two_segments_v1", random=1, time_limit=time_limit,
               control_timestep=control_timestep)

# Model Loading
model = load("../docs/model_for_two_segments_v1.pkl")
source = collect_reference_source(
    env, seeds=[20], constant_after=True, constant_after_T=3.0)

# Training
p_gain = 0.01
i_gain = 0.0

options = create_pi_controller(p_gain, i_gain, delta_t=control_timestep)
controller = LinearController(options)

training_options = TrainingOptionsController(
    optax.adam(3e-3), 0.0, 500, 1, models=[model]
)

controller, losses = train_controller(controller, source, training_options)

#env_w_source = AddReferenceObservationWrapper(env, source)
#replay_sample_environment = collect_exhaust_source(
#     env=env_w_source, source=source, controller=controller)
#
#replay_sample_model = collect_exhaust_source(
#     env=env_w_source, source=source, controller=controller, model=model)

real_sample, model_sample  = collect_combined(env, source, controller=controller, model=model)

plt.plot(real_sample.obs["obs"]
         ["xpos_of_segment_end"][0], label="observation in environemnt")
plt.plot(real_sample.obs["ref"]
         ["xpos_of_segment_end"][0], label="reference in environemnt")
plt.plot(model_sample.obs["obs"]
         ["xpos_of_segment_end"][0], label="observation in model")
plt.plot(model_sample.obs["ref"]
         ["xpos_of_segment_end"][0], label="reference in model")



plt.legend()
plt.show()

#env.reset()
#replay_sample2 = collect_exhaust_source(
#     env=env_w_source, source=source, controller=controller)
#
#plt.plot(replay_sample2.obs["obs"]
#         ["xpos_of_segment_end"][0], label="observation no model")
#plt.plot(replay_sample2.obs["ref"]
#         ["xpos_of_segment_end"][0], label="reference no model")
#
#plt.legend()
#plt.show()
