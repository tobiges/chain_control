

import matplotlib.pyplot as plt
import optax

from cc.collect import *
from cc import save, load
from cc.controller import create_pi_controller, LinearController, LinearControllerOptions
from cc.env.wrappers import RecordVideoWrapper
from cc.env import make_env
from cc.train import TrainingOptionsController, train_controller
from jax import random
import pprint
import numpy
time_limit = 10.0
control_timestep = 0.01

# Environment creation
env = make_env("two_segments_v1", random=1, time_limit=time_limit,
               control_timestep=control_timestep)

# Model Loading
model = load("../docs/model_for_two_segments_v1.pkl")
source = collect_reference_source(
    env, seeds=[20], constant_after=True, constant_after_T=3.0)

def get_controller(p_gain = 0.01, i_gain = 0.0):
    # Training
    options = create_pi_controller(p_gain, i_gain, delta_t=control_timestep)
    controller = LinearController(options)

    training_options = TrainingOptionsController(
       optax.adam(3e-3), 0.0, 500, 1, models=[model]
    )

    controller, losses = train_controller(controller, source, training_options)
    return controller, f"p_gain: {p_gain} i_gain: {i_gain}"
controllers = []
controllers.append(get_controller())

pp = pprint.PrettyPrinter(indent=4)
numpy.set_printoptions(threshold=5)
for p in range(0, 100, 20):
    for i in range(0, 100, 20):
        controllers.append(get_controller(p / 100, i / 100))

time_limit, control_timestep, ts = extract_timelimit_timestep_from_env(env)

#real_env_w_source = ModelBasedEnv(env, model, time_limit=time_limit, control_timestep=control_timestep) # <--- collect 
env_with_ref = AddReferenceObservationWrapper(env, source)
result = collect_multiple(env, source, [c[0] for c in controllers])

result_ranked = sorted(enumerate(result.obs["obs"]["xpos_of_segment_end"]), key=lambda x: result.extras["aggr_rew"][x[0]])

plt.plot(result.obs["ref"]
         ["xpos_of_segment_end"][0], label="reference")

for rank, (index, value) in enumerate(result_ranked[:5]):
    plt.plot(value, label=f"observation {controllers[index][1]} rank: {rank}")

for rank, (index, value) in enumerate(result_ranked[-5:]):
    plt.plot(value, label=f"observation {controllers[index][1]} rank: {rank}")

plt.legend()
plt.show()
