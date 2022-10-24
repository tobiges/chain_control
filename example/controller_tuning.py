import matplotlib.pyplot as plt
import optax

from cc.collect import collect_reference_source, collect_exhaust_source, collect
from cc import save, load
from cc.controller import create_pi_controller, LinearController, LinearControllerOptions
from cc.env.wrappers import AddReferenceObservationWrapper, RecordVideoWrapper
from cc.env import make_env
from cc.train import TrainingOptionsController, train_controller
from cc.utils.utils import generate_ts

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


# Testing
env_w_source = AddReferenceObservationWrapper(env, source)

replay_sample = collect_exhaust_source(
    env=env_w_source, source=source, controller=controller, model=model)

# OrderedDictionary ["obs" + "ref"] -> OrderedDictionary ["xpos of segment end"] -> Array shape [1, 1001, 1]

print(replay_sample.obs.keys())

print(replay_sample.obs["obs"].keys())
print(type(replay_sample.obs["obs"]))

print(replay_sample.obs["ref"].keys())
print(type(replay_sample.obs["ref"]))
print(type(replay_sample.obs["ref"]["xpos_of_segment_end"]))
print(replay_sample.obs["ref"]["xpos_of_segment_end"].shape)

print("shapes")
print(replay_sample.action.shape)
print(replay_sample.rew.shape)
print(replay_sample.done.shape)

print(len(replay_sample.obs["obs"]["xpos_of_segment_end"]))


# First
plt.plot(replay_sample.obs["obs"]
         ["xpos_of_segment_end"][0], label="observation 1")
plt.plot(replay_sample.obs["ref"]
         ["xpos_of_segment_end"][0], label="reference 1")

plt.plot(replay_sample.obs["obs"]
         ["xpos_of_segment_end"][1], label="observation 2")
plt.plot(replay_sample.obs["ref"]
         ["xpos_of_segment_end"][1], label="reference 2")

plt.legend()
plt.show()
