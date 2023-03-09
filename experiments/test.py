import numpy as np 
from cc.env import make_env, ReferenceEnvironmentConfig
from cc.env.collect import RandomActor
from cc.utils.visual.viewer import launch_viewer


ref_config = ReferenceEnvironmentConfig(ref_start = -3.0, 
                                        ref_actions = np.random.uniform(-1.0, 1.0, 1000))

env = make_env("two_segments_v1", random=1, reference_config = ref_config)

actor = RandomActor(env.action_spec(), reset_key=True)

launch_viewer(env, actor)