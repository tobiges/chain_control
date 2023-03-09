#!/usr/bin/env python
# coding: utf-8

# # Defining an Environment
# 
# Here we will re-define the environment `two_segments_v1`
# 
# Every environment consists of
# - a Mujoco Specification File (end in .xml)
# - a Python File 
# 
# The .xml file is used be Mujoco to simulate the system. 
# 
# The Python file is used to define how to interface/interact with this Mujoco Simulation, e.g. what we are allowed to alter at every timestep (usually the control).
# 

# In[1]:


from dm_control.rl import control
from dm_control import mujoco
from collections import OrderedDict
import numpy as np 
from cc.env.envs.common import read_model, ASSETS 
from cc.env.sample_from_spec import _spec_from_observation
from cc.env import make_env


# ----
# Let's take a closer look at the content of the .py-file.
# 
# It contains to objects
# - a `mujoco.Physics` object
# - a `control.Task`

# The `mujoco.Physics` object gives us a way to interact with the Mujoco simulation from Python.

# In[2]:


class SegmentPhysics(mujoco.Physics):

    def xpos_of_segment_end(self):
        return self.named.data.xpos["segment_end", "x"]

    def set_torque_of_cart(self, u):
        u = np.arctan(u)
        self.set_control(u)


def load_physics():
    xml_path = "two_segments_v1.xml"
    return SegmentPhysics.from_xml_string(read_model(xml_path), assets=ASSETS)

load_physics()


# The `control.Task` precisely defines when we can and and when we will interact with the `mujoco.Physics`-object.

# In[3]:


class SegmentTask(control.Task):

    def __init__(self, random: int = 1):
        # seed is unused 
        del random 
        super().__init__()
        
    def initialize_episode(self, physics):
        pass 

    def before_step(self, action, physics: SegmentPhysics):
        physics.set_torque_of_cart(action)

    def after_step(self, physics):
        pass 

    def action_spec(self, physics):
        return mujoco.action_spec(physics)

    def get_observation(self, physics) -> OrderedDict:
        obs = OrderedDict()
        obs["xpos_of_segment_end"] = np.atleast_1d(physics.xpos_of_segment_end())
        return obs 

    def get_reward(self, physics):
        return np.array(0.0)

    def observation_spec(self, physics):
        return _spec_from_observation(self.get_observation(physics))

SegmentTask()


# With these two components we can register a new Environment at `cc.env.register`
# 
# Here, this has already been done and we can simply load it using its string-identifier.

# In[4]:


env = make_env("two_segments_v1", random=1)


# In[5]:


action=np.array([0.2])
env.step(action)


# In[6]:


env.step(action)


# Without going into any details: Let's just take a look at some randomly acting controller in this environment.
# 
# Press the backspace key to reset the environment.

# In[7]:


from cc.collect.actor import RandomActor
from cc.visual.policy_for_viewer import policy_for_viewer
from dm_control import viewer

actor = RandomActor(env.action_spec(), reset_key=True)

viewer.launch(env, policy_for_viewer(actor))


# In[ ]:





# In[ ]:




