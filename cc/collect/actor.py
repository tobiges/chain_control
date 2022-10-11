import dm_env 
from acme import core 
from ..buffer import AbstractAdder
from ..env.sample_from_spec import sample_action_from_action_spec
from ..utils import to_jax, to_numpy
from ..types import *
import jax.random as jrand 
from ..abstract import AbstractController


class PolicyActor(core.Actor):
    def __init__(self, 
        action_spec,
        policy: Optional[AbstractController] = None,
        key: PRNGKey = jrand.PRNGKey(1,), 
        adder: Optional[AbstractAdder] = None,
        reset_key = False,
    ):

        self.action_spec = action_spec
        self._adder = adder 
        self._policy = policy
        self.reset_key=reset_key 
        self._initial_key = self._key = key
        self.reset()

    def observe_first(self, timestep: dm_env.TimeStep):
        self.reset()
        if self._adder:
            self._adder.add_first(timestep)

    def update_policy(self, new_policy: AbstractController):
        if new_policy:
            self._policy = new_policy.reset()

    def reset(self):
            
        if self.reset_key:
            self._key = self._initial_key
        else:
            pass 

        if self._policy:
            self._policy = self._policy.reset()

        self.count = 0
        self._last_extras = None 
    
    def observe(self, action, next_timestep):
        if self._adder:
            if self._last_extras:
                self._adder.add(action, next_timestep=next_timestep,extras=self._last_extras)
            else: 
                self._adder.add(action, next_timestep=next_timestep)

    def update(self, wait: bool = False):
        pass 

    def select_action(self, obs: Observation) -> Action:
        self.count += 1 
        self._key, consume = jrand.split(self._key)
        action = self.query_policy(to_jax(obs), consume)
        return to_numpy(action)
        
    def query_policy(self, obs: Observation, key: PRNGKey) -> Action:
        self._policy, action = eqx.filter_jit(self._policy)(obs)
        return action 


class RandomActor(PolicyActor):
    def query_policy(self, obs: Observation, key: PRNGKey) -> Action:
        return sample_action_from_action_spec(key, self.action_spec)
