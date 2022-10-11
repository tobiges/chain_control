from ..rhs.common_controller_model import (
    rhs_state_LinearControllerModel,
    rhs_state_NonlinearControllerModel,
    LinearControllerModelOptions,
    NonlinearControllerModelOptions
)
from ..types import *
from ..abstract import AbstractModel, Y
from ..rhs.wrapped_rhs import WrappedRHS


LinearModelOptions = LinearControllerModelOptions
NonlinearModelOptions = NonlinearControllerModelOptions


def default_postprocess_y(y):
    d = OrderedDict()
    d["xpos_of_segment_end"] = y 
    return d 


class Model(WrappedRHS, AbstractModel):

    def y0(self) -> Observation:
        """Initial measurement
        """
        _, y0 = eqx.filter_jit(self)(jnp.zeros((self.input_size)))
        return Observation(y0) 

    @staticmethod
    def postprocess_y(y: Y) -> PyTree:
        return default_postprocess_y(y)


class LinearModel(Model):
    def __init__(self, options: LinearModelOptions):
        rhs, state = rhs_state_LinearControllerModel(options)
        self.rhs = rhs 
        self.state = state 
        self.input_size = options.input_size
        self.output_size = options.output_size        


class NonlinearModel(Model):
    def __init__(self, options: NonlinearModelOptions):
        rhs, state = rhs_state_NonlinearControllerModel(options)
        self.rhs = rhs 
        self.state = state 
        self.input_size = options.input_size
        self.output_size = options.output_size        

