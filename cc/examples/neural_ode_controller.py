import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand

from ..core import AbstractController, PyTree
from ..utils import (
    ArraySpecs,
    batch_concat,
    make_postprocess_fn,
    sample_from_tree_of_specs,
)
from .nn_lib.integrate import integrate
from .nn_lib.mlp_network import mlp_network


def make_neural_ode_controller(
    input_specs: ArraySpecs,
    output_specs: ArraySpecs,
    control_timestep: float,
    state_dim: int,
    key,
    f_integrate_method: str = "RK4",
    f_use_bias: bool = True,
    f_time_invariant: bool = True,
    f_use_dropout: bool = False,
    f_dropout_rate: float = 0.4,
    f_width_size: int = 10,
    f_depth: int = 2,
    f_activation=jax.nn.relu,
    f_final_activation=lambda x: x,
    g_use_bias: bool = True,
    g_time_invariant: bool = True,
    g_use_dropout: bool = False,
    g_dropout_rate: float = 0.4,
    g_width_size: int = 10,
    g_depth: int = 0,
    g_activation=jax.nn.relu,
    g_final_activation=lambda x: x,
):
    toy_input = sample_from_tree_of_specs(input_specs)
    toy_output = sample_from_tree_of_specs(output_specs)
    input_dim = batch_concat(toy_input, 0).size
    output_dim = batch_concat(toy_output, 0).size

    f_input_dim = state_dim + input_dim
    if not f_time_invariant:
        f_input_dim += 1

    f_output_dim = state_dim
    g_input_dim = state_dim
    if not g_time_invariant:
        g_input_dim += 1
    g_output_dim = output_dim

    f_key, g_key = jrand.split(key, 2)

    has_time_state = (not f_time_invariant) or (not g_time_invariant)
    if has_time_state:
        # second state is time
        init_state = (jnp.zeros((state_dim,)), jnp.array(0.0))
    else:
        init_state = jnp.zeros((state_dim,))

    postprocess_fn = make_postprocess_fn(toy_output=toy_output)

    f_init = mlp_network(
        f_input_dim,
        f_output_dim,
        f_width_size,
        f_depth,
        f_activation,
        f_final_activation,
        f_use_bias,
        f_use_dropout,
        f_dropout_rate,
        f_key,
    )

    g_init = mlp_network(
        g_input_dim,
        g_output_dim,
        g_width_size,
        g_depth,
        g_activation,
        g_final_activation,
        g_use_bias,
        g_use_dropout,
        g_dropout_rate,
        g_key,
    )

    class NeuralOdeController(AbstractController):
        f: eqx.Module
        g: eqx.Module
        state: PyTree[jnp.ndarray]
        init_state: PyTree[jnp.ndarray]

        def reset(self):
            return NeuralOdeController(self.f, self.g, self.init_state, self.init_state)

        def step(self, u):  # u has shape identical to (`toy_input`, PRNGKey)
            u, key = u

            if has_time_state:
                (x, t) = self.state  # pytype: disable=attribute-error
            else:
                x = self.state
                t = jnp.array(0.0)

            if f_use_dropout:
                key, consume = jrand.split(key)

            if not f_time_invariant:
                rhs = lambda t, x: self.f(batch_concat((x, t, u), 0), key=consume)
            else:
                rhs = lambda t, x: self.f(batch_concat((x, u), 0), key=consume)

            x_next = integrate(rhs, x, t, control_timestep, f_integrate_method)

            if g_use_dropout:
                key, consume = jrand.split(key)

            if not g_time_invariant:
                y_next = self.g(batch_concat((x_next, t), 0), key=consume)
            else:
                y_next = self.g(batch_concat((x_next,), 0), key=consume)

            y_next = postprocess_fn(y_next)

            if has_time_state:
                state_next = (x_next, t + control_timestep)
            else:
                state_next = x_next

            return NeuralOdeController(self.f, self.g, state_next, self.init_state), (
                y_next,
                key,
            )  # y_next has shape identical to (`toy_output`, PRNGKey)

        def grad_filter_spec(self) -> PyTree[bool]:
            filter_spec = super().grad_filter_spec()
            filter_spec = eqx.tree_at(
                lambda model: (model.state, model.init_state),
                filter_spec,
                (False, False),  # both `state` and `init_state` are not optimized
            )
            return filter_spec

    return NeuralOdeController(f_init, g_init, init_state, init_state)
