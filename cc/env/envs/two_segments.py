from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Union

import numpy as np
from dm_control import mujoco
from dm_control.rl import control

from cc.env.make_env import EnvConfig

from ...utils.sample_from_spec import _spec_from_observation
from .common import ASSETS, read_model


class Color(Enum):
    SELF = "self"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    CYAN = "cyan"
    MAGENTA = "magenta"
    WHITE = "white"
    GRAY = "gray"
    BROWN = "brown"
    ORANGE = "orange"
    PINK = "pink"
    PURPLE = "purple"
    LIME = "lime"
    TURQUOISE = "turquoise"
    GOLD = "gold"
    MATPLOTLIB_GREEN = "matplotlib_green"
    MATPLOTLIB_BLUE = "matplotlib_blue"
    MATPLOTLIB_SALMON = "matplotlib_salmon"
    MATPLOTLIB_LIGHTBLUE = "matplotlib_lightblue"

@dataclass
class JointParams:
    damping: float = 0
    springref: float = 0
    stiffness: float = 0


@dataclass
class CartParams:
    name: str
    slider_joint_params: JointParams
    hinge_joint_params: JointParams
    material: Color = Color.SELF

@dataclass
class Marker:
    pos: float = 0
    material: Color = Color.SELF


def generate_body(
    name: str,
    slider_joint_params: JointParams,
    hinge_joint_params: JointParams,
    material: Color,
) -> bytes:
    """
    Generates a single movable body object, consisting of two poles connected by hinges.
    Joints of both the slider and the hinges can be parameterized.
    """
    return rf"""
    <body name="{name}_cart" pos="0 0 2">
      <joint name="{name}_slider" type="slide" limited="true" axis="1 0 0" range="-999.8 999.8" damping="{slider_joint_params.damping}" springref="{slider_joint_params.springref}" stiffness="{slider_joint_params.stiffness}"/>
      <geom name="{name}_cart" type="box" size="0.1 0.15 0.05" material="{material.value}"  mass="1"/>
      <body name="{name}_pole_1" childclass="pole" euler="0 180 0" pos="0 0 -0.1">
        <joint name="{name}_hinge_1" axis="0 1 0" damping="{hinge_joint_params.damping}" springref="{hinge_joint_params.springref}" stiffness="{hinge_joint_params.stiffness}"/>
        <geom name="{name}_pole_1" material="{material.value}"/>
        <body name="{name}_pole_2" childclass="pole" pos="0 0 1.1">
          <joint name="{name}_hinge_2" axis="0 1 0" damping="{hinge_joint_params.damping}" springref="{hinge_joint_params.springref}" stiffness="{hinge_joint_params.stiffness}"/>
          <geom name="{name}_pole_2" material="{material.value}"/>
          <body name="{name}_segment_end" pos="0 0 1.0"/>
        </body>
      </body>
    </body>
    """.encode()


def generate_motor(name: str) -> bytes:
    """
    Generates a single actuator used for sending inputs to a previously created body.
    """
    return rf"""
        <motor name="{name}_slide" joint="{name}_slider" gear="5" ctrllimited="false"/>
    """.encode()


def generate_camera(name: str) -> bytes:
    """
    Generates a camera which is fixed on the passed object (typically a cart).
    """
    return rf"""
        <camera name="{name}_lookatchain" mode="targetbody" target="{name}_cart" pos="0 -6 1"/>
    """.encode()


def generate_marker(
    name: str,
    pos: float,
    material: Color,
) -> bytes:
    """
    Generates a marker.
    """
    return rf"""
        <geom name="x-marker-generated{name}" priority="1" type="box" pos="{pos} 0 3" size="0.1 0.2 0.2" material="{material.value}" />
    """.encode()


class SegmentPhysics(mujoco.Physics):
    obs_cart_names: List[str]

    def set_obs_cart_names(self, cart_names: List[str]):
        self.obs_cart_names = cart_names

    def xpos_of_segment_end(self):
        return np.asarray(
            [
                self.named.data.xpos[f"{cart_name}_segment_end", "x"]
                for cart_name in self.obs_cart_names
            ]
        )

    def set_torque_of_cart(self, u):
        u = np.arctan(u)
        self.set_control(u)


def _load_physics(
    cart_params: Union[List[CartParams], CartParams],
    marker_params: Optional[List[Marker]],
) -> SegmentPhysics:
    """
    Creates a mujoco physics object using the provided cart parameters.
    """

    # if cart_params not list make it list
    if not isinstance(cart_params, list):
        cart_params = [cart_params]

    xml_path = "two_segments.xml"
    xml_content = read_model(xml_path)

    assert len(cart_params) >= 1, "At least one cart is required"

    bodies = b""
    motors = b""
    cameras = b""
    markers = b""

    for cart_param in cart_params:
        bodies += generate_body(
            cart_param.name,
            cart_param.slider_joint_params,
            cart_param.hinge_joint_params,
            cart_param.material,
        )
        motors += generate_motor(cart_param.name)
        cameras += generate_camera(cart_param.name)

    if marker_params:
        for i, marker_param in enumerate(marker_params):
            markers += generate_marker(
                f"{i}",
                marker_param.pos,
                marker_param.material,
            )

    # Insert bodies into template
    xml_content = xml_content.replace(b"<!-- Bodies -->", bodies)
    # Insert motors into template
    xml_content = xml_content.replace(b"<!-- Motors -->", motors)
    # Insert cameras into template
    xml_content = xml_content.replace(b"<!-- Cameras -->", cameras)
    # Insert markers into template
    xml_content = xml_content.replace(b"<!-- Markers -->", markers)

    seg_phy = SegmentPhysics.from_xml_string(xml_content, assets=ASSETS)
    seg_phy.set_obs_cart_names([cart_param.name for cart_param in cart_params])
    return seg_phy


def load_physics(
    cart_params: Union[List[CartParams], CartParams],
    marker_params: Optional[List[Marker]],
) -> Callable[[], mujoco.Physics]:
    def load_physics_helper():
        return _load_physics(cart_params, marker_params)

    return load_physics_helper


def generate_env_config(
    cart_params: Union[List[CartParams], CartParams],
    marker_params: Optional[List[Marker]] = None,
):
    return EnvConfig(
        load_physics=load_physics(cart_params, marker_params),
        task=SegmentTask,
    )


def generate_duplicate_env_config(
    cart_params: CartParams,
    num: int,
    materials: Optional[List[Color]] = None,
    marker_params: Optional[List[Marker]] = None,
):
    if materials is None:
        materials = [e for e in Color]
        materials = [materials[i % len(materials)] for i in range(num)]
    else:
        assert len(materials) == num, "Number of materials must match number of carts"

    return EnvConfig(
        load_physics=load_physics(
            [
                CartParams(
                    **{
                        **cart_params.__dict__,
                        "name": cart_params.name + f"_{i}",
                        "material": material,
                    }
                )
                for i, material in enumerate(materials)
            ],
            marker_params,
        ),
        task=SegmentTask,
    )


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
