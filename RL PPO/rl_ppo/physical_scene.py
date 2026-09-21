"""Build the three-finger physical catch-and-throw UMArm scene.

The fitted arm remains contact-free. This separate scene adds only a free
ball and collision-enabled gripper geoms. A disabled weld is activated after
a real contact and a kinematically compatible grasp, modelling the instant at
which the hardware fingers finish closing around the ball.
"""
from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

import numpy as np

from digital_twin.mjcf_generator import generate_xml


@dataclass(frozen=True)
class CatcherSpec:
    """Versioned assumptions for the ball and open three-finger gripper."""

    ball_mass_kg: float = 0.025
    ball_radius_m: float = 0.018
    grip_radius_m: float = 0.031
    finger_length_m: float = 0.065
    finger_radius_m: float = 0.004
    palm_z_m: float = 0.022
    contact_time_constant_s: float = 0.010
    contact_damping_ratio: float = 1.0
    friction: float = 0.85
    grasp_radial_limit_m: float = 0.017
    grasp_axial_bounds_m: tuple[float, float] = (-0.028, 0.046)
    grasp_relative_speed_mps: float = 0.45
    grasp_alignment_cos: float = 0.75
    grasp_dwell_s: float = 0.004
    close_energy_j: float = 0.12
    open_energy_j: float = 0.08
    # Local +z follows the nominal incoming velocity [0, +1, -0.75]. The open
    # end (local -z) therefore faces the launcher on the negative world-y side.
    mount_quat_wxyz: tuple[float, float, float, float] = (
        0.4472135955, -0.8944271910, 0.0, 0.0
    )

    def validate(self) -> None:
        if self.ball_mass_kg <= 0 or self.ball_radius_m <= 0:
            raise ValueError("ball mass and radius must be positive")
        if self.grip_radius_m <= self.ball_radius_m or self.finger_length_m <= 0:
            raise ValueError("gripper must be larger than the ball")
        if self.finger_radius_m <= 0 or self.grasp_radial_limit_m <= 0:
            raise ValueError("finger radius and grasp region must be positive")
        if self.grasp_axial_bounds_m[0] >= self.grasp_axial_bounds_m[1]:
            raise ValueError("invalid grasp axial bounds")
        if (not 0 <= self.grasp_alignment_cos <= 1
                or self.grasp_relative_speed_mps <= 0 or self.grasp_dwell_s <= 0):
            raise ValueError("invalid grasp kinematic thresholds")
        if len(self.mount_quat_wxyz) != 4 or not np.isclose(
                np.linalg.norm(self.mount_quat_wxyz), 1.0, atol=1e-6):
            raise ValueError("gripper mount quaternion must be a unit wxyz quaternion")


BALL_BODY = "catch_ball"
BALL_JOINT = "catch_ball_free"
BALL_GEOM = "catch_ball_geom"
CUP_BODY = "catcher_gripper"  # Backward-compatible public name.
CUP_SITE = "catcher_center"
CUP_AXIS_SITE = "catcher_axis"
CUP_BACK_GEOM = "catcher_palm"
CUP_WALL_PREFIX = "catcher_finger_"
GRASP_WELD = "ball_grasp_weld"
FINGER_COUNT = 3


def catcher_geom_names(spec: CatcherSpec) -> tuple[str, ...]:
    del spec
    return (CUP_BACK_GEOM,) + tuple(f"{CUP_WALL_PREFIX}{index}" for index in range(FINGER_COUNT))


def _element(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    return ET.SubElement(parent, tag, {name: str(value) for name, value in attributes.items()})


def build_catch_xml(*, catcher: CatcherSpec | None = None, **fitted_tunables: object) -> str:
    """Return the fitted arm with an open gripper, free ball, and disabled weld."""
    spec = catcher or CatcherSpec()
    spec.validate()
    root = ET.fromstring(generate_xml(**fitted_tunables))
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("generated UMArm XML has no worldbody")
    plate = root.find(".//body[@name='canarm_seg3_plate2']")
    if plate is None:
        raise RuntimeError("generated UMArm XML has no final plate body")

    # The gripper is rigidly attached to the final plate. Mechanical softness
    # comes from UMArm compliance and velocity matching, not hidden mount DOFs.
    mount_quat = " ".join(f"{value:.10g}" for value in spec.mount_quat_wxyz)
    gripper = _element(
        plate, "body", name=CUP_BODY, pos="0 0 0", quat=mount_quat
    )
    _element(gripper, "site", name=CUP_SITE, pos="0 0 0", size="0.004", rgba="0.1 0.9 0.2 1")
    _element(gripper, "site", name=CUP_AXIS_SITE, pos="0 0 -0.025", size="0.002", rgba="0.1 0.7 1 1")

    contact = dict(
        contype="2", conaffinity="1", friction=f"{spec.friction:.6g} 0.01 0.001",
        solref=f"{spec.contact_time_constant_s:.6g} {spec.contact_damping_ratio:.6g}",
        solimp="0.90 0.95 0.001", rgba="0.15 0.75 0.35 1",
    )
    _element(
        gripper, "geom", name=CUP_BACK_GEOM, type="cylinder",
        pos=f"0 0 {spec.palm_z_m:.6g}", size=f"{spec.grip_radius_m + .008:.6g} .004",
        mass="0.014", **contact,
    )
    z_front = spec.palm_z_m - spec.finger_length_m
    for index in range(FINGER_COUNT):
        angle = 2 * np.pi * index / FINGER_COUNT
        x, y = spec.grip_radius_m * np.cos(angle), spec.grip_radius_m * np.sin(angle)
        _element(
            gripper, "geom", name=f"{CUP_WALL_PREFIX}{index}", type="capsule",
            fromto=f"{x:.6g} {y:.6g} {z_front:.6g} {x:.6g} {y:.6g} {spec.palm_z_m:.6g}",
            size=f"{spec.finger_radius_m:.6g}", mass="0.002", **contact,
        )

    ball = _element(world, "body", name=BALL_BODY, pos="0 0 0")
    _element(ball, "freejoint", name=BALL_JOINT)
    _element(
        ball, "geom", name=BALL_GEOM, type="sphere", size=f"{spec.ball_radius_m:.6g}",
        mass=f"{spec.ball_mass_kg:.6g}", contype="1", conaffinity="2",
        friction=f"{spec.friction:.6g} 0.01 0.001",
        solref=f"{spec.contact_time_constant_s:.6g} {spec.contact_damping_ratio:.6g}",
        solimp="0.90 0.95 0.001", rgba="0.95 0.20 0.12 1",
    )

    equality = root.find("equality")
    if equality is None:
        equality = _element(root, "equality")
    _element(
        equality, "weld", name=GRASP_WELD, body1=CUP_BODY, body2=BALL_BODY,
        active="false", solref="0.004 1", torquescale="0.02",
    )
    return ET.tostring(root, encoding="unicode")
