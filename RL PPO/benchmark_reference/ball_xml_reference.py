from __future__ import annotations

import xml.etree.ElementTree as ET
import math
from pathlib import Path

from .mckibben import make_force_motor_xml
from .xml import _indent_xml, make_motorized_xml


GLOVE_WELD_SOFT_SOLREF: tuple[float, float] = (0.004, 1.0)
GLOVE_WELD_SOFT_SOLIMP: tuple[float, float, float] = (0.95, 0.99, 0.001)
GLOVE_WELD_STIFF_SOLIMP: tuple[float, float, float] = (0.99, 0.999, 0.0005)
GLOVE_WELD_STIFF_TIMESTEP_MULTIPLIER = 2.0


def _xml_timestep_s(src_xml: Path) -> float:
    root = ET.parse(src_xml).getroot()
    option = root.find("option")
    if option is None or "timestep" not in option.attrib:
        raise ValueError(f"{src_xml} has no <option timestep=...> for stiff glove weld preset")
    timestep = float(option.attrib["timestep"])
    if timestep <= 0.0:
        raise ValueError(f"{src_xml} has non-positive timestep {timestep}")
    return timestep


def glove_weld_stiff_solref_for_timestep(timestep_s: float) -> tuple[float, float]:
    return (GLOVE_WELD_STIFF_TIMESTEP_MULTIPLIER * float(timestep_s), 1.0)


def add_base_yaw_to_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    joint_name: str = "base_yaw",
    body_name: str = "base_link",
    range_deg: tuple[float, float] = (-180.0, 180.0),
    remove_world_axes: bool = False,
    damping: float = 0.5,
) -> None:
    """Add a model-variant turntable hinge at the arm mount.

    The hinge is inserted on ``base_link`` so the mount, tendon anchor sites,
    and all universal-joint bodies yaw together about the world vertical axis.
    Existing fixed-base helpers do not call this function.
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML has no <worldbody> element.")

    if remove_world_axes:
        for child in list(worldbody):
            if child.attrib.get("name") == "world_axes":
                worldbody.remove(child)

    base_body = root.find(f".//body[@name='{body_name}']")
    if base_body is None:
        raise ValueError(f"XML has no body named {body_name!r}.")
    if base_body.find(f"joint[@name='{joint_name}']") is None:
        lo, hi = (float(v) for v in range_deg)
        if not lo < hi:
            raise ValueError("range_deg must be increasing")
        base_body.insert(
            0,
            ET.Element(
                "joint",
                {
                    "name": joint_name,
                    "type": "hinge",
                    "axis": "0 0 1",
                    "range": f"{lo:.6f} {hi:.6f}",
                    "limited": "true",
                    "damping": f"{float(damping):.6f}",
                    "frictionloss": "0.000000",
                    "armature": "0.000500",
                },
            ),
        )

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def add_paddle_to_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    body_name: str = "ee_sphere_body",
    paddle_name: str = "ee_paddle",
    radius_m: float = 0.055,
    half_thickness_m: float = 0.004,
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Attach a simple circular paddle geom to the end-effector body."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    target_body = root.find(f".//body[@name='{body_name}']")
    if target_body is None:
        raise ValueError(f"XML has no body named {body_name!r}.")

    for child in list(target_body):
        if child.attrib.get("name") in {f"{paddle_name}_geom", f"{paddle_name}_site"}:
            target_body.remove(child)

    x, y, z = (float(v) for v in offset_m)
    ex, ey, ez = (float(v) for v in euler_deg)
    target_body.append(
        ET.Element(
            "geom",
            {
                "name": f"{paddle_name}_geom",
                "type": "cylinder",
                "pos": f"{x:.6f} {y:.6f} {z:.6f}",
                "euler": f"{ex:.6f} {ey:.6f} {ez:.6f}",
                "size": f"{float(radius_m):.6f} {float(half_thickness_m):.6f}",
                "rgba": "0.10 0.25 0.95 0.65",
                "mass": "0.020000",
                "condim": "3",
                "friction": "0.8 0.02 0.001",
                "solref": "0.006 1",
                "solimp": "0.9 0.95 0.001",
            },
        )
    )
    target_body.append(
        ET.Element(
            "site",
            {
                "name": f"{paddle_name}_site",
                "type": "cylinder",
                "pos": f"{x:.6f} {y:.6f} {z:.6f}",
                "euler": f"{ex:.6f} {ey:.6f} {ez:.6f}",
                "size": f"{float(radius_m):.6f} {float(half_thickness_m) * 1.2:.6f}",
                "rgba": "0.10 0.25 0.95 0.25",
                "group": "2",
            },
        )
    )

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def add_catcher_to_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    body_name: str = "ee_sphere_body",
    catcher_name: str = "ee_catcher",
    catcher_variant: str = "wide_posts",
    half_width_m: float = 0.045,
    floor_half_thickness_m: float = 0.004,
    wall_half_thickness_m: float = 0.004,
    wall_height_m: float = 0.044,
    floor_offset_m: tuple[float, float, float] = (0.0, 0.0, -0.022),
    rim_posts: int = 12,
    tight_inner_radius_m: float = 0.022,
    tight_wall_element_radius_m: float = 0.003,
    tight_wall_elements: int = 36,
    catcher_mount_offset_m: tuple[float, float, float] | None = None,
    catcher_mount_euler_deg: tuple[float, float, float] | None = None,
    catcher_mount_quat: tuple[float, float, float, float] | None = None,
    catcher_mount_stub_radius_m: float = 0.0035,
    catcher_solref: tuple[float, float] = (0.012, 1.0),
    catcher_solimp: tuple[float, float, float] = (0.92, 0.98, 0.001),
    wrist_roll_joint: bool = False,
    wrist_roll_range_deg: tuple[float, float] = (-180.0, 180.0),
    rgba: str = "0.10 0.65 0.35 0.70",
) -> None:
    """Attach a shallow convex-geom catcher to the end-effector body.

    ``catcher_variant="wide_posts"`` preserves the original broad catcher.
    ``catcher_variant="tight_cup"`` builds a precision cup with a dense ring
    of vertical capsules. ``catcher_variant="tight_cup_wrist_roll"`` inserts
    one unactuated axial hinge between ``ee_sphere_body`` and the cup mount for
    model-only previews. The tight-cup variants may be mounted on a rotated,
    offset child body; the catcher site remains at the cup center, and the cup
    opening faces the mount body's local +z. This keeps the strike paddle XML
    path untouched and avoids unsupported MJX collision pairs.
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    target_body = root.find(f".//body[@name='{body_name}']")
    if target_body is None:
        raise ValueError(f"XML has no body named {body_name!r}.")

    prefixes = {
        f"{catcher_name}_floor",
        f"{catcher_name}_wall",
        f"{catcher_name}_site",
        f"{catcher_name}_mount",
    }
    mount_body_name = f"{catcher_name}_mount_body"
    wrist_roll_body_name = f"{catcher_name}_wrist_roll_body"
    for child in list(target_body):
        name = child.attrib.get("name", "")
        if name in {mount_body_name, wrist_roll_body_name} or any(name.startswith(prefix) for prefix in prefixes):
            target_body.remove(child)

    variant = str(catcher_variant).strip().lower()
    if variant == "tight_cup_wrist_roll":
        wrist_roll_joint = True
        variant = "tight_cup"
    floor_hz = float(floor_half_thickness_m)
    wall_hz = 0.5 * float(wall_height_m)
    fx, fy, fz = (float(v) for v in floor_offset_m)
    geom_common = {
        "rgba": rgba,
        "condim": "3",
        "friction": "1.2 0.04 0.002",
        "solref": f"{float(catcher_solref[0]):.6f} {float(catcher_solref[1]):.6f}",
        "solimp": (
            f"{float(catcher_solimp[0]):.6f} "
            f"{float(catcher_solimp[1]):.6f} "
            f"{float(catcher_solimp[2]):.6f}"
        ),
    }

    if variant == "wide_posts":
        post_radius = float(wall_half_thickness_m) * 1.45
        wall_z = fz + floor_hz + wall_hz
        wall_center = float(half_width_m) + post_radius
        target_body.append(
            ET.Element(
                "geom",
                {
                    **geom_common,
                    "name": f"{catcher_name}_floor_geom",
                    "type": "cylinder",
                    "pos": f"{fx:.6f} {fy:.6f} {fz:.6f}",
                    "size": f"{float(half_width_m):.6f} {floor_hz:.6f}",
                    "mass": "0.012000",
                },
            )
        )
        for post_idx in range(max(4, int(rim_posts))):
            theta = 2.0 * 3.141592653589793 * float(post_idx) / float(max(4, int(rim_posts)))
            x = wall_center * math.cos(theta)
            y = wall_center * math.sin(theta)
            target_body.append(
                ET.Element(
                    "geom",
                    {
                        **geom_common,
                        "name": f"{catcher_name}_wall_{post_idx:02d}_geom",
                        "type": "cylinder",
                        "pos": f"{x:.6f} {y:.6f} {wall_z:.6f}",
                        "size": f"{post_radius:.6f} {wall_hz:.6f}",
                        "mass": "0.001500",
                    },
                )
            )
        site_size = float(half_width_m) * 0.18
    elif variant == "tight_cup":
        default_mount_offset = (0.033512592828394486, -0.043610848672299904, 0.0)
        default_mount_quat = (
            0.6695006574519818,
            0.6695006574519818,
            0.22752773385098365,
            0.22752773385098365,
        )
        mount_offset = (
            default_mount_offset
            if catcher_mount_offset_m is None
            else tuple(float(v) for v in catcher_mount_offset_m)
        )
        mount_quat = (
            default_mount_quat
            if catcher_mount_euler_deg is None and catcher_mount_quat is None
            else catcher_mount_quat
        )
        mount_euler = (
            tuple(float(v) for v in catcher_mount_euler_deg)
            if catcher_mount_euler_deg is not None
            else None
        )
        mx, my, mz = (float(v) for v in mount_offset)
        mount_attrs = {
            "name": mount_body_name,
            "pos": f"{mx:.6f} {my:.6f} {mz:.6f}",
        }
        if mount_quat is not None:
            qw, qx, qy, qz = (float(v) for v in mount_quat)
            mount_attrs["quat"] = f"{qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f}"
        else:
            mex, mey, mez = (float(v) for v in mount_euler)
            mount_attrs["euler"] = f"{mex:.6f} {mey:.6f} {mez:.6f}"
        parent_body = target_body
        if bool(wrist_roll_joint):
            lo, hi = (float(v) for v in wrist_roll_range_deg)
            if not lo < hi:
                raise ValueError("wrist_roll_range_deg must be increasing")
            parent_body = ET.Element("body", {"name": wrist_roll_body_name, "pos": "0.000000 0.000000 0.000000"})
            parent_body.append(
                ET.Element(
                    "joint",
                    {
                        "name": f"{catcher_name}_wrist_roll",
                        "type": "hinge",
                        "axis": "0 0 1",
                        "range": f"{lo:.6f} {hi:.6f}",
                        "limited": "true",
                        "damping": "0.020000",
                        "frictionloss": "0.000000",
                        "armature": "0.000100",
                    },
                )
            )
            target_body.append(parent_body)

        mount_body = ET.Element("body", mount_attrs)
        stub_len = float(math.sqrt(mx * mx + my * my + mz * mz))
        if stub_len > 1e-8 and float(catcher_mount_stub_radius_m) > 0.0:
            parent_body.append(
                ET.Element(
                    "geom",
                    {
                        "name": f"{catcher_name}_mount_stub_geom",
                        "type": "capsule",
                        "fromto": f"0.000000 0.000000 0.000000 {mx:.6f} {my:.6f} {mz:.6f}",
                        "size": f"{float(catcher_mount_stub_radius_m):.6f}",
                        "rgba": "0.08 0.38 0.24 0.72",
                        "mass": "0.002000",
                        "contype": "0",
                        "conaffinity": "0",
                    },
                )
            )
        parent_body.append(mount_body)
        cup_body = mount_body
        inner_radius = float(tight_inner_radius_m)
        element_radius = float(tight_wall_element_radius_m)
        wall_count = max(12, int(tight_wall_elements))
        if inner_radius <= 0.0 or element_radius <= 0.0:
            raise ValueError("tight_cup radii must be positive")
        if float(wall_height_m) <= 2.0 * element_radius:
            raise ValueError("tight_cup wall_height_m must exceed two wall element radii")
        floor_radius = inner_radius + 2.0 * element_radius + 0.001
        wall_center = inner_radius + element_radius
        floor_top_z = fz + floor_hz
        capsule_half_length = 0.5 * float(wall_height_m) - element_radius
        wall_z = floor_top_z + 0.5 * float(wall_height_m)
        cup_body.append(
            ET.Element(
                "geom",
                {
                    **geom_common,
                    "name": f"{catcher_name}_floor_tight_geom",
                    "type": "cylinder",
                    "pos": f"{fx:.6f} {fy:.6f} {fz:.6f}",
                    "size": f"{floor_radius:.6f} {floor_hz:.6f}",
                    "mass": "0.008000",
                },
            )
        )
        for wall_idx in range(wall_count):
            theta = 2.0 * 3.141592653589793 * float(wall_idx) / float(wall_count)
            x = wall_center * math.cos(theta)
            y = wall_center * math.sin(theta)
            cup_body.append(
                ET.Element(
                    "geom",
                    {
                        **geom_common,
                        "name": f"{catcher_name}_wall_tight_{wall_idx:02d}_geom",
                        "type": "capsule",
                        "pos": f"{x:.6f} {y:.6f} {wall_z:.6f}",
                        "size": f"{element_radius:.6f} {capsule_half_length:.6f}",
                        "mass": "0.000500",
                    },
                )
            )
        site_size = max(0.004, inner_radius * 0.25)
    else:
        raise ValueError(f"unknown catcher_variant {catcher_variant!r}")

    site_parent = target_body if variant == "wide_posts" else mount_body
    site_parent.append(
        ET.Element(
            "site",
            {
                "name": f"{catcher_name}_site",
                "type": "sphere",
                "pos": "0.000000 0.000000 0.000000",
                "size": f"{site_size:.6f}",
                "rgba": "0.10 0.65 0.35 0.35",
                "group": "2",
            },
        )
    )

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def add_baseball_glove_to_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    body_name: str = "ee_sphere_body",
    glove_name: str = "ee_baseball_glove",
    ball_name: str = "task_ball",
    ball_radius_m: float = 0.018,
    wrist_roll_range_deg: tuple[float, float] = (-180.0, 180.0),
    wrist_pitch_range_deg: tuple[float, float] = (-90.0, 90.0),
    pocket_inner_radius_m: float = 0.026,
    pocket_floor_radius_m: float = 0.028,
    pocket_floor_half_thickness_m: float = 0.003,
    pocket_depth_m: float = 0.014,
    rim_radius_m: float = 0.0035,
    rim_elements: int = 8,
    wrist_to_pocket_offset_m: tuple[float, float, float] = (0.0, -0.050, 0.0),
    glove_weld_solref: tuple[float, float] | None = None,
    glove_weld_solimp: tuple[float, float, float] | None = None,
    glove_weld_stiff: bool = False,
    pocket_collision: bool = True,
) -> None:
    """Add a two-axis wrist and baseball-glove end-effector model variant.

    ``pocket_collision=False`` keeps the glove visible (pocket floor, rim, fingers,
    thumb, heel) but disables their collision, so the ball is held/released only by
    the magnet weld and exits cleanly without brushing the pocket walls. Use this for
    the throw task (the pocket-exit deflection otherwise perturbs the release velocity).
    Note: the catch phase needs a collidable glove surface to trigger the magnet.

    This is a preview/control-future variant. It does not alter the existing
    catcher or strike end-effector paths. The glove pocket opens along the
    glove body's local +z axis, and ``ee_baseball_glove_catcher_site`` marks
    the intended ball hold point. The glove body origin is the proximal wrist
    base; palm and fingers extend distally along the glove body's local -y.
    """
    tree = ET.parse(src_xml)
    root = tree.getroot()
    if bool(glove_weld_stiff):
        timestep_s = _xml_timestep_s(src_xml)
        weld_solref = (
            glove_weld_stiff_solref_for_timestep(timestep_s)
            if glove_weld_solref is None
            else tuple(float(v) for v in glove_weld_solref)
        )
        weld_solimp = (
            GLOVE_WELD_STIFF_SOLIMP
            if glove_weld_solimp is None
            else tuple(float(v) for v in glove_weld_solimp)
        )
    else:
        weld_solref = (
            GLOVE_WELD_SOFT_SOLREF
            if glove_weld_solref is None
            else tuple(float(v) for v in glove_weld_solref)
        )
        weld_solimp = (
            GLOVE_WELD_SOFT_SOLIMP
            if glove_weld_solimp is None
            else tuple(float(v) for v in glove_weld_solimp)
        )
    target_body = root.find(f".//body[@name='{body_name}']")
    if target_body is None:
        raise ValueError(f"XML has no body named {body_name!r}.")

    roll_body_name = f"{glove_name}_wrist_roll_body"
    pitch_body_name = f"{glove_name}_wrist_pitch_body"
    glove_body_name = f"{glove_name}_body"
    roll_joint_name = f"{glove_name}_wrist_roll"
    pitch_joint_name = f"{glove_name}_wrist_pitch"
    weld_name = f"{ball_name}_to_{glove_name}_weld"

    for child in list(target_body):
        if child.attrib.get("name") == roll_body_name:
            target_body.remove(child)

    roll_lo, roll_hi = (float(v) for v in wrist_roll_range_deg)
    pitch_lo, pitch_hi = (float(v) for v in wrist_pitch_range_deg)
    if not roll_lo < roll_hi:
        raise ValueError("wrist_roll_range_deg must be increasing")
    if not pitch_lo < pitch_hi:
        raise ValueError("wrist_pitch_range_deg must be increasing")

    tan = "0.64 0.40 0.18 1.00"
    dark_tan = "0.38 0.22 0.10 1.00"
    lace = "0.14 0.07 0.03 1.00"
    common = {
        "condim": "3",
        "friction": "1.3 0.04 0.002",
        "solref": "0.010000 1.000000",
        "solimp": "0.920000 0.980000 0.001000",
    }
    # Pocket geoms (floor, rim, fingers, thumb, heel) use pocket_common. When
    # pocket_collision is False they stay visible but non-colliding (contype/conaffinity 0).
    pocket_common = dict(common)
    if not pocket_collision:
        pocket_common["contype"] = "0"
        pocket_common["conaffinity"] = "0"

    roll_body = ET.Element("body", {"name": roll_body_name, "pos": "0.000000 0.000000 0.000000"})
    roll_body.append(
        ET.Element(
            "inertial",
            {
                "pos": "0.000000 0.000000 0.000000",
                "mass": "0.001000",
                "diaginertia": "0.000001 0.000001 0.000001",
            },
        )
    )
    roll_body.append(
        ET.Element(
            "joint",
            {
                "name": roll_joint_name,
                "type": "hinge",
                "axis": "0 0 1",
                "range": f"{roll_lo:.6f} {roll_hi:.6f}",
                "limited": "true",
                "damping": "0.030000",
                "frictionloss": "0.000000",
                "armature": "0.000200",
            },
        )
    )
    pitch_body = ET.Element("body", {"name": pitch_body_name, "pos": "0.000000 0.000000 0.000000"})
    pitch_body.append(
        ET.Element(
            "inertial",
            {
                "pos": "0.000000 0.000000 0.000000",
                "mass": "0.001000",
                "diaginertia": "0.000001 0.000001 0.000001",
            },
        )
    )
    pitch_body.append(
        ET.Element(
            "joint",
            {
                "name": pitch_joint_name,
                "type": "hinge",
                "axis": "1 0 0",
                "range": f"{pitch_lo:.6f} {pitch_hi:.6f}",
                "limited": "true",
                "damping": "0.030000",
                "frictionloss": "0.000000",
                "armature": "0.000200",
            },
        )
    )
    pocket_x, pocket_y, pocket_z = (float(v) for v in wrist_to_pocket_offset_m)
    glove_body = ET.Element(
        "body",
        {
            "name": glove_body_name,
            "pos": "0.000000 0.000000 0.000000",
            "quat": "0.000000000 0.000000000 0.707106781 0.707106781",
        },
    )

    floor_hz = float(pocket_floor_half_thickness_m)
    floor_radius = float(pocket_floor_radius_m)
    inner_radius = float(pocket_inner_radius_m)
    rim_r = float(rim_radius_m)
    hold_z = float(ball_radius_m)
    if min(floor_hz, floor_radius, inner_radius, rim_r, float(pocket_depth_m), hold_z) <= 0.0:
        raise ValueError("glove dimensions must be positive")
    if floor_radius < inner_radius:
        raise ValueError("pocket_floor_radius_m must be at least pocket_inner_radius_m")
    if float(pocket_depth_m) <= rim_r:
        raise ValueError("pocket_depth_m must exceed rim_radius_m")
    rim_center = inner_radius
    rim_z = float(pocket_depth_m) - rim_r

    pocket_len = float(math.sqrt(pocket_x * pocket_x + pocket_y * pocket_y + pocket_z * pocket_z))
    if pocket_len > 1e-8:
        glove_body.append(
            ET.Element(
                "geom",
                {
                    "name": f"{glove_name}_wrist_mount_stub_geom",
                    "type": "capsule",
                    "fromto": (
                        "0.000000 0.000000 0.000000 "
                        f"{pocket_x * 0.34:.6f} {pocket_y * 0.34:.6f} {pocket_z * 0.34:.6f}"
                    ),
                    "size": "0.006000",
                    "rgba": "0.33 0.18 0.08 1.00",
                    "mass": "0.003000",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
        )

    glove_body.append(
        ET.Element(
            "geom",
            {
                **pocket_common,
                "name": f"{glove_name}_pocket_floor_geom",
                "type": "cylinder",
                "pos": f"{pocket_x:.6f} {pocket_y:.6f} {pocket_z - floor_hz:.6f}",
                "size": f"{floor_radius:.6f} {floor_hz:.6f}",
                "rgba": tan,
                "mass": "0.018000",
            },
        )
    )

    rim_count = max(6, int(rim_elements))
    for idx in range(rim_count):
        theta0 = 2.0 * math.pi * float(idx) / float(rim_count)
        theta1 = 2.0 * math.pi * float(idx + 1) / float(rim_count)
        x0 = pocket_x + rim_center * math.cos(theta0)
        y0 = pocket_y + rim_center * math.sin(theta0)
        x1 = pocket_x + rim_center * math.cos(theta1)
        y1 = pocket_y + rim_center * math.sin(theta1)
        glove_body.append(
            ET.Element(
                "geom",
                {
                    **pocket_common,
                    "name": f"{glove_name}_rim_{idx:02d}_geom",
                    "type": "capsule",
                    "fromto": (
                        f"{x0:.6f} {y0:.6f} {pocket_z + rim_z:.6f} "
                        f"{x1:.6f} {y1:.6f} {pocket_z + rim_z:.6f}"
                    ),
                    "size": f"{rim_r:.6f}",
                    "rgba": tan,
                    "mass": "0.000450",
                },
            )
        )

    # Asymmetric lobes make axial roll visible in preview renders.
    finger_base_y = pocket_y - inner_radius - 0.006
    finger_tip_y = finger_base_y - 0.026
    for idx, x in enumerate((-0.016, 0.0, 0.016)):
        glove_body.append(
            ET.Element(
                "geom",
                {
                    **pocket_common,
                    "name": f"{glove_name}_finger_{idx}_geom",
                    "type": "capsule",
                    "fromto": (
                        f"{pocket_x + x:.6f} {finger_base_y:.6f} {pocket_z + 0.003500:.6f} "
                        f"{pocket_x + x:.6f} {finger_tip_y:.6f} {pocket_z + 0.011000:.6f}"
                    ),
                    "size": "0.007000",
                    "rgba": tan,
                    "mass": "0.001300",
                },
            )
        )

    glove_body.append(
        ET.Element(
            "geom",
            {
                **pocket_common,
                "name": f"{glove_name}_thumb_geom",
                "type": "capsule",
                "fromto": (
                    f"{pocket_x - inner_radius - 0.008:.6f} {pocket_y + 0.020000:.6f} {pocket_z - 0.001500:.6f} "
                    f"{pocket_x - inner_radius - 0.026:.6f} {pocket_y - 0.026000:.6f} {pocket_z + 0.010000:.6f}"
                ),
                "size": "0.008000",
                "rgba": dark_tan,
                "mass": "0.001600",
            },
        )
    )
    glove_body.append(
        ET.Element(
            "geom",
            {
                **pocket_common,
                "name": f"{glove_name}_heel_geom",
                "type": "capsule",
                "fromto": (
                    f"{pocket_x - 0.024000:.6f} {pocket_y + inner_radius + 0.012:.6f} {pocket_z - 0.003000:.6f} "
                    f"{pocket_x + 0.024000:.6f} {pocket_y + inner_radius + 0.012:.6f} {pocket_z - 0.003000:.6f}"
                ),
                "size": "0.009000",
                "rgba": dark_tan,
                "mass": "0.001800",
            },
        )
    )

    for idx, x in enumerate((-0.014, 0.0, 0.014)):
        glove_body.append(
            ET.Element(
                "geom",
                {
                    "name": f"{glove_name}_lace_{idx}_geom",
                    "type": "capsule",
                    "fromto": (
                        f"{pocket_x + x:.6f} {pocket_y + 0.018000:.6f} {pocket_z + 0.004000:.6f} "
                        f"{pocket_x + x * 0.65:.6f} {pocket_y - 0.024000:.6f} {pocket_z + 0.010000:.6f}"
                    ),
                    "size": "0.001200",
                    "rgba": lace,
                    "mass": "0.000100",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )
        )

    glove_body.append(
        ET.Element(
            "site",
            {
                "name": f"{glove_name}_catcher_site",
                "type": "sphere",
                "pos": f"{pocket_x:.6f} {pocket_y:.6f} {pocket_z + hold_z:.6f}",
                "size": f"{max(0.005, inner_radius * 0.16):.6f}",
                "rgba": "1.00 0.92 0.30 0.30",
                "group": "2",
            },
        )
    )

    pitch_body.append(glove_body)
    roll_body.append(pitch_body)
    target_body.append(roll_body)

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for child in list(actuator):
        if child.attrib.get("name") in {f"{roll_joint_name}_motor", f"{pitch_joint_name}_motor"}:
            actuator.remove(child)
    actuator.append(
        ET.Element(
            "motor",
            {
                "name": f"{roll_joint_name}_motor",
                "joint": roll_joint_name,
                "ctrlrange": "-1.500000 1.500000",
                "forcelimited": "true",
                "forcerange": "-2.000000 2.000000",
            },
        )
    )
    actuator.append(
        ET.Element(
            "motor",
            {
                "name": f"{pitch_joint_name}_motor",
                "joint": pitch_joint_name,
                "ctrlrange": "-1.500000 1.500000",
                "forcelimited": "true",
                "forcerange": "-2.000000 2.000000",
            },
        )
    )

    equality = root.find("equality")
    if equality is None:
        contact = root.find("contact")
        insert_at = list(root).index(contact) if contact is not None else len(list(root))
        equality = ET.Element("equality")
        root.insert(insert_at, equality)
    for child in list(equality):
        if child.attrib.get("name") == weld_name:
            equality.remove(child)
    equality.append(
        ET.Element(
            "weld",
            {
                "name": weld_name,
                "body1": ball_name,
                "body2": glove_body_name,
                "active": "false",
                "solref": f"{float(weld_solref[0]):.6f} {float(weld_solref[1]):.6f}",
                "solimp": (
                    f"{float(weld_solimp[0]):.6f} "
                    f"{float(weld_solimp[1]):.6f} "
                    f"{float(weld_solimp[2]):.6f}"
                ),
            },
        )
    )

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def add_ball_to_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    radius_m: float = 0.018,
    mass_kg: float = 0.025,
    ball_solref: tuple[float, float] = (0.01, 1.0),
    ball_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
) -> None:
    """Add one free spherical ball to an existing MuJoCo XML."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML has no <worldbody> element.")

    for child in list(worldbody):
        if child.attrib.get("name") == ball_name:
            worldbody.remove(child)

    x, y, z = (float(v) for v in initial_pos_m)
    ball_body = ET.Element("body", {"name": ball_name, "pos": f"{x:.6f} {y:.6f} {z:.6f}"})
    ball_body.append(ET.Element("freejoint", {"name": f"{ball_name}_freejoint"}))
    ball_body.append(
        ET.Element(
            "geom",
            {
                "name": f"{ball_name}_geom",
                "type": "sphere",
                "size": f"{float(radius_m):.6f}",
                "mass": f"{float(mass_kg):.6f}",
                "rgba": "0.95 0.85 0.15 1",
                "condim": "3",
                "friction": "0.8 0.02 0.001",
                "solref": f"{float(ball_solref[0]):.6f} {float(ball_solref[1]):.6f}",
                "solimp": (
                    f"{float(ball_solimp[0]):.6f} "
                    f"{float(ball_solimp[1]):.6f} "
                    f"{float(ball_solimp[2]):.6f}"
                ),
            },
        )
    )
    ball_body.append(
        ET.Element(
            "site",
            {
                "name": f"{ball_name}_site",
                "type": "sphere",
                "size": f"{float(radius_m) * 0.45:.6f}",
                "rgba": "1 1 1 0.7",
                "group": "2",
            },
        )
    )
    worldbody.append(ball_body)

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def add_mocap_ball_to_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    radius_m: float = 0.018,
    mass_kg: float = 0.025,
) -> None:
    """Add one kinematic mocap-controlled spherical ball to an XML."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("XML has no <worldbody> element.")

    for child in list(worldbody):
        if child.attrib.get("name") == ball_name:
            worldbody.remove(child)

    x, y, z = (float(v) for v in initial_pos_m)
    ball_body = ET.Element(
        "body",
        {"name": ball_name, "mocap": "true", "pos": f"{x:.6f} {y:.6f} {z:.6f}"},
    )
    ball_body.append(
        ET.Element(
            "geom",
            {
                "name": f"{ball_name}_geom",
                "type": "sphere",
                "size": f"{float(radius_m):.6f}",
                "mass": f"{float(mass_kg):.6f}",
                "rgba": "0.95 0.85 0.15 1",
                "condim": "3",
                "friction": "0.8 0.02 0.001",
                "solref": "0.01 1",
                "solimp": "0.9 0.95 0.001",
            },
        )
    )
    ball_body.append(
        ET.Element(
            "site",
            {
                "name": f"{ball_name}_site",
                "type": "sphere",
                "size": f"{float(radius_m) * 0.45:.6f}",
                "rgba": "1 1 1 0.7",
                "group": "2",
            },
        )
    )
    worldbody.append(ball_body)

    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)


def make_motorized_ball_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    radius_m: float = 0.018,
    mass_kg: float = 0.025,
    ball_solref: tuple[float, float] = (0.01, 1.0),
    ball_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
) -> int:
    """Create an MJX-compatible motorized UMARM XML with one free ball."""
    tmp_motor = dst_xml.with_name(dst_xml.stem + "_motor_only.xml")
    replaced = make_force_motor_xml(src_xml, tmp_motor, make_motorized_xml)
    add_ball_to_xml(
        tmp_motor,
        dst_xml,
        ball_name=ball_name,
        initial_pos_m=initial_pos_m,
        radius_m=radius_m,
        mass_kg=mass_kg,
        ball_solref=ball_solref,
        ball_solimp=ball_solimp,
    )
    try:
        tmp_motor.unlink()
    except OSError:
        pass
    return replaced


def make_motorized_paddle_ball_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    ball_radius_m: float = 0.018,
    ball_mass_kg: float = 0.025,
    paddle_radius_m: float = 0.055,
    paddle_half_thickness_m: float = 0.004,
    paddle_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    paddle_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> int:
    """Create a motorized UMARM XML with an end-effector paddle and one free ball."""
    tmp_motor = dst_xml.with_name(dst_xml.stem + "_motor_only.xml")
    tmp_paddle = dst_xml.with_name(dst_xml.stem + "_paddle_only.xml")
    replaced = make_force_motor_xml(src_xml, tmp_motor, make_motorized_xml)
    add_paddle_to_xml(
        tmp_motor,
        tmp_paddle,
        radius_m=paddle_radius_m,
        half_thickness_m=paddle_half_thickness_m,
        offset_m=paddle_offset_m,
        euler_deg=paddle_euler_deg,
    )
    add_ball_to_xml(
        tmp_paddle,
        dst_xml,
        ball_name=ball_name,
        initial_pos_m=initial_pos_m,
        radius_m=ball_radius_m,
        mass_kg=ball_mass_kg,
    )
    for path in (tmp_motor, tmp_paddle):
        try:
            path.unlink()
        except OSError:
            pass
    return replaced


def make_motorized_catcher_ball_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    ball_radius_m: float = 0.018,
    ball_mass_kg: float = 0.025,
    catcher_variant: str = "wide_posts",
    catcher_half_width_m: float = 0.045,
    catcher_floor_half_thickness_m: float = 0.004,
    catcher_wall_half_thickness_m: float = 0.004,
    catcher_wall_height_m: float = 0.044,
    catcher_floor_offset_m: tuple[float, float, float] = (0.0, 0.0, -0.022),
    tight_inner_radius_m: float = 0.022,
    tight_wall_element_radius_m: float = 0.003,
    tight_wall_elements: int = 36,
    catcher_mount_offset_m: tuple[float, float, float] | None = None,
    catcher_mount_euler_deg: tuple[float, float, float] | None = None,
    catcher_mount_quat: tuple[float, float, float, float] | None = None,
    catcher_mount_stub_radius_m: float = 0.0035,
    catcher_solref: tuple[float, float] = (0.012, 1.0),
    catcher_solimp: tuple[float, float, float] = (0.92, 0.98, 0.001),
    ball_solref: tuple[float, float] = (0.01, 1.0),
    ball_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
    catcher_wrist_roll_joint: bool = False,
    catcher_wrist_roll_range_deg: tuple[float, float] = (-180.0, 180.0),
) -> int:
    """Create a motorized UMARM XML with an end-effector catcher and one free ball."""
    tmp_motor = dst_xml.with_name(dst_xml.stem + "_motor_only.xml")
    tmp_catcher = dst_xml.with_name(dst_xml.stem + "_catcher_only.xml")
    replaced = make_force_motor_xml(src_xml, tmp_motor, make_motorized_xml)
    add_catcher_to_xml(
        tmp_motor,
        tmp_catcher,
        catcher_variant=catcher_variant,
        half_width_m=catcher_half_width_m,
        floor_half_thickness_m=catcher_floor_half_thickness_m,
        wall_half_thickness_m=catcher_wall_half_thickness_m,
        wall_height_m=catcher_wall_height_m,
        floor_offset_m=catcher_floor_offset_m,
        tight_inner_radius_m=tight_inner_radius_m,
        tight_wall_element_radius_m=tight_wall_element_radius_m,
        tight_wall_elements=tight_wall_elements,
        catcher_mount_offset_m=catcher_mount_offset_m,
        catcher_mount_euler_deg=catcher_mount_euler_deg,
        catcher_mount_quat=catcher_mount_quat,
        catcher_mount_stub_radius_m=catcher_mount_stub_radius_m,
        catcher_solref=catcher_solref,
        catcher_solimp=catcher_solimp,
        wrist_roll_joint=catcher_wrist_roll_joint,
        wrist_roll_range_deg=catcher_wrist_roll_range_deg,
    )
    add_ball_to_xml(
        tmp_catcher,
        dst_xml,
        ball_name=ball_name,
        initial_pos_m=initial_pos_m,
        radius_m=ball_radius_m,
        mass_kg=ball_mass_kg,
        ball_solref=ball_solref,
        ball_solimp=ball_solimp,
    )
    for path in (tmp_motor, tmp_catcher):
        try:
            path.unlink()
        except OSError:
            pass
    return replaced


def make_motorized_catcher_ball_base_yaw_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    base_yaw_joint_name: str = "base_yaw",
    base_yaw_range_deg: tuple[float, float] = (-180.0, 180.0),
    remove_world_axes: bool = False,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    ball_radius_m: float = 0.018,
    ball_mass_kg: float = 0.025,
    catcher_variant: str = "wide_posts",
    catcher_half_width_m: float = 0.045,
    catcher_floor_half_thickness_m: float = 0.004,
    catcher_wall_half_thickness_m: float = 0.004,
    catcher_wall_height_m: float = 0.044,
    catcher_floor_offset_m: tuple[float, float, float] = (0.0, 0.0, -0.022),
    tight_inner_radius_m: float = 0.022,
    tight_wall_element_radius_m: float = 0.003,
    tight_wall_elements: int = 36,
    catcher_mount_offset_m: tuple[float, float, float] | None = None,
    catcher_mount_euler_deg: tuple[float, float, float] | None = None,
    catcher_mount_quat: tuple[float, float, float, float] | None = None,
    catcher_mount_stub_radius_m: float = 0.0035,
    catcher_solref: tuple[float, float] = (0.012, 1.0),
    catcher_solimp: tuple[float, float, float] = (0.92, 0.98, 0.001),
    ball_solref: tuple[float, float] = (0.01, 1.0),
    ball_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
    catcher_wrist_roll_joint: bool = False,
    catcher_wrist_roll_range_deg: tuple[float, float] = (-180.0, 180.0),
) -> int:
    """Create a motorized catcher-ball XML variant with a base-yaw turntable."""
    tmp_catcher = dst_xml.with_name(dst_xml.stem + "_fixed_base_catcher_only.xml")
    replaced = make_motorized_catcher_ball_xml(
        src_xml,
        tmp_catcher,
        ball_name=ball_name,
        initial_pos_m=initial_pos_m,
        ball_radius_m=ball_radius_m,
        ball_mass_kg=ball_mass_kg,
        catcher_variant=catcher_variant,
        catcher_half_width_m=catcher_half_width_m,
        catcher_floor_half_thickness_m=catcher_floor_half_thickness_m,
        catcher_wall_half_thickness_m=catcher_wall_half_thickness_m,
        catcher_wall_height_m=catcher_wall_height_m,
        catcher_floor_offset_m=catcher_floor_offset_m,
        tight_inner_radius_m=tight_inner_radius_m,
        tight_wall_element_radius_m=tight_wall_element_radius_m,
        tight_wall_elements=tight_wall_elements,
        catcher_mount_offset_m=catcher_mount_offset_m,
        catcher_mount_euler_deg=catcher_mount_euler_deg,
        catcher_mount_quat=catcher_mount_quat,
        catcher_mount_stub_radius_m=catcher_mount_stub_radius_m,
        catcher_solref=catcher_solref,
        catcher_solimp=catcher_solimp,
        ball_solref=ball_solref,
        ball_solimp=ball_solimp,
        catcher_wrist_roll_joint=catcher_wrist_roll_joint,
        catcher_wrist_roll_range_deg=catcher_wrist_roll_range_deg,
    )
    add_base_yaw_to_xml(
        tmp_catcher,
        dst_xml,
        joint_name=base_yaw_joint_name,
        range_deg=base_yaw_range_deg,
        remove_world_axes=remove_world_axes,
    )
    try:
        tmp_catcher.unlink()
    except OSError:
        pass
    return replaced


def make_motorized_baseball_glove_ball_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    ball_radius_m: float = 0.018,
    ball_mass_kg: float = 0.025,
    ball_solref: tuple[float, float] = (0.01, 1.0),
    ball_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
    glove_name: str = "ee_baseball_glove",
    wrist_roll_range_deg: tuple[float, float] = (-180.0, 180.0),
    wrist_pitch_range_deg: tuple[float, float] = (-90.0, 90.0),
    pocket_inner_radius_m: float = 0.026,
    pocket_floor_radius_m: float = 0.028,
    pocket_floor_half_thickness_m: float = 0.003,
    pocket_depth_m: float = 0.014,
    rim_radius_m: float = 0.0035,
    rim_elements: int = 8,
    wrist_to_pocket_offset_m: tuple[float, float, float] = (0.0, -0.050, 0.0),
    glove_weld_solref: tuple[float, float] | None = None,
    glove_weld_solimp: tuple[float, float, float] | None = None,
    glove_weld_stiff: bool = False,
    pocket_collision: bool = True,
) -> int:
    """Create the preview-only two-DOF baseball-glove catch/throw XML variant."""
    tmp_motor = dst_xml.with_name(dst_xml.stem + "_motor_only.xml")
    tmp_ball = dst_xml.with_name(dst_xml.stem + "_ball_only.xml")
    replaced = make_force_motor_xml(src_xml, tmp_motor, make_motorized_xml)
    add_ball_to_xml(
        tmp_motor,
        tmp_ball,
        ball_name=ball_name,
        initial_pos_m=initial_pos_m,
        radius_m=ball_radius_m,
        mass_kg=ball_mass_kg,
        ball_solref=ball_solref,
        ball_solimp=ball_solimp,
    )
    add_baseball_glove_to_xml(
        tmp_ball,
        dst_xml,
        glove_name=glove_name,
        ball_name=ball_name,
        ball_radius_m=ball_radius_m,
        wrist_roll_range_deg=wrist_roll_range_deg,
        wrist_pitch_range_deg=wrist_pitch_range_deg,
        pocket_inner_radius_m=pocket_inner_radius_m,
        pocket_floor_radius_m=pocket_floor_radius_m,
        pocket_floor_half_thickness_m=pocket_floor_half_thickness_m,
        pocket_depth_m=pocket_depth_m,
        rim_radius_m=rim_radius_m,
        rim_elements=rim_elements,
        wrist_to_pocket_offset_m=wrist_to_pocket_offset_m,
        glove_weld_solref=glove_weld_solref,
        glove_weld_solimp=glove_weld_solimp,
        glove_weld_stiff=glove_weld_stiff,
        pocket_collision=pocket_collision,
    )
    for path in (tmp_motor, tmp_ball):
        try:
            path.unlink()
        except OSError:
            pass
    return replaced


def make_motorized_baseball_glove_ball_base_yaw_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    base_yaw_joint_name: str = "base_yaw",
    base_yaw_range_deg: tuple[float, float] = (-140.0, 140.0),
    base_yaw_damping: float = 0.5,
    base_yaw_servo_kp: float = 40.0,
    base_yaw_servo_kv: float = 8.0,
    base_yaw_servo_forcerange: tuple[float, float] = (-30.0, 30.0),
    remove_world_axes: bool = False,
    **glove_kwargs,
) -> int:
    """Baseball-glove catch/throw XML variant with a base-yaw turntable at the arm mount.

    Composes :func:`make_motorized_baseball_glove_ball_xml` (all keyword arguments are
    forwarded) with :func:`add_base_yaw_to_xml`, mirroring the catcher-variant composition,
    and adds a position servo (`<position>` actuator) on the yaw joint so the policy
    commands a target angle rather than a torque.
    """
    tmp_glove = dst_xml.with_name(dst_xml.stem + "_fixed_base_glove_only.xml")
    replaced = make_motorized_baseball_glove_ball_xml(src_xml, tmp_glove, **glove_kwargs)
    add_base_yaw_to_xml(
        tmp_glove,
        dst_xml,
        joint_name=base_yaw_joint_name,
        range_deg=base_yaw_range_deg,
        remove_world_axes=remove_world_axes,
        damping=base_yaw_damping,
    )

    tree = ET.parse(dst_xml)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    servo_name = f"{base_yaw_joint_name}_servo"
    for child in list(actuator):
        if child.attrib.get("name") == servo_name:
            actuator.remove(child)
    lo_rad = math.radians(float(base_yaw_range_deg[0]))
    hi_rad = math.radians(float(base_yaw_range_deg[1]))
    f_lo, f_hi = (float(v) for v in base_yaw_servo_forcerange)
    actuator.append(
        ET.Element(
            "position",
            {
                "name": servo_name,
                "joint": base_yaw_joint_name,
                "kp": f"{float(base_yaw_servo_kp):.6f}",
                "kv": f"{float(base_yaw_servo_kv):.6f}",
                "ctrlrange": f"{lo_rad:.6f} {hi_rad:.6f}",
                "forcelimited": "true",
                "forcerange": f"{f_lo:.6f} {f_hi:.6f}",
            },
        )
    )
    _indent_xml(root)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)

    try:
        tmp_glove.unlink()
    except OSError:
        pass
    return replaced


def make_motorized_mocap_ball_xml(
    src_xml: Path,
    dst_xml: Path,
    *,
    ball_name: str = "task_ball",
    initial_pos_m: tuple[float, float, float] = (0.10, -0.20, 0.35),
    radius_m: float = 0.018,
    mass_kg: float = 0.025,
) -> int:
    """Create an MJX-compatible motorized UMARM XML with one kinematic mocap ball."""
    tmp_motor = dst_xml.with_name(dst_xml.stem + "_motor_only.xml")
    replaced = make_motorized_xml(src_xml, tmp_motor)
    add_mocap_ball_to_xml(
        tmp_motor,
        dst_xml,
        ball_name=ball_name,
        initial_pos_m=initial_pos_m,
        radius_m=radius_m,
        mass_kg=mass_kg,
    )
    try:
        tmp_motor.unlink()
    except OSError:
        pass
    return replaced
