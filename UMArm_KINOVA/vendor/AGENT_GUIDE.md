# Coding-Agent Cheat-Sheet — Kinova Gen3

Condensed reference for writing code against the lab's Kinova Gen3 arm. Read
this first, then `README.md` for install/network details.

## TL;DR

```python
from kinova_driver import KinovaArm

with KinovaArm(ip="192.168.1.10") as arm:   # default IP/creds: 192.168.1.10 / admin / admin
    pose = arm.get_pose()                   # [x,y,z (m), theta_x,y,z (deg)] in BASE frame
    arm.move_home()                         # built-in Home (blocking)
    arm.move_to_pose([0.45, 0.0, 0.35, 180.0, 0.0, 90.0])   # absolute (blocking)
    arm.move_delta(dx=0.05, dz=-0.03)       # base-frame relative (blocking)
    arm.move_tool_frame_delta(dz=0.02)      # tool-frame relative (blocking)
    arm.move_joints([0,0,0,0,0,0,0])        # absolute joint angles, deg (blocking)
    arm.send_twist(linear=[0,0,0.01], frame="tool"); arm.stop()   # velocity (continuous!)
    arm.smooth_move("z", 0.01, duration_s=4.0, frame="tool")      # ramped velocity (blocking)
```

## Units & frames (do not get these wrong)

- **Position: meters. Orientation: degrees** (intrinsic XYZ Euler `theta_x/y/z`).
- A **pose** is always `[x, y, z, theta_x, theta_y, theta_z]`.
- Default reference is the **robot base frame**. Twists accept `frame="base"`
  or `frame="tool"`.
- Twist **linear is m/s**, **angular is deg/s**.

## Two control modes

1. **Action / pose moves** (`move_to_pose`, `move_delta`, `move_tool_frame_delta`,
   `move_joints`, `move_home`): high-level, **blocking** by default (wait for
   END/ABORT, 20 s timeout). Pass `wait=False` to fire-and-forget.
2. **Twist / velocity** (`send_twist`): **non-blocking and continuous** — the arm
   keeps moving until another twist or `arm.stop()`. Use `smooth_move(...)` for a
   cosine-ramped single-axis move that stops itself.

## Reading state

- `arm.get_pose()` -> 6-list (m, deg), base frame.
- `arm.get_pose_SE3()` -> 4x4 numpy SE(3) matrix, base frame.
- `arm.get_joint_angles()` -> per-actuator angle list (deg); length 7 on a Gen3.

## SE(3) helpers (no hardware needed)

```python
from kinova_driver import pose_to_SE3, SE3_to_pose
T = pose_to_SE3([0.4, 0.0, 0.3, 180, 0, 90])   # 6-list -> 4x4
pose = SE3_to_pose(T)                            # 4x4 -> 6-list
```

## Raw Kortex access (when the wrapper isn't enough)

`KinovaArm` exposes the underlying services:
- `arm.base` -> `BaseClient` (actions, `SendTwistCommand`, notifications, `Stop`).
- `arm.base_cyclic` -> `BaseCyclicClient` (`RefreshFeedback()` for tool pose,
  actuator positions, currents, etc.).

Underlying message/enum modules:
```python
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
# e.g. Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL, Base_pb2.SINGLE_LEVEL_SERVOING
```

## Connecting without the wrapper (low-level pattern)

```python
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

args = utilities.parseConnectionArguments()      # reads --ip/-u/-p, defaults to 192.168.1.10
with utilities.DeviceConnection.createTcpConnection(args) as router:
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    fb = base_cyclic.RefreshFeedback()
    print(fb.base.tool_pose_x, fb.base.tool_pose_theta_z)
```
Use `createUdpConnection` instead for high-rate (1 kHz) cyclic control.

## Gotchas

- `send_twist` never stops on its own — always follow with `arm.stop()` (the
  `with` block also stops on exit).
- Pose moves silently no-op if the arm is in a fault state; clear faults in the
  Web App (`http://192.168.1.10`).
- protobuf MUST stay at `3.5.1`; a newer protobuf breaks the Kortex stubs.
- The bundled wheel is pure Python (`py3-none-any`) — same file works on any OS.
