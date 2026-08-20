"""The Kinova Gen3 half of the lab: connect, move safely, tie it to the cameras.

``README.md`` in this folder is the entry point; this docstring is the map.

    setup_env.py          builds .venv_kinova - the ONE interpreter that can
                          import kortex_api, because kortex_api pins a 2017
                          protobuf nothing else in this repo may see
    vendor/               the lab's portable driver, verbatim (PROVENANCE.md)
    kinova_arm.py         SafeKinovaArm: keep-out envelope, speed-limited moves
    kinova_mocap.py       KinovaMocapRx: rigid body 1008 and its four markers
    mocap_calibration.py  the robot-world / hand-eye solve.  PURE MATH - it
                          imports no kortex_api, so it is tested against
                          synthetic truth with no hardware anywhere
    calibrate_mocap.py    the 21-pose campaign and its analysis
    arm_bridge.py         the stdio-JSON bridge, run ON .venv_kinova
    bridge_client.py      KinovaLink: spawns the bridge from an ordinary
                          interpreter and never imports kortex_api itself
    bridge_mocap.py       the mocap half that lives inside the bridge process
    sensor_frame.py       command the MOCAP frame instead of the arm's frame
    kinova_kinematics.py  Gen3 FK/IK/reach, mujoco + numpy only
    kinova_scene.py       a standalone Gen3 MJCF builder, assets/kinova_gen3/

Nothing here is imported at package level, deliberately: importing this package
must not require ``kortex_api``, so that ``mocap_calibration`` and
``bridge_mocap`` stay usable — and testable — from the workspace's ordinary
interpreter.

PORT NOTE (workspace port, 2026-08-20).  ``pad_frames.py`` and
``check_pad_markers.py`` were left behind: both are about the printed collision
pad rather than the Gen3 controller, and both reach into ``UMArm_COLLAB``.  See
``PORTING.md`` beside this file.
"""

__all__ = []
