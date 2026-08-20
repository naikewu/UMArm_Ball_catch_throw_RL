# UMArm Koopman compliance-control workspace

This workspace hosts the bench software for a 24-actuator pneumatic UMArm whose upper eight
boards are TLE92464-driven proportional valves and whose lower sixteen boards remain the legacy
Clippard 7 mm bang-bang boards, together with the motion-capture, Kinova, and MuJoCo pieces the
compliance-control experiments need. Both firmware trees are carried here in buildable form
(`firmware/tle`, `firmware/legacy`) alongside the exact images currently flashed on the arm
(`firmware/images`); the host side keeps the TLE tooling (`TLE_PCB`), the legacy CAN and mocap
bundle (`legacy_host`), and one bench-wide constants module (`bench_env.py`) that resolves the
USB–CAN adapter by USB identity rather than by a COM number. This README is a stub — the
supervisor finalizes it.
