# What the RS485 twin does, read out of its collision branch

These five files are a read of `C:\RUNZE_SRC\RS485_VEMA\.claude\worktrees\kmppi-collision`
(branch `feat/kmppi-collision`) taken on 2026-09-10, before any of `digital_twin/`
was written. They are kept in the repo rather than regenerated because the source
is a *worktree of another project*: other agents work there, it moves, and a
report of what it said on the day the port was made is the only durable record of
what the port was made from.

| file | what it covers |
| --- | --- |
| `sim_core.md` | `SimArm`/`SimNode`, the 1 ms quantum, `advance_to`'s determinism contract, `SimMaster`, and **the invariant list** (§7) that *is* the sim/real protocol |
| `actuator_net.md` | the flow net: architecture, features, the shooting/BPTT trainer, the checkpoint format, and every physical constraint baked in |
| `mjcf_fit.md` | the MuJoCo model's tunables, `ring_analysis` → `fit_bounce` → `twin_compare` → `force_audit` |
| `collect_video.md` | the simultaneous-hardware collection loop, `scene_camera`, and how the side-by-side video is composited |
| `here_hw.md` | this repo's own hardware API, read at the same time so the two sit side by side |

Read `../CONTRACT.md` for what this workspace actually implements, and why it
departs from the above where it does. Where the two disagree, `CONTRACT.md` wins:
these files describe a **different plant**.
