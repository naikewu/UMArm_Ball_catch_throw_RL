# Try Dynamic Controller on the UMArm ProMax Simulation

## Background

Previously, we have tried implmeneting a koopman MPPI controller on the original UMArm, in repo: (C:\RUNZE_SRC\RS485_VEMA). The koopman MPPI is on one of the worktree.

Then, we developed the new digital twin here in the current repo. The major difference between the Original UMArm and the ProMax UMarm is:

1. The ProMax UMarm is significantly bigger
2. The ProMax UMArm uses a distinct Y struct and actuator tendon + bearing routing scheme.
3. The ProMax UMArm features CAN bus communication for all the actuators, meaning that every actuator receives and feedback pressures at 150Hz, synchronized.
4. Assume the MOCAP will be run at 240Hz. Add the proper noise to the simulated MOCAP for use to test more realistically.
5. The ProMax UMArm uses the TLE PCB and the Clippard DVP proportional valves for top 8 actuators, featuring much smaller tracking error, and much faster flowrate compared to the 7mm actuator

## Problem statement

You need to implement the Koopman MPPI controller, and a feedforward PID controller, and a vanila PID controller on the UMArm ProMax, and do the 'Soft' word writing demo. All controller are present in the RS485 repo, in various work trees.

Deliverables:

1. Koopman MPPI controller, feedforward PID controller, and vanila PID controllers, selectable.
2. Videos of the 'Soft' work writing demo, side-by-side comparison for all 3 controllers.
3. In the canarm_control_gui, if controller started (run controller on seperate process), pop up a new slide gui that let me control joint angles, and another button area that let me nudge the tip position, which the robot will move the tip to my given target location.
4. Concise report showing the koopman training result, benchmark numbers, etc.

## Other informations

1. The 'Soft' writing size and speed should adjust according to the size different of the original vs promax UMArm.
2. If given a scaled correctly demo task, I expect the new hardware to have better performance, given the upgrade in valves, and electronics. You must make change to the controller, such that it not only compatible with the new faster report rate and valve flowrate, but also fully exploit the advantage of the hardware upgrade.
3. You should definitely re-train the Koopman. Train a good one even if it takes longer, use this PC's hardware to do a good train. You should collect new data using the digital twin. You should set the data collection such that all modalities of the arm is collected. You should consider the sim-to-real problem when we try out on the new arm next step. We should mostly train the koopman using simulated data, but then we should also be able to close the sim to real gap through possibly an extra round of training on top of the exisiting weight, if that is a good method. You may also consider any other methods that works.

## Agentic workflow

This is a big task. If the main agent do all the task from begin to finish, the context window should not fit. Instead, we should let the main agent be the supervisor and orchestrator. for every stage of the experiment, it should spawn a maximum of 3 subagent workflow to complete task and report back.

You should not ask questions as I am not at the PC; Commit along the way, push after you are done.
