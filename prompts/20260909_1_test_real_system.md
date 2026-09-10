# Test real system, collect real system data, and train digital twin

## Background

In previous task, we transported the firmware, software, mocap, etc. all the stuff from different folder to this folder, and we are going to start a project on controlling a big version of the McKibben actuator driven hyperredundant arm, or UMArm (the promax version UMArm).

In this task, we want to try to see if mocap & arm works, and try to collect some first data for the digital twin, and set up a process of updating the digital twin using the real data constantly.

## Related Resources

(C:\RUNZE_SRC\RS485_VEMA) This git repo, and its worktree, contains work we've done using the original UMArm (current arm is UMArm ProMax) it:

1. have a digital twin that trained on real system. The digital twin runs on MuJoCo, and have 'actuator net' as one of the system identification method
2. The digital twin fully respect the communication bandwidth, valve flow rate, etc. It measure the real data and try to be as realistic as possible
3. In different worktrees, we tried different controllers, like koopman MPPI, etc.
4. In one of the worktrees, we tried collision between the kinova arm and the UMArm. It contains workflow of using kinova arm, the ATI nano 17 force gauage, the UMArm, and the Mocap at the same time.

(C:\RUNZE_SRC\VEMA_MAX22200) This git repo contains the PCB design, firmwares, and software, for the TLE PCB.


## System status

Current system has 24 actuators, kinematically similar to the UMArm. The top segment 8 actuators are using Clippard DVP proportional valves. Currently, the outlet of the proportional valves are connected to a venturi vacuum. the inlet is connected to ~40 psi. The TLE PCB is controlling the proportional valves. The other 2 segments are using old PCB, which they even don't have standalone ADC for reading pressures. They are using ESP32 internal ADC to do so. I think the current TLE_PCB folder VEMA_TLE_controller.py already can control the pressure target of all the actuators.

In my understanding, all the actuators should be sync at 150Hz as of now. All of them will receive the next pressure target, and activate them at the same time according to a CAN bus sync signal. they will also feedback pressure at the sync signal, not at the actual momemnt it send the feedback.

MOCAP is all connected with ID range from 2000-2005. The old original UMArm is also in the space, you should be able to see it.

The arm is pressurized and powered, ready to go. One of the actuator has a leaking inlet, which will slowly let pressure inside when turned off.

## Task

Your task is to:

1. Verify if the UMArm promax and the Mocap can be seen by you.
2. Go through the current code base, compare the current digital twin implementation with the RS485 repo implementation. the RS485 repo collision branch contains the newest protocol of how to let simulation behaves similarily with the real system. You should implement that method here.
3. Run data collection using the real arm. Note: you should not actuate the arm for antagonistic pressure pair A,B, such that pressure A+B > 30. the max pressure of any pressure should be capped at 30psi.
4. A camera is connected to the PC via a USB collection card. The camera is pointing at the arm. You should use the camera to shoot videos of the arm collecting data. Whole process is not needed since that would be too big of a data, you should collect videos once in a while, at a reasonable data size.
5. Your data collection should let us be able to use the data to train the actuator net that is implemented in the previous step. Plan your collection sequence carefully to let us get a good result.
6. Do a full-out train using the collected data, to shorten the sim to real gap. fully utilize the computational resource on board on this PC.
7. You should deliver a video, where the simulated twin run side by side with the real collected data arm, using the same input sequence, so that I can see if the trained twin behaves similarily with the real arm.
8. You should capture and flag anomallies when running. if you think you actuated a pressure but the joint is not moving as expected, you should use the camera to take a clip, and flag it to me at the end.

## Workflow

Use a small workflow of up to 5 agents. Commit on the way, and push after you are done. Do not ask questions, I give you full permission to use the hardware, I will be not happy if you not use the hardware when things are OK to run.