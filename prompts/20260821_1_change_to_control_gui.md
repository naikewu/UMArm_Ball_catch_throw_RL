# Change control gui, and verify motion capture

Headline: I enabled the arm, supplied the power, and supplied the pressure. In the following task, if I told you to use hardware to test, then you have full access to use the hardware. Do not avoid using hardware. Using hardware and get feedback data is a very efficient way to complete our task.

The pressure source is tuned down to 20 psi (roughly, do not worry about exact number). the arm is also placed in a safe, monitored area. The general rule of using the arm is that you should not put too much antagonistic pressure on the arm. avoid sum of actuator pressure exceeding 30 psi FOR A SINGLE PAIR ACTUATOR controlling one revolute joint.

## First task, verify the room-view motion capture.

This folder is created by a coding agent task in a previous folder. The previous prompt is located in:

(C:\ESP\ESP_Projects\VEMA_MAX22200\TLE92464_extension\TLE_prompts\20260820_TLE_controller_improvement.md)

It also referenced folders such as:

(C:\RUNZE_SRC\RS485_VEMA) for kinova arm, mocap, etc.
(C:\ESP\ESP_Projects\VNEMA_MK8_PIDPWM) for 7mm valve PCB firmware, and
(C:\ESP\ESP_Projects\VEMA_MAX22200) for TLE PCB firmware.

Due to my mistake, I did not enable mocap streaming when running the task. as a result, the mocap part of the task was not verified.

In previous RS485 arm task, we implemented a motion capture calibration procedure. It allow us to not only use the mocap rigibody frame data, but also use the rigidbody individual marker data.

This robot is kinematically same as the RS485 arm, so the mocap calibration should follow similar procedure.

For your reference, 
(C:\Users\zuorunze\OneDrive - Umich\PHD_Courses\Research\project2023_variable_stiffness\UMARM_Variable_Stiffness_Oct2025)

is the legacy folder that contains the mocap + forward/inverse kinematics code for the larger CAN bus arm we are working on. The kinematic still stays the same, but the actuator ID for the top 8 actuators are different as we changed the top regulator platform to a new PCB and new proportional valve. The mocap to config mapping should still work, but the arm streaming ID has changed. the current streaming ID is 2000-2005, from top to bottom.

Each of the mocap rigid body has four markers. Each rigidbody is the bracket connecting one side of the ujoint. Each segment is like: upper Bracket -> upper ujoint -> link -> lower ujoint -> lower bracket. actuators are connecting between brackets and links, across the ujoint so they move the ujoint. The four markers are on the bracket. if you connect these four points into a rectangle, the intersection of the two diagonals is exactly the center of the bracket, AND is the center of the ujoint. We utilize that information to compute the robot joint angle. the diagonals should be 45 degree offset wrt the vertical z axis from the revolute joint. i.e. the markers arm are poking out between two actuator attachment point on the bracket.

However, at MOTIVE software side, I can also set the rigidbody orientation, which get streamed as well. For that, I have to manually align the frame using their software, which will always induce error. So what I want you to do is to infer the rigidbody FRAME from the four markers. Note that 3 markers per rigidbody is enough to do this, but four is to protect us from dropping one marker. We should define the rigidbody frame as follows:

Take segment 1 as an example, we imagine the plane created by the four markers as the X-Y plane, the z-axis is normal to xy plane, the origin is the center of the diagonal. The diagonal is 45 degree offset. define the rotated(45 deg) as x-axis, make the robot x-axis close to the mocap world spatial x-axis. Then, the y-axis should be solvable by right hand rule. The robot z-axis is also pointing upward at initial pose. So now you have the robot x-y-z axis, we also know that the first revolute joint of the segment is along the robot positive x-axis. the second revolute joint along the robot positive y-axis. The third and fourth is lower at the second ujoint after the link. the lower bracket is designed similarily to the upper bracket, but it is rotated 45 deg CCW. So will the revolute joint also rotate 45 deg CCW wrt the segment z axis at default 0 position.

Using the information above, I believe we can create our own inferred rigidbody frame by using the streamed rigidbody marker positions. This inferred rigidbody should be robust when one marker is dropped.

In the old folder (UMARM_Variable_Stiffness_Oct2025), there's a forward kinematics code. and also in the viusalizer, we plot the mocap and the fkine together. I want you to actuate the robot, and see if you are able to make the mocap and the fkine match up. i.e. use fkine and joint angle acquired from mocap to compute the ujoint center position, and compare it with real mocap rigidbody center position. They should be able to match up.

If not, you need to figure out, if it is a parameter issue, or the fkine need some touch up? Anyway, do what you need to get us an accurate enough forward kinematics function. You need to report to me what you have imporved and what is the quantitative result.

## Permission to drive the arm

You have full permission to drive the robot arm. everything is connected.

## Summary

Many of the prompt is copied from (C:\RUNZE_SRC\RS485_VEMA\prompts\08112026_further_benchmarking.md) We are basically doing the same work, but just on a different arm.

Your task: 

1. verify actuator - axis mapping
2. develop new mocap frame inference methods.

You are required to drive the robot to use real data to do these.

Make sure that you connect to motion capture, and actually drive the arm.
do not ask questions, commit and push after you are done.