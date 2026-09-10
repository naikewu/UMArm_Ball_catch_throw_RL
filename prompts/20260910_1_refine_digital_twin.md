# Refine digital twin

## Background

In the previous task we collected real data on the UMArm ProMax and tried to build a digital twin of the arm.

## Robot visualization

Headline: the mujoco visualizer looks wrong in the provided deliverable video in the previous run.

Why it is wrong:

1. The ProMax UMArm has a 'Y' structure on the center rod, between the two ujoints, inside one segment. The actautor one end would connect to the upper arm of the Y, and the other end route toward the lower foot of the Y, around a bearing, and connect to a ujoint bracket ceneterd near the lower foot of the Y, to control the rotational axis at that ujoint. The cone shaped cavity of the Y, will have the upper ujoint inside, which is controlled by the lower actuators which attached to an upside down Y. There are in total of 8 actuator per segment, 2 upright Y placed in normal to each other, and 2 upside down Y, have 45 deg offset wrt the upper 2 Y bracket. one ujoint each, two total, will be in the cavity of the upright and the upside down Y strcuture. the ujoint can be simply plot by a flat cylindrical disk, the Ys should all be on the center rod rigidbody. I downloaded a paper: (2606.29731v1.pdf) which is the paper for my exact arm. that would help you understand the Y, actuator, tendon, bearing, ujoint arrangement.

Currently in the video, the Y arm looks connected to the Ujoint, which is not correct. In this run, you will look at the figure and discription in that paper, and revise the arm simulation.

(C:\RUNZE_SRC\UMArm_dynamic_koopman_compliance\koopman_ckpt_continuous_latest_z160_compliance_jointspace_v3.pt) contains the MuJoCo that reflects the UMArm Promax.

## Simulator tuning

I see in the deliverable video that the two robot doesn't move alike. Give me a bulleted list of what have you done to let the two fit using collected data. Did you tune the segment weight? It is kinda hard for me to provide the weight as of now, you might need to run an optimizer to figure out.

## Same gui control both real and sim

The canarm_control_gui.py should be able to start a simulated arm as well. Sharing the same interface. The controller in general shouldn't know if it is controlling the real or the sim.


## Workflow

Use a small workflow of 5 agent max. Do not ask questions, commit on the way and push after done.