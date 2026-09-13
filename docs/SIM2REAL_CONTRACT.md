# OpenArm v1 simulation-to-real contract

This scene uses the official v1 bimanual asset at commit
161039cd74ea8675fb8197836fe5674659825c75. Right-arm insertion and a parked left arm
match the requested embodiment. No physical deployment has been validated.

## What is fixed for pilot data

- Both seven-joint arms, both grippers, upstream collision geometry and joint limits.
- Eight policy commands: absolute right joint1..7 targets in radians, followed
  by symmetric right-finger slide travel in metres (0..0.044 per finger).
  This last field is NOT total jaw aperture or a normalized open/close value.
- Sixteen proprioceptive values: left joint1..7 and mean finger travel, then right
  joint1..7 and mean finger travel. The parked arm remains observed.
- Command rate 50 Hz, physics 2 kHz. Rate limits apply before commands enter the
  simulated motor controller. RGB frames are captured at the command boundary.
- The plug is a free body with physical finger contacts. There is no plug weld,
  externally applied holder force, or direct plug gravity compensation.
- Joint PD plus model bias feedforward generates motor torque, limited by the
  upstream actuator force ranges. This is an explicit simulated controller,
  not a claim to reproduce the hardware firmware.

Upstream v1 uses torque motors on arm joints and the left fingers, and position
servos on the right fingers. The adapter handles this asymmetry without replacing
the robot geometry or silently treating raw torque as a position command.

## Things requiring measurement before transfer

| Quantity | Current simulation assumption | Real-world check |
| --- | --- | --- |
| Controller | Configured PD, bias feedforward, ideal timing | Gains, friction, payload compensation, rate, delay, torque/current mapping |
| Gripper | Ideal symmetric slides, contact preload | Travel-to-aperture mapping, finger compliance, pad friction, slip under insertion load |
| Connector | Design dimensions, equal blades, no cable/springs | Actual slot/blade geometry, polarization, lead-in, axial force and cable loads |
| Cameras | Illustrative fixed/wrist mounts, ideal RGB | Intrinsics, distortion, exposure, crop, extrinsics, time offsets |
| Fixture | Rigid mounted socket | Mount pose/compliance and reachable approach envelope |
| Observations | Synchronous, noiseless proprioception | Sensor resolution, noise, missing/stale data and timestamps |
| Reward/termination | Geometry oracle in simulation | Measurable seating, contact and intervention criteria |

Camera metadata records parent body, local position, quaternion and vertical FOV.
For the current undistorted rendering, derive square-pixel focal length as
height / (2*tan(fovy/2)); principal point is the image center. Real camera
calibration should replace this model rather than assume equality.

The wrist camera is aimed at the approach region. It is a proposed mounting
location, not a verified CAD mount; gripper occlusion is present in the renders.

## Contact choices and limitations

Finger torsional friction is enabled with condim=4. The robot scene uses an
elliptic cone, impratio=100, and fingertip solref=0.002 s. These settings reduced
numerical grasp creep during diagnosis. They are not measured material properties.
The holder baseline retains its earlier solver settings for comparisons.
See PHYSICS_CHANGELOG.md for the sequence and measured effects.

A 1 mm grasp-drift limit aborts an episode. Validation reports the actual maximum,
rather than claiming perfectly rigid or zero-slip contact. The initial reset
places an already-held plug near finger contact and lets the grasp settle before
recording; reset time is excluded from episode duration and must not be confused
with real acquisition or pickup capability.

The expert uses simulator plug/socket pose and contact diagnostics to align and
recover. These are labels/control assistance for the demonstrator, not inputs to
the VLA. The learning dataset adapter exposes only RGB, joint state and language.

## Transfer sequence

1. Reproduce robot action replay with measured controller and camera timing.
2. Measure static grasps, slip, connector force/depth and geometry on an unpowered fixture.
3. Update parameters one group at a time and retain a held-out simulation benchmark.
4. Collect a small hardware demonstration/evaluation set; compare error sources.
5. Fine-tune/evaluate the supervised policy before any hardware online RL.
6. Validate interruption, withdrawal and grasp-loss detection on hardware.

Domain randomization is deferred until ranges can be justified. The current pilot
uses only small initial Y/Z offsets; it is not a sim-to-real generalization test.

RLT must use the same action adapter and timing contract. Future replay records
must retain executed chunk duration and termination masks. A cached RL token must
be associated with the exact frozen checkpoint/adapter that produced it.

