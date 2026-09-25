# mjbatch

## Humanoid Cricket by Kishan

This public fork is maintained by [Kishan (@kishanpb)](https://github.com/kishanpb)
for the [Gym-Cricket](https://github.com/kishanpb/gym-cricket) integration proposed
in [upstream PR #5](https://github.com/kevinzakka/mjbatch/pull/5).
The contribution is included on this fork's default branch; upstream acceptance
is separate. mjbatch is developed upstream by Kevin Zakka and contributors.

| Batting: right and left handed | Bowling: right and left handed |
| --- | --- |
| [![Native mjbatch humanoid batting highlights](examples/assets/cricket_humanoid_batting_preview.gif)](https://github.com/kishanpb/gym-cricket/releases/download/v0.1.1-mjbatch-preview/gym_cricket_mjbatch_humanoid_batting.mp4) | [![Native mjbatch humanoid running-bowling highlights](examples/assets/cricket_humanoid_bowling_preview.gif)](https://github.com/kishanpb/gym-cricket/releases/download/v0.1.1-mjbatch-preview/gym_cricket_mjbatch_humanoid_bowling.mp4) |

Click either preview for the full 42-second PPO/A2C reel with both hands,
misses, and 0.25x replays. These are retained-checkpoint transfers through native
CPU-batched integration, not new training or a speedup result. Stance, swing
reference, and running action retain scripted components.

The reproducible fixture contains 32 fixed episodes with exact serial-model parity;
the previews are highlights, not success-rate estimates.
[Run the example and inspect all results](examples/cricket_humanoid.md) |
[Source and checkpoints](https://github.com/kishanpb/gym-cricket/tree/2a6641ddc030407b10e2320f07d6b88a92e23072) |
[Related UniLab fork](https://github.com/kishanpb/Cricket-Gym-Unilab)

**Contact diagnostics:** [force and simulated-touch reporting](examples/cricket_humanoid.md#contact-forces-and-simulated-touch)
now covers the full fixed PPO/A2C replay, with per-contact force/torque, touch states
and explicit sensor-timing limits. This does not turn the scripted components into
learned robot control; a Unitree G1 cricket extension is the next milestone.

**Unitree G1 research:** [robot setup and complete results](examples/cricket_g1.md).
The [reference-closure comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/df9bedba4e7489a12dc3c17765c8befaa2a291c4/docs/g1_cricket_projected_batting.md)
retains all 16 physical outcomes: exact grip/foot reference closure separates
inverse constraint loads but does not fix the remaining swing-tracking error.
The [bounded motor-inertia experiment](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/c217748d77b46bf1d07d3fe1e8dc957620b15b62/docs/g1_cricket_inertial_feedforward.md#complete-results)
reduces finest PPO error to 9.72 cm right / 8.69 cm left while preserving
contact/stability checks across all 16 trials. The 8 cm limit still fails;
compensation remains opt-in, with frozen actors rather than new training.
The shared UniLab task now trains separate right/left PPO actors on native
CPU `Batch.step()`, with two mechanical hand grips and all 29 joints following
a cricket swing reference. This replaces the isolated-arm/frozen-walking-prior
direction; it is not an independent native learner or a new Menagerie-model
result. The [new ball/contact-observed PPO video](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/3ecd65db68b1fe16f4921fe97a5d8b46156d46f5/g1_cricket_results/bimanual_batting_learning_v1/two_hand_ppo_learned_batting.mp4)
shows fresh right/left actors making two-handed swings, hitting a one-bounce
practice feed and recovering upright at 0.5x. The
[complete learning pilot](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/3ecd65db68b1fe16f4921fe97a5d8b46156d46f5/docs/g1_cricket_batting_learning.md#complete-results)
retains both final checkpoints and all eight evaluations: contact/stability
checks pass, but both actors fail the 8 cm bat-path limit. Reference-only
control also hits; this is not robust learned interception or an independent
Menagerie result. The [earlier frozen-actor video](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/579221676ac5165285fc68cf90197875f45d965a/g1_cricket_results/bimanual_bounced_delivery_v1/two_hand_ppo_bounced_delivery.mp4)
shows a one-bounce incoming delivery, two-handed hit and upright recovery in
both stances at 0.5x. Frozen dry-swing actors have no ball observation; this is
not learned interception. The [complete eight-episode study](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/579221676ac5165285fc68cf90197875f45d965a/docs/g1_cricket_bounced_delivery.md)
retains both reference and PPO outcomes: all finish and hit, but bat-path
accuracy still fails. A [third physics resolution](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/47583d1979c071c8071e9755587b9290780083cb/docs/g1_cricket_bounced_delivery.md#refinement-results)
passes the finer contact comparisons while retaining the original coarse-grid
failures; this does not qualify the final showcase. The
[motor-timing comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/0f46074160e9848a804c706d7dd79b8b8faf0baf/docs/g1_cricket_motor_lead.md#complete-results)
reduces bat-path error but still fails the unchanged accuracy gate across
both hands, retaining all 24 episodes and contact checks. The earlier
[soft-toss study](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/d88ae36ec444af5fbb18b100fff1a700133f944f/docs/g1_cricket_bimanual_contact.md)
is preserved. Neither input is regulation-speed bowling or a final showcase.
The [closed-loop PPO follow-up](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7d00b0dcea4850cbab3ea4d1e3e271d87684ded6/docs/g1_cricket_approach_feedback.md#complete-results)
keeps the external locomotion prior's live feedback. Both local whole-body
residual actors complete eight-second approaches and stop upright at two
physics resolutions, travelling 2.78-2.81 m. Full videos and all eight outcomes
are retained; slip/drift checks still fail. These are shared UniLab G1 walking
diagnostics on native Batch, not independent Menagerie training or bowling.
The [frozen-policy lane trial](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/924d9d838d8ce2bd62676f92421d073559c09f4c/docs/g1_cricket_approach_lane.md#complete-results)
reduces right/left drift from about 17/27 cm to 10/15 cm, but left still fails
the lane limit and touchdown slip remains unresolved. Full outcomes and videos
include the finer-grid right stance-slip regression; no bowling promotion.
The matched [peak-slip reward pilot](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/294373e629722f5fc37414c56531443e2efa0ab2/docs/g1_cricket_approach_peak_slip.md#complete-results)
retains fresh right/left PPO runs and all eight outcomes, but fails the same
physical gates. It is rejected as a reward-only repair, not advertised as progress
in bowling quality.
The earlier [measured-command PPO follow-up](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/aa88ad7eb09379495c01f6910f6dcd790fd82206/docs/g1_cricket_measured_approach.md#complete-ppo-results)
trains both 29-joint actors for 49,152 transitions each on native Batch.
All eight outcomes are retained: PPO loses balance at 2.76 s right and 2.18 s
left, while fixed reference commands are timestep-sensitive. Both full failed
videos remain diagnostics, not bowling highlights or independent Menagerie training.
The [measured approach study and both-hand videos](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/874af274779349c2f5007c8ee9a0546f9b0996a8/docs/g1_cricket_approach_teacher.md#complete-results)
retain eight complete external-prior trials: all travel 2.82-2.85 m and stop
upright, but all fail foot-slip checks and some drift sideways. These are
shared UniLab G1 carry diagnostics on native mjbatch, not locally learned
running deliveries or independently trained Menagerie policies.
**Running bowling is unfinished.** The [continuous moving-delivery prototype and full videos](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/e01b3eee57b13e3f62a21f66d2badf910ee7ce49/docs/g1_cricket_moving_delivery.md#arm-servo-follow-up)
reach overarm releases in both hands with live body feedback, and right-hand
recovery stays upright at both timesteps. Backward ball velocity, left-foot
wicket contact and right joint-limit violations still block qualification.
All eight outcomes are retained: shared UniLab G1 on native CPU Batch,
not an independently trained Menagerie policy or a finished bowling highlight.
Earlier both-hand references preserve the approach,
gather, delivery and recovery, but physical controllers and the PPO pilot fall
before release. The latest [planted-foot reference and joint-tracking comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/fcf57f917d5d696a08d5e0d0f9efb6c21a576497/docs/g1_cricket_ground_momentum.md#foreaft-pendulum-comparison)
removes audited reference self-intersections and passes sampled support bounds,
but every physical trial still falls before release. Stronger tracking gains
worsen joint-limit violations; transition into running and landing dynamics remain unresolved.
The new [two-foot startup test](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/72004f4e31af68458aec131ea2fcfe070b073103/docs/g1_cricket_running_startup.md)
completes settling and an 80 mm weight shift in both hands at two physics
timesteps, without joint-stop or unintended-contact violations. It verifies
only the from-rest startup stage, not foot lift, strides or a full delivery.
The subsequent [first-step comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/274c562adc699e920563b329f26eabaa770b50d5/docs/g1_cricket_running_startup.md#first-step-results)
achieves actual foot lift and landing with both PD variants, but all 12 trials
fall before settling. Repeated strides and the complete delivery remain unfinished.
The [whole-body first-step PPO pilot](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/e22b1d05cb8be969b500bcd3fca81a1b3b70df9d/docs/g1_cricket_first_step_learning.md)
retains both independently trained final actors and all eight evaluations:
right PPO delays instability, left PPO regresses, and neither passes the step gate.
The [uniform-start comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/1d580e72a5e6ac02d98798f0c0011cbcb134adcb/docs/g1_cricket_first_step_uniform.md)
improves left-hand landing but regresses the right hand; all eight from-rest
evaluations still fail the full step gate.
The [world-frame foot-reward trial](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/2dc8259c40c36082390ac35d71d16feec6c889ad/docs/g1_cricket_first_step_foot_reward.md)
regresses both hands and is not promoted; all eight outcomes remain available.
An [opt-in recorder optimization](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/14be04674e2675b068fd0f58501832c90bc22884/docs/g1_cricket_cpu_grouping.md)
preserves exact G1 replay and measures 1.49-2.08x local recorder speedup,
not an end-to-end training or policy-quality gain.
Its [compact-recording extension](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/3ca859631b52fc9fa60dcbd9a2b2ae425cdf398b/docs/g1_cricket_cpu_grouping.md#compact-sensor-recording)
reduces retained sensor arrays and measures a further 1.73-4.09x local recorder
speedup without changing physical sensors or policy outcomes.
This is shared UniLab G1 work, not a separately trained Menagerie-model result
or bowling showcase. The [research log](examples/cricket_g1.md) retains complete
episodes, controls, failed videos and substep force/contact audits.

Earlier [force/touch diagnostics](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7f936c78b9e0d882087be6deedadba4525bd7224/g1_cricket_results/bc_v1/learned_development_diagnostic.mp4)
and complete failed experiments remain in the research log. Simulated loads
are uncalibrated, mechanical grips are not learned finger grasping, and no G1
advertising reel is claimed. The original highlights above are unchanged.

---

[![Build](https://img.shields.io/github/actions/workflow/status/kevinzakka/mjbatch/ci.yml?branch=main)](https://github.com/kevinzakka/mjbatch/actions)
[![PyPI version](https://img.shields.io/pypi/v/mjbatch)](https://pypi.org/project/mjbatch/)

`mjbatch` is a Python library for running thousands of MuJoCo simulations in parallel on CPU.

Features include:

* C++ thread pool execution, with the GIL released;
* Live array access to simulation state and controls across the batch, with `bind` for MjData fields;
* Per-simulation model parameters, with `expand` for MjModel fields and `set_const` to recompute derived constants.

For example:

```python
import mujoco, numpy as np
from mjbatch import Batch

model = mujoco.MjModel.from_xml_path("scene.xml")
batch = Batch(model, num_sims=4096)  # threads default to every logical CPU
qpos, ctrl = batch.bind("qpos"), batch.bind("ctrl")
batch.expand("geom_friction")[:, :, 0] = np.random.uniform(0.4, 1.2, (4096, 1))
for _ in range(1000):
  ctrl[:] = policy(qpos)             # your controller, all 4096 at once
  batch.step()                       # step them in parallel; qpos updates in place
```

## Examples

We showcase a range of applications built using `mjbatch`: RL, MPC, SysID, and hardware
co-design. Each example is a self-contained, performant implementation. For instance, the Go1
RL controller learns to walk in under a minute on a five-year-old M1 laptop.

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/cartpole_swingup.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/cartpole_swingup.gif" alt="cart-pole swing-up"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/cartpole_mpc.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/cartpole_mpc.gif" alt="cart-pole MPC"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A two-pole cart swung upright with <a href="https://ieeexplore.ieee.org/document/6386025">iLQR</a></td>
    <td align="center">A cart-pole swing-up controller using <a href="https://arxiv.org/abs/2212.00541">predictive sampling</a></td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/g1_flip.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/g1_flip.gif" alt="G1 backflip"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/go1_joystick.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/go1_joystick.gif" alt="Go1 joystick"></a>
    </td>
  </tr>
  <tr>
    <td align="center">A G1 humanoid tracking a reference backflip with receding-horizon iLQR</td>
    <td align="center">A Go1 quadruped joystick controller trained with PPO</td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/arm_throw.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/arm_throw.gif" alt="throwing arm co-design"></a>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/kevinzakka/mjbatch/blob/main/examples/rizon_inertia.py"><img width="400" src="https://raw.githubusercontent.com/kevinzakka/mjbatch/main/examples/assets/rizon_inertia.gif" alt="Rizon inertia identification"></a>
    </td>
  </tr>
  <tr>
    <td align="center">CEM jointly optimizes a robot arm's proportions, gears, and controls</td>
    <td align="center">Damped Gauss–Newton fits a Rizon arm's inertial parameters to synthetic motion data</td>
  </tr>
</table>

Run with `uv run examples/<file>.py`; some need `uv sync --group examples`. The ones that open
a window need a display; `--headless` runs the solver without one.

### Humanoid Cricket Integration

[`cricket_humanoid.py`](examples/cricket_humanoid.py) replays retained PPO/A2C
checkpoints for right- and left-handed batting and running-bowling drills through
native CPU-batched integration. It checks serial parity and preserves scripted
motion and learned-control labels; it is not a throughput or new-training result.
See the [pinned setup, videos, model limits, and tests](examples/cricket_humanoid.md).

## License

Apache-2.0.
