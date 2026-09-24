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
The shared UniLab task now trains separate right/left PPO actors on native
CPU `Batch.step()`, with two mechanical hand grips and all 29 joints following
a cricket swing reference. This replaces the isolated-arm/frozen-walking-prior
direction; it is not an independent native learner or a new Menagerie-model
result. The [latest two-hand PPO diagnostic](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/812cc13d5b948aa73faf8db41c1e12ad14e2a444/g1_cricket_results/bimanual_balanced_small_residual_v1/two_hand_ppo_diagnostic.mp4)
completes both three-second swings with ankle feedback and smaller learned
corrections. Small joint-stop excursions and bat-path error still fail the
physical/accuracy gates; ball hitting and running bowling remain unfinished.
The linked research log retains complete episodes and substep force, grip,
joint-limit and collision audits, not just selected successful-looking frames.

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
