# Handed Humanoid Cricket

An optional integration example for the corrected humanoid batting and running-bowling
drills in [Gym-Cricket](https://github.com/kishanpb/gym-cricket/tree/2a6641ddc030407b10e2320f07d6b88a92e23072).
The model, controller, and adapter are pinned to that MIT-licensed revision; this example
does not change mjbatch's native API, runtime dependencies, or default examples group.

## Reproduce

Python 3.13 is required for the pinned checkpoint runtime, not for mjbatch itself.
Clone the public checkpoint source and check out the exact revision:

```sh
git clone https://github.com/kishanpb/gym-cricket.git ../gym-cricket-models
git -C ../gym-cricket-models checkout 2a6641ddc030407b10e2320f07d6b88a92e23072
uv run --with-editable . examples/cricket_humanoid.py \
  --checkpoints ../gym-cricket-models/examples/pretrained \
  --output ../cricket-humanoid-results
```

The script's inline dependencies isolate JAX/SKRL from the normal mjbatch environment.
`--with-editable .` uses this mjbatch checkout, not an unrelated installed wheel.
Checkpoint deserialization is intended only for the trusted, pinned files above;
do not substitute untrusted pickle files.

Add `--video` to generate both 1280 x 720 reels, contact/release sheets, and a JSON
report containing every evaluated row, source/checkpoint hashes, native integration
counts, frame ranges, and decode checks. Rendering needs a working MuJoCo OpenGL
backend; omit the flag for headless validation.

## Physics and Policy Scope

- Each `HumanoidBatch` contains four independent humanoid-and-ball simulations of
  one task and handedness. Its integration boundary calls `mjbatch.Batch.step`.
- Python retains motion generation, motor targets, aerodynamic-force setup,
  stabilization, policy inference, and episode bookkeeping. Model body poses and
  adaptive timesteps are expanded per row; finished rows stop independently.
- Batting uses seven PPO/A2C motor residuals around a scripted stance and swing.
  Corrected forearms, elbow direction, forward torso bend, and two-hand grip are
  retained. Ball contact is force-bearing native contact, not an injected hit impulse.
- Bowling uses PPO/A2C release speed, angle, and line with a scripted running action.
  Release is hand-attached followed by native free flight. Both handed models are
  simulated; the video is not flipped. The bowling checkpoint is shared across hands.
- RK4 contact sensors and the legacy last-internal-stage contact buffer have different
  timing. The serial reference uses the same sensor-based detector as the batch.
  These are fresh checkpoint-transfer results, not relabelled legacy scores.

This correctness-oriented adapter still has substantial per-step Python work. It is
not a fully vectorized training environment, a throughput demonstration, a full match,
or a validated human-biomechanics model. No speedup or algorithm-ranking claim is made.

## Fixed Evaluation

Four fixed seeds per algorithm and hand, with no outcome filtering: batting
3101-3104 on the wicket profile; bowling 4101-4104. Total: 16 episodes per task.
The videos show the first and fourth seeds, including misses, with 0.25x replays.

| Task | Hand | PPO | A2C |
| --- | --- | --- | --- |
| Batting native contacts | Right | 4/4 | 3/4 |
| Batting native contacts | Left | 3/4 | 1/4 |
| Bowling mean target error | Right | 0.4391 m | 0.1580 m |
| Bowling mean target error | Left | 0.4391 m | 0.1580 m |
| Bowling native wicket contacts | Right | 1/4 | 4/4 |
| Bowling native wicket contacts | Left | 1/4 | 4/4 |

This small checkpoint-transfer fixture is not a policy tournament or a training-seed
comparison. Reflected bowling rows are not independent training replications.

## Validation

The replay checks exact qpos, qvel, observations, rewards, and stop flags against
serial MuJoCo at every control step. It also checks batting contacts, runs, grip and
elbow geometry, and bowling outcomes, hand release gap, and delivery foot faults.
All 32 episodes completed with 95,746 native row-steps. No bowling foot faults were
observed; maximum batting grip error was 0.00008194 m.

Optional integration tests cover both hands and tasks, including a monkeypatched
Python `mj_step` that raises if the batched path uses it. To run these with the core
suite in a separate environment:

```sh
uv venv --python 3.13 .venv-cricket
uv pip install --python .venv-cricket/bin/python -e . pytest \
  'cricket-gym[video,telemetry] @ git+https://github.com/kishanpb/gym-cricket.git@2a6641ddc030407b10e2320f07d6b88a92e23072' \
  gymnasium==1.3.0 jax==0.10.0 jaxlib==0.10.0 skrl==2.0.0 \
  numpy==2.4.4 scipy==1.17.1 Pillow==12.2.0 flax==0.12.5
.venv-cricket/bin/python -m pytest tests
```

Without Gym-Cricket installed, only this optional test module is skipped. The pinned
Gym-Cricket revision also carries contact-stage, limb-length, continuity, reset,
adaptive-step, and packaging tests. Local validation is on macOS/Apple M2 Pro;
Linux rendering and large-batch performance have not been established.

## Contact Forces and Simulated Touch

Add `--contact-telemetry` to the replay command to record ball contact with the bat,
pitch, outfield and stumps. The [complete 32-episode report](cricket_contact_results/mjbatch_humanoid_validation.json)
retains all fixed seeds, both hands and both algorithms, including misses. Every
control step checks exact serial/batch telemetry, state, reward and termination parity.
The source and checkpoint hashes are included; no checkpoint was retrained.

| Integrated-stage diagnostics | Batting | Bowling |
| --- | ---: | ---: |
| Complete episodes | 16 | 16 |
| Episodes with load-bearing samples | 16 | 16 |
| Active substep samples | 3,529 | 28 |
| Largest channel normal-load sample | 175,797.74 N | 1,127.18 N |

These counts include all tracked surfaces, not just bat hits or wickets. Ten bowling
episodes also contain a load-bearing **terminal forward solve**, counted separately.
The largest batting load is the right-handed PPO seed 3101 bat-blade sample at a
2.5 microsecond timestep. This force spike is retained as an uncalibrated model
limitation, not advertised as realistic cricket impact. Dynamic robot control and
contact-parameter/timestep checks are required before making physical-load claims.

Local validation passed 90 shared-package tests, 44 mjbatch tests, and this full
32-episode replay (95,746 native row-steps). The replay was rerun using the public
Git-pinned package and this fork's editable native mjbatch build.

Each channel reports load-bearing touch, contact count, normal/shear load and peaks
in N. Sparse substep records retain contact-frame force/torque, world position and
frame vectors, distance and timestep. Terminal reports include the full active trace.
These are simulated geometry-level touch states, not hardware taxels or pressure
measurements; grip load and dynamic foot support are unavailable in this model.
RK4 sensor samples are not integrated impulses. Bowling's terminal pose/forward
solve is reported separately and never counted as an integrated impact.

See the [signal contract and support-load fixture](https://github.com/kishanpb/gym-cricket/blob/2a6641ddc030407b10e2320f07d6b88a92e23072/docs/contact_telemetry.md).
The option is off by default and does not change observations, rewards or existing
checkpoint dimensions. The existing videos below predate these telemetry reports.
Unitree G1 learned batting/bowling is the next extension, not a result of this replay.

## Videos

[Batting reel](https://github.com/kishanpb/gym-cricket/releases/download/v0.1.1-mjbatch-preview/gym_cricket_mjbatch_humanoid_batting.mp4)
and [running-bowling reel](https://github.com/kishanpb/gym-cricket/releases/download/v0.1.1-mjbatch-preview/gym_cricket_mjbatch_humanoid_bowling.mp4).

![Both handed batting variants and algorithms](assets/cricket_humanoid_batting.png)

![Both handed bowling variants and algorithms](assets/cricket_humanoid_bowling.png)
