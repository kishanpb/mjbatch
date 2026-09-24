# Unitree G1 Cricket Development

## Current Direction: Two-Hand Whole-Body Tracking

The companion [motion-tracking task and complete pilots](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/381d7b4bccd65138a634cc48c7611d5bc8d24b7d/docs/g1_cricket_bimanual_tracking.md)
retargets the earlier cricket choreography to UniLab's G1 geometry. Native
mjbatch trains independent right/left whole-body PPO actors with two mechanical
bat grips and all 29 joint controls. It does not use the frozen walking policy
or overwrite robot poses during a policy step. This is the shared UniLab task
on this executor, not a separate learner or the legacy Menagerie scene below.

The initial 196,608-transition pilot per hand fails during downswing at 1.56
and 1.68 seconds, respectively; reference-only control fails at 0.32 seconds.
These are deterministic dry-swing diagnostics, not ball hits or held-out
performance. An equal, predeclared continuation reaches 393,216 cumulative
transitions per hand: final duration improves to 2.00 seconds (right) and
2.64 seconds (left), but both still fail anchor height before completing the
three-second motion. Complete traces, final checkpoints and failed diagnostic
videos are retained in the linked report, not advertised as successful cricket.
Running approach, gather, legal plant, overarm release and
recovery still need whole-body retargeting and physical learning before a new
bowling showcase. The previous scripted-component highlights remain unchanged.

## Historical Single-Wrist Foundation

This is a **floating-base robot-learning foundation**, not a finished cricket
policy or replacement for the existing humanoid highlights. The native mjbatch
environment preserves the Menagerie G1's 29 joint actuators, link inertias,
joint limits and actuator-force limits. There is no root support constraint,
gravity compensation or step-time pose overwrite.

![Untrained right-hand reset geometry](cricket_g1_results/ready_pose.png)

The pinned `unitree_g1/g1_mjx` asset is BSD-3-Clause, tree
`57c00d310bfd8ae7d5676c64b959c86fbdd61d20`. It has articulated wrists but fixed
hands. The 1.12 kg bat is a **rigid single-wrist fixture**, not a learned grasp.
The ball is a free 0.156 kg sphere with 36 mm radius. A marked pitch, both wickets
and an outfield boundary provide cricket context; regulation delivery/no-ball
gates are not implemented yet. The incoming ball's initial velocity is a declared
bowling-machine reset condition, not learned bowling.

## Shared UniLab Task Executor

The experimental `mjbatch.held_control.HeldControlRollout` also executes the
companion UniLab fork's G1 cricket task through native `Batch.step()`. It receives
the fully materialized models, including tracking/contact sensors, and preserves
the task's policy observations, reward, warmstart boundaries and float32 endpoint
caches. This path uses **UniLab's 0.70 kg wrist fixture and soft-toss curriculum**;
it does not relabel the 1.12 kg scene and SB3 experiments documented below.

Only fixed models and held CTRL/XFRC_APPLIED/EQ_ACTIVE intervals are supported. Initial
model variants, state resets and external wrenches are covered; reset-time model
mutation is rejected by the UniLab adapter. Every physics substep retains the
solved contact sensors without an extra forward call. Any MuJoCo warning fails
the interval, so a solver auto-reset cannot masquerade as healthy execution.
Different model groups are stepped sequentially; no speedup is claimed.

The recorder's tests compare every native-dtype state and sensor through actual
blade contact against official MuJoCo Rollout, including heterogeneous fixed
models, partial resets, wrenches and warmstart reset. A deliberate divergence
must raise, and the following healthy call must recover exact parity.

See the companion [G1 integration guide](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/4303f1987c237f784fb71c864c82e5088e3eca0c/G1_CRICKET.md#experimental-native-mjbatch-execution)
for its optional installation and `g1_cricket_tanh_v1/mjbatch` owner. The complete
[frozen-PPO evaluation](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/4303f1987c237f784fb71c864c82e5088e3eca0c/g1_cricket_results/native_mjbatch_v1/evaluation.json)
reproduces all **192** parent rows exactly, including returns and contact-force
evidence: 96 identities at each of 0.25/0.125 ms, both hands, zero residual/PPO,
three lanes and eight seeds. Every interval also passes independent serial
state/sensor replay. Right PPO contacts 24/24 balls but still has zero qualified
forward shots; left PPO is untrained transfer and misses all 24. This is not
new mjbatch training, material calibration or a replacement for the videos.

### Bounded Scripted Reversal Diagnostic

The companion [fixed search and complete evidence](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/6271395fc34da213a4fdb79b7e5f7f7e17249516/G1_CRICKET.md#bounded-reversal-diagnostic)
retain all 161 motion prefixes, eight exact full baseline replays and four full
candidate replays. This uses the shared UniLab task on native mjbatch, not the
legacy 1.12 kg fixture. Only one right-handed seed/center toss was searched;
the seven-arm-joint reversal is scripted, not a newly learned policy.

The sole eligible candidate produces identical complete two-second outcomes in
both executors: first-exit vx 1.028620 m/s at 0.25 ms physics dt and 1.066471 m/s
at 0.125 ms. It passes the coarse shot gate, but fine-step bat-ball penetration
is **6.620823 mm**, exceeding the unchanged 6 mm limit. Penetration also fails
the timestep agreement check, so there are **zero validated witnesses**.
Fine-step blade/fixture force peaks are 273.684 N / 99.198 N, with fixture torque
28.911 Nm; these remain uncalibrated simulated loads. No training, policy
promotion or new showcase video follows from this diagnostic.

### Terminal Command Attenuation

The companion [frozen experiment and all results](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/53955bc4a22ab91853a590871ce5cac1f5b22b1f/G1_CRICKET.md#terminal-command-attenuation)
change only the parent's tick-19 residual scale, testing six values from 0 to 1
in both engines at both timesteps. All 24 rows complete two seconds; all four
unchanged-parent baselines and all 12 cross-executor outcome/impact pairs match
exactly. Six individual coarse rows pass, but **none of the six scales passes
all contexts**. Every fine-step trial exceeds 6 mm penetration (6.422581 to
6.620823 mm), while stronger attenuation also loses the >1 m/s forward exit.

The result closes this finite last-command attenuation family without changing
physics, gates or the prior. All 67 source/input hashes are checked; 45 focused
tests pass, including 12 new tests. This is scripted, single-reset diagnostic
evidence, not fresh training, physical calibration or a showcase. The wrist-only
orientation experiment below follows this result.

### Wrist Pitch And Motor Saturation

The [companion experiment and motor audit](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/32b9fa15a0fb65420ca2ad8fbc686eb717d32f63/G1_CRICKET.md#wrist-pitch-and-motor-saturation)
retain 28 complete wrist-pitch trials across seven command scales, both engines
and both timesteps. All executor outcomes and first-impact evidence match;
ten individual coarse rows pass but **no scale passes all four contexts**.
Fine-step penetration remains 6.427966-6.620823 mm, above the unchanged 6 mm
limit. The intended contact-orientation change is barely achieved.

Four additional full native-mjbatch replays trace requested versus applied
wrist-pitch torque at every physics solve (48,000 total). For scales 1 and 0,
every forward-phase solve saturates at -5 N m at both timesteps, explaining
why different commands produce identical impacts. The affine torque/clipping
reconstruction error is zero in that phase and at most 1.78e-15 N m overall.
All replay results exactly match their retained parent rows. The sweep checks
72 source/input hashes, and the motor audit checks 75; 57 focused tests pass.

No gains, motor limits, contact physics, reward or prior were changed. The
finding concerns this specific wrist/control phase, not all wrist motion or a
physical batting-speed ceiling. It motivates a bounded proximal-arm test with
achieved joint motion and torque reporting, not more wrist-target tuning or
weaker gates. These are simulated diagnostics, not fresh learning or new videos.

### Elbow Motion With Motor Telemetry

The [companion elbow experiment](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/2cd11a09ac4a301292116bca1434a78f69a90ac0/G1_CRICKET.md#elbow-motion-with-motor-telemetry)
tests four forward-phase elbow-command scales with all other commands and
physics unchanged. All 16 full two-second rows, four exact parent baselines,
and eight exact executor pairs are retained, including motor traces. Only the
two coarse parent rows pass; **none of the scales qualifies across contexts**.
Attenuation reduces fine-step outgoing vx from 1.066471 m/s to
0.999878/0.977664/0.947774 m/s, while penetration stays above 6 mm.

Unlike the wrist plateau, the elbow moves: its fine-step forward-phase minimum
angle changes from 0.959749 to 1.118130 rad between scales 1 and 0. Only the
parent briefly saturates (4/1,600 fine-step forward-phase solves); the three
alternatives never saturate in that phase. The resulting contact normals tilt
the wrong way for the intended mechanism. All 192,000 physics solves are
checked against unchanged +/-25 N m motor limits, with maximum torque
reconstruction error 7.11e-15 N m. All 81 source/input hashes verify; 68 focused
tests pass. This closes a control family, not all humanoid batting.

No task, reward, prior, gains, motor limits or contact parameters were changed.
The separate isolated contact-compliance study below follows this diagnosis;
it is not a policy improvement or a reason to weaken gates.
Learned both-hand batting, bowling and the new README videos remain unfinished.

### Isolated Compliance Study

The [companion model study](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/aed9732d025b76feadd792d75cca39694a7a2ef0/G1_CRICKET.md#isolated-compliance-study)
retains all 16 fixed-blade trials, 240,000 physics steps and 2,821 contact
samples. It runs directly in MuJoCo, not native Batch, and leaves the robot task
unchanged. At the finest 0.03125 ms step and 4.67 m/s incident speed, shortening
the contact time constant from 4 to 2 ms reduces penetration from 6.642890 to
3.315848 mm but increases peak normal force from 374.469 to 751.048 N. Rebound
ratio stays near 0.132; the model is not calibrated to cricket materials.

All force-accounting checks pass, but only 8/12 adjacent-resolution comparisons
pass: four coarser comparisons fail for the 2 ms candidate. Its two finest
steps (0.0625 and 0.03125 ms) agree under the declared tolerances at both
incident speeds. All rows reproduce, all nine input hashes verify, and 82
focused companion tests pass. These results justify a separately versioned
robot transfer test with the 4 ms control and 2 ms candidate at the same fine
timesteps, not a learned-policy claim or permission to ignore doubled loads.
Existing highlights are preserved; no new G1 batting or bowling video is ready.

### Contact Model Transfer to G1

The [companion transfer report](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/86d88d479b57f3f3e4f40eeb6174de1691f6688f/G1_CRICKET.md#contact-model-transfer-to-g1)
now retains all 32 robot trials: both 4/2 ms contact models, both fine timesteps,
both executors, both hands and zero/scripted controls. Unlike the isolated
study, this includes native Batch execution. All 16 executor pairs match
exactly, including impact and force evidence, with zero endpoint state/sensor
error; 97 input hashes verify and 146 focused companion tests pass.

At the finest 0.03125 ms timestep the right-hand 2 ms candidate reduces
penetration from 6.578690 to 3.252920 mm, but exit velocity drops from 1.064265
to 0.996949 m/s and fails the unchanged strict >1 m/s requirement. Its
1.000953 m/s pass at 0.0625 ms therefore does not qualify. Blade peak load
increases from 269.054 to 536.053 N; wrist-fixture force from 97.345 to
190.750 N and torque from 28.454 to 52.885 N m. These are simulated loads,
not hardware safety or cricket-material calibration.

Only 2/32 individual rows pass and 0/8 comparison groups qualify. All eight
left-hand scripted transfers terminate on guarded contact at 0.18 seconds;
they are untrained same-command transfers, not mirrored batting skills. The
remaining 24 trials complete two seconds. The opt-in companion owner
`task=g1_cricket_compliance_v2/mjbatch` changes only bat-ball contact compliance;
historical owners and checkpoints stay unchanged. This is not a new native
mjbatch learner or new training evidence.

The actor-only initialization and bounded PPO run below follow this test.
No learned G1 batting/bowling showcase is ready yet.

### Imitation Initialization and Bounded PPO

The [companion experiment](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7f936c78b9e0d882087be6deedadba4525bd7224/G1_CRICKET.md#imitation-initialization-and-bounded-ppo)
has completed 2,000 actor-only imitation updates, then 256 PPO updates and
24,576 new transitions in 399.20 seconds. Training runs in the companion
MuJoCo task, not a separate native Batch learner. It uses the opt-in 2 ms
model, unchanged reward/physical limits, a fresh critic and fresh PPO optimizer.

All 24 teacher episodes and 1,885 observation/action samples are retained,
including seven early failures and four training-timestep passes; demonstrations
are not filtered for success. BC leaves the critic, initial 0.2 noise and PPO
state untouched. Both the BC-only and final PPO checkpoints, model/data hashes
and all 256 scalar iterations are retained. The 105 input pins and final payloads
verify; 53 focused companion tests pass. Final mean training reward 6.077679
and episode length 99.25 ticks are training diagnostics, not qualified shots.

The [completed 576-row evaluation](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7f936c78b9e0d882087be6deedadba4525bd7224/g1_cricket_results/bc_v1/evaluation.json)
compares zero, BC-only and PPO across both hands, three lanes, eight reused
development seeds, two fine timesteps and both executors. All 115 input hashes
verify, all 144 four-way comparisons have exact executor outcome/impact equality,
and endpoint state/sensor errors are zero. Six comparisons fail timestep
resolution checks; none fail fixture-load resolution. Executor copies are not
independent samples, and these development seeds are not held-out evaluation.

BC qualifies in 2/24 right-hand contexts across all four engine/timestep cases;
its coarse-step speeds are only 1.008442 and 1.000203 m/s above a strict >1 m/s
threshold. It has eight guarded-contact episodes and six early terminations.
PPO completes all 24 right-hand contexts with blade contact and no guard failures,
but none qualify: finest-step first-exit speeds are 0.605781-0.858866 m/s. It
improves stability/contact while losing BC's two qualified shots. Left-hand BC/PPO
use remains untrained transfer, with guard failures in all 24 contexts and 21
early terminations each. No robust policy, learned bowling or generalization
claim follows from these results.

The [16.08-second development video](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7f936c78b9e0d882087be6deedadba4525bd7224/g1_cricket_results/bc_v1/learned_development_diagnostic.mp4)
replays six predeclared final-PPO contexts through native Batch: seed 4301, both
hands and all lanes at 31.25 us. It retains failures and early terminations,
with true 0.5x playback and physics-substep force, shear, impulse, touch-occupancy
and wrist-fixture overlays. All six complete outcomes match the evaluation and
all 804 frames decode. The [frame-level manifest](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7f936c78b9e0d882087be6deedadba4525bd7224/g1_cricket_results/bc_v1/learned_development_media.json)
retains hashes and simulated, uncalibrated telemetry; 25 focused companion
BC/media tests pass. This uses the shared task's 0.70 kg bat, not a checkpoint
transfer into this standalone example's different 1.12 kg model. It is not an
advertising reel or learned bowling; earlier highlights remain available.

### Bowling Release Foundation

The native held-control recorder now accepts explicit `EQ_ACTIVE` alongside
`CTRL` and `XFRC_APPLIED`, with exact official-rollout/direct-physics comparisons.
Activation must be supplied every interval: omission intentionally restores
model defaults, not the previous release state. The companion UniLab public
equality capability owns persistence and selected-environment reset semantics.

The [both-hand G1 holder scene and tests](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/b524c8b4b7761f124270d865c38cd9e42f17886b/G1_CRICKET.md#bowling-release-foundation)
use a 0.156 kg free-joint ball, 4 ms finite-compliance wrist weld and 0.25 ms
physics step. Default and stance-keyframe ball poses come from each wrist's
forward kinematics; nominal stance clearance exceeds 3 mm. A loaded moving-wrist
test verifies continuous release, gravity-only flight and zero ball constraint
force after release. Native Batch reproduces both-hand G1 substep states and
sensors exactly; 47 local Batch/held-control tests and 163 focused companion
tests pass. These are mechanical tests, not learned bowling or hardware evidence.

The holder is an abstract constraint beside the fixed rubber hand, not an
articulated gripper. The task extension below adds randomized wrist/ball reset
alignment and holder-force telemetry; learning and full bowling evaluation remain open.
No old media or frozen evaluation is replaced. To reproduce the earlier batting
matrix, use its documented frozen revisions rather than these modified adapters.

### Both-Hand Delivery Learning Pilot

The [frozen shared-task experiment](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/10e125fdbc92e5039ecb93674516a023a45b1226/docs/g1_cricket_delivery_v1.md)
trains separate right/left PPO actors on native mjbatch through UniLab, each
with 256 updates / 24,576 actual transitions. This uses the shared Unitree prior
and abstract ball holder, not the legacy standalone environment, a learned
grasp or newly learned locomotion. The versioned scene places the robot beside
the wicket and adds regulation-spaced wickets and explicitly edged crease paint.

[All 32 evaluation rows and checkpoints](https://github.com/kishanpb/Cricket-Gym-Unilab/tree/10e125fdbc92e5039ecb93674516a023a45b1226/g1_cricket_results/delivery_v1)
are retained: two hands, zero/final-PPO, two development seeds, .25/.125 ms and
both executors. Every row completes four seconds; all 16 executor outcome pairs
match exactly and serial replay matches every control-boundary state/sensor.
**Neither trained actor releases the ball; all four PPO comparison contexts
fail**, as do all four baseline contexts. Positive reward is not delivery success.
Peak simulated holder force is 20.155614 N; no ball contacts or penetration occur.

The independent full-episode gate checks true foot lift/landing order, footprint
crease limits, overarm motion and elbow extension, all robot/ball contacts,
joint/actuator limits and post-release bounce/target crossing. Its limited
timestep checks pass for these failed carry episodes, not a demonstrated
release or impact. It is not ICC certification or hardware calibration.
The [result discussion](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/10e125fdbc92e5039ecb93674516a023a45b1226/G1_CRICKET.md#both-hand-delivery-learning-pilot)
includes complete aggregates, source/runtime pins and 49 focused passing tests.
No new showcase or upstream submission follows from this failed pilot.

### Overarm Search and Pitch Contact Audit

The [frozen 32-attempt search](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/d0cba961b0dfce81d0f1c4089bea9fdb9f235f17/g1_cricket_results/delivery_motion_v1/evaluation.json)
tests both hands with bounded motor targets, not pose or velocity injection.
Every attempt releases but fails the unchanged delivery gate: neither arm
crosses shoulder height, forward release speed stays below .282 m/s, and pitch
penetration reaches 48.405 mm. These are failed scripted trials, not learned
policies or imitation teachers. The tested family is closed without extra budget.

The [opt-in shared-task repair](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/52c894ac94978879ea11fcc92af2f645597f9355/docs/g1_cricket_pitch_contact_v2.md)
adds one explicit ball/pitch contact pair and requires physics steps <=.0625 ms.
Robot, holder, motor limits, other contacts, rewards and delivery gates remain
unchanged. This is an engineering contact model, not measured material calibration.

[All 24 isolated impacts and eight robot drops](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/52c894ac94978879ea11fcc92af2f645597f9355/g1_cricket_results/pitch_contact_v2/evaluation.json)
are retained. Native Batch matches serial MuJoCo states/sensors exactly; maximum
world-impulse/momentum residual is 5.29e-13 N s. Revised .0625/.03125 ms isolated
impacts stay below 3.651 mm penetration and 3.30% peak-force difference. Coarser
oblique comparisons fail the unchanged 5% force tolerance and remain reported.

Both-hand robot drops complete four seconds in both executors and fine steps,
with 1.287-1.307 mm maximum ball penetration, but **all eight fail the delivery
gate**. The left drop hits the foot/linkage after bouncing; later contact forces
and bounce counts remain timestep-sensitive. Full motion convergence, overarm
control and learned bowling are not established. The repair has 38 focused
UniLab tests and 47 passing native Batch tests; no new showcase replaces old media.
Reproduce at the linked revisions, not against historical source-hash contracts.

### Absolute Arm Reach and Prior Target Guard

The [versioned shared-task control change](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/3b7afaf6d6be76199f6c1c60c69fa3706fba4492/G1_CRICKET.md#absolute-arm-reach-and-prior-target-guard)
commands the selected seven arm joints with bounded absolute references while
retaining the external locomotion prior for the other 22. Seven of eight initial
scripted trials reach a sampled overhead pose, but all violate leg joint limits.
The [exact diagnostic replay](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/4cf98c6bec837883be526cba05bd7ca7f451dd83/g1_cricket_results/overarm_v1/failure_attribution.json)
reproduces every outcome and identifies unbounded prior motor targets in six
first-limit events. Physical safety cannot be inferred from reaching overhead.

The [guarded comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/3b7afaf6d6be76199f6c1c60c69fa3706fba4492/g1_cricket_results/overarm_guard_v1/evaluation.json)
repeats all eight controls at seed 6301 and .0625 ms, changing only those 22
targets to the central 90% of existing joint ranges. Motor gains, physical
joint/torque limits, scene, holder, contact response, reward and gates remain
unchanged. Native Batch and independent serial replay match each control-boundary
state and named sensor; the actual applied guard is checked every interval.

Three left-hand cases now pass the complete four-second scripted reach/recovery
checks, with 7, 32 and 31 consecutive qualifying overhead endpoints. All four
left cases have no forbidden contacts or joint/stability failures, but one does
not meet the preload criterion. **All right-hand cases still fail**, including
foot crossing and three early terminations; one joint-overshoot result worsens.
All eight intentionally retain the ball, so none is a qualified delivery.

These are development preload witnesses, not learned policies, held-out success
rates, paired-resolution validation or advertising videos. The full reports
retain all failures, poses and force diagnostics with 108 local input pins,
104 learner-runtime file pins and native executor hashes. 50 focused UniLab
tests and 47 native Batch tests pass. The drive/release study below advances
the left-hand diagnostic; right-hand gait/recovery remains unresolved.

### Fixed Overarm Drive and Release

The [frozen six-case study](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/e45cfdcddf18996a7158a3d073d17756e8e71760/G1_CRICKET.md#fixed-overarm-drive-and-release)
uses the shared guarded UniLab task on native Batch, not a separate native
learner. Starting from the left -2.80 / 1.40 rad preload, a shoulder-only target
step at 2.2 s compares +0.3 and +1 rad with release at 2.28, 2.36 and 2.44 s.
Original robot/holder dynamics, motor limits, pitch and full delivery gate are
unchanged; no pose or velocity injection is used.

All six release and complete four seconds without forbidden contacts, joint
excess, balance or actuator-limit failures. Three satisfy the current overarm
and stride proxies, but **all six fail the full delivery gate**. Forward release
speeds are [2.804, 2.021, -0.800, 3.128, 1.518, -2.311] m/s in the fixed case
order, below the >6 m/s requirement; first bounces span x=0.087-1.237 m, all
short of the x>4 m zone. No ball reaches the target. Later release redirects
motion downward rather than fixing the velocity deficit.

The [complete traces, outcomes and force diagnostics](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/e45cfdcddf18996a7158a3d073d17756e8e71760/g1_cricket_results/overarm_release_v1/evaluation.json)
retain all cases at one seed/timestep, with exact independent serial endpoint
state and named-sensor replay. Peak simulated holder force is 9.74-11.60 N,
pitch-contact peaks 1.14-1.62 kN and maximum penetration 2.846 mm; these are
uncalibrated simulator loads, not hardware safety evidence. A second complete
replay proves corrected contact/phase labels change no physical result or trace.
The retained input contract pins 111 local files and 104 learner-runtime files.

The unsigned elbow-angle proxy folds near straight, so it cannot certify
bowling legality; actual joint traces are retained and a signed extension audit
is needed before variable-elbow training. Late release also limits flight time
within the fixed four-second horizon. This result is not learned bowling, a
usable qualified teacher, a general G1 speed limit or an advertising video.
55 focused UniLab tests and 47 native Batch tests pass locally, not upstream CI.
Existing videos are unchanged; coordinated wind-up/control and both-hand learned
batting/bowling showcases remain unfinished.

### Signed Elbow and Release Reward

The [new signed hinge audit](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7835d48f9d85b5a263bc6cc72c3b1a96d1aa1351/docs/g1_cricket_signed_release_v1.md)
avoids the old unsigned angle folding through straight. It uses same-phase
world geometry and the actual elbow axis, with its proximal landmark on the
rigid elbow-parent link so shoulder yaw cannot masquerade as elbow extension.
Both-hand full-range/randomized-pose tests verify this relationship. It adds
failures for >15-degree signed extension without clearing any old failure;
the measure remains robot geometry, not umpiring certification.

The [six complete native/serial replays](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/7835d48f9d85b5a263bc6cc72c3b1a96d1aa1351/g1_cricket_results/overarm_release_v1/signed_elbow_audit.json)
exactly preserve every original outcome and motion trace. Signed extension is
0.882-1.838 degrees, so no new elbow failure is added; **all six still fail the
delivery gate**. No faster ball, successful teacher or new showcase is claimed.

New shared owner `g1_cricket_overarm_reward_v2/mjbatch` removes only the old
reward's ball-x<0 proxy for foot legality. The six retained foot/stride-legal
release states all had ball x>0 and therefore zero old release bonus. Under
the new formula, four release components become positive and two stay zero;
these are counterfactual shaping values, not new episode returns or learned
improvement. Actual foot faults remain disqualifying in independent evaluation.
The old owner and physical/action/observation contracts are preserved.

Both-hand CPU PPO smoke checks update finite actors for eight transitions each,
with 122 observations and eight actions; no skill checkpoint is retained.
64 focused UniLab tests, 47 native Batch tests and 19 documentation checks pass
locally. This is the shared UniLab task running on native Batch, not a separately
trained native implementation. Original videos remain unchanged.

### Coordinated Shoulder Search

The [bounded shoulder search](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/aec22f49cd20718dbc9bbd299e99aaf0b0d9c3e8/G1_CRICKET.md#coordinated-shoulder-search)
uses the shared UniLab task on native Batch, not an independent native learner.
SciPy differential evolution evaluates 32 coordinated shoulder-reference
trajectories at one left-hand development seed, preserving original motor
authority, fixed elbow/wrists/release time, guarded prior, full four-second
recovery and every full + signed delivery gate. It exhausts the budget without
convergence; no optimality or RL training claim is made.

[All 32 traces, failures and forces](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/aec22f49cd20718dbc9bbd299e99aaf0b0d9c3e8/g1_cricket_results/shoulder_search_v1/evaluation.json)
remain available. Twenty-three are physically clean, three have ball/hand
contacts and six have other robot self/wicket contacts. All finish four seconds,
but **zero qualify**: all fail speed, bounce-zone, bounce-count and target gates.
Forward release speed is 2.338-2.926 m/s and first-bounce x is 0.939-1.141 m.
The lowest scalar search cost does not represent a cricket-performance step
or a successful teacher for imitation learning.

Independent serial replay checks every substep and matches native endpoint
states/named sensors exactly; all initial 98 control traces match. Simulated
holder peaks are 6.686-9.054 N, maximum pitch force is 1.419 kN and maximum
penetration is 2.489 mm, not hardware-certified loads. The experiment source is
frozen at UniLab `36d2e70d`; a later tied-minimum report-selection fix changes no
retained result. 68 focused UniLab tests, 47 native Batch tests and 19 doc tests
pass locally. Existing videos are unchanged; alternate swing geometry, safe
braking, learned control and both-hand qualification remain unfinished.

### Positive Arc and Controller Damping

The [paired six-case comparison](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/a9d7cfedee86fa24e2e6c84113c87d10ccb84c11/G1_CRICKET.md#positive-arc-and-controller-damping)
tests positive shoulder launch/braking schedules at the same left-hand
development seed and fixed release time. These are scripted controls in the
shared UniLab task on native Batch, not a new native learner or RL result.
The compiled original arm controller uses kp 40/kd 10; the separately named
retuned owner changes only selected shoulder-pitch kd to 2, retaining kp,
torque caps, joint limits, remaining motors and all delivery gates.

[All six original-controller cases](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/a9d7cfedee86fa24e2e6c84113c87d10ccb84c11/g1_cricket_results/positive_arc_v1/evaluation.json)
finish four seconds but have ball/hand or wrist contact and invalid stride.
[All six retuned cases](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/a9d7cfedee86fa24e2e6c84113c87d10ccb84c11/g1_cricket_results/shoulder_damping_v1/evaluation.json)
remain available, including three incomplete episodes with leg-limit,
self-contact and pelvis-height failures. Two retuned cases are physically
clean, but neither qualifies: stride, speed, bounce-zone, bounce-count and
target-corridor gates still fail. **Zero of twelve deliveries qualify.**

Lower damping increases peak forward shoulder speed from 4.85-5.01 to
10.23-11.09 rad/s, but release timing and whole-body recovery remain unresolved.
The throwing shoulder itself stays within its original hard stop. Retuning
does not preserve the prior's closed-loop calibration or establish hardware
safety. No successful teacher, trained checkpoint or new showcase video is
claimed; existing videos remain unchanged.

Frozen experiment sources are UniLab `b8918a72` and `e53d9fea`. Every substep
is independently replayed, with exact native endpoint/sensor agreement and
input hashes checked before and after execution. A distinct action-owner
identity rejects old/new cross-owner checkpoints through the strict contract.
Local validation passes 79 focused UniLab, 48 checkpoint-resolver, 19 docs and
47 native Batch tests; this is not an upstream CI or full-suite claim.

### Bowling Task and Force Audit

The [companion task contract](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/f2b58d9e8b3d8a67274239f6ed98e43ecab9457c/docs/g1_cricket_bowling_v1.md)
adds `g1_cricket_bowling_v1/mjbatch`: seven bounded arm offsets and a one-way
policy release channel over a frozen Unitree 0.4 m/s locomotion command. This
uses the shared UniLab scene on native Batch, not the legacy standalone G1
environment or a new native learner. Ball pose is aligned to randomized wrist
kinematics only during reset; release preserves integrated position/velocity.

[All 32 retained carry/drop rows](https://github.com/kishanpb/Cricket-Gym-Unilab/blob/f2b58d9e8b3d8a67274239f6ed98e43ecab9457c/g1_cricket_results/bowling_v1/mechanics_smoke.json)
complete four seconds: both hands, both executors, .25/.125 ms physics, four
jittered rows at seed 5301, with zero arm residual and hold/drop controls. All
16 executor outcome/telemetry pairs match exactly. Minimum sampled pelvis
height is .779097 m; maximum holder force is 4.208319 N. The frozen prior carries
the ball, but no bowling policy is trained or qualified by this smoke test.

Validated signals include local fixture force, interval peak force, world-frame
holder impulse and selected-hand geometric touch occupancy. Load is not touch:
all carry/drop rows have zero hand-contact occupancy. These are simulated
fixture signals, not finger sensors, total impact force or hardware calibration.
Weld-site torque fails physical accounting in the installed MuJoCo version and
is omitted. Full-body contact, delivery-arm motion, front-foot legality and
flight/bounce gates remain necessary before advertising learned bowling.

Thirteen new bowling/report tests cover reset/release/force semantics and exact
short native trajectories. The companion focused suite passes 176 tests and
the local native Batch/held-control suite passes 47; neither is full upstream CI.
Earlier media and frozen learned-batting results remain unchanged.

## Reproduce

```sh
uv sync --python 3.13 --group examples
uv pip install --python .venv/bin/python stable-baselines3==2.7.1
uv run --no-sync python -m pytest tests/test_cricket_g1.py tests/test_cricket_g1_train.py -q
uv run --no-sync python examples/cricket_g1.py --audit examples/cricket_g1_results/physical_baseline.json
uv run --no-sync python examples/cricket_g1_train.py \
  --output examples/cricket_g1_results/balance_ppo_right --steps 524288
```

MuJoCo 3.11.0 and mujoco-menagerie 2026.9.0 were used for the retained results.
Training uses Stable-Baselines3 PPO or A2C, not a custom algorithm. The CPU runner
uses 32 native environments, four physics workers and two Torch threads. PPO has
two 128-unit hidden layers, 64 rollout steps per environment, batch size 256,
five epochs, learning rate 0.0003 and training seed 1. Control runs at 50 Hz with
ten 2 ms physics substeps. A2C is supported by the runner but is not yet benchmarked.

Each action is a bounded 29-joint target offset from the reset pose. The balance
reward tracks uprightness and height, penalizes drift, action changes, posture
deviation and joint-limit violations, and terminates on falls. `--task batting`
adds provisional ball-distance/contact shaping; that objective is **not validated**
and no trained batting result is currently claimed.

Observations include simulator joint/root state, privileged ball state, the last
action, bat position, foot support loads and fixture wrench. This is not a
vision-only or deployable hardware sensing setup. Kinematic/sensor fields are
from the final physics substep's evaluation stage, which precedes the resulting
integrated state by one step; no extra forward force solve replaces those samples.

## External Locomotion Prior

The next route uses Unitree's **externally trained** 29-DoF G1 velocity policy,
not one of our failed stance checkpoints. The pinned
[Unitree RL Lab source](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3)
provides `deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx`
and its paired `params/deploy.yaml`. No cricket training or vendor pretraining
performed locally is claimed. We do not redistribute these two external assets:
the upstream README advertises Apache-2.0 but the pinned tree has no root license
file; checkpoint redistribution has not been cleared.

The adapter verifies both SHA-256 hashes before inference. Its 480 inputs use
pelvis-frame angular velocity and projected gravity, zero velocity commands,
joint positions/velocities in the official policy order, and previous raw actions.
Each term contains five frames, oldest first, initialized by repeating the first
frame. Policy output is mapped back to all 29 native joints. Official SDK-order
PD gains, policy-order default pose and 20 ms control are explicit changes from
our custom stance controller. Robot inertias, collision pairs, joint/torque limits,
free base and 2 ms physics remain intact; there is no elastic band, gravity
compensation or pose overwrite after reset. Targets are bounded by native joint
limits; no retained prior step required that clipping.

All eight predeclared seeds (4201-4208, joint-reset jitter +/-0.005 rad) were run
for each of three models and two controllers, with a ten-second horizon. The
original matrix below is frozen at `7c90ade898c83e0b44a138f90fbe69acfec1ce77`:

| Model | Constant default target | External Unitree policy |
| --- | --- | --- |
| No bat | 8/8 falls, 1.234-1.280 s | 8/8 completed 10 s |
| Right wrist fixture | 8/8 falls, 1.330-1.396 s | 8/8 completed 10 s |
| Left wrist fixture | 8/8 falls, 1.330-1.396 s | 8/8 completed 10 s |

[All 48 rows](cricket_g1_results/unitree_prior/evaluation.json) retain failures,
contact counts/loads, state extrema, runtime and source hashes. Every physics
step matches an independent serial MuJoCo shadow exactly in qpos/qvel. All 24
prior trials have no incidental bat contact, no non-foot robot-ground contact,
no joint-limit excess and no applied joint-torque-limit violation. Maximum XY
drift is 0.016 m; minimum pelvis height is 0.7869 m. This is narrow zero-command
transfer evidence, not robustness certification, a matched PPO/A2C comparison,
or a learned batting/bowling result. The bat pose is not a cricket-ready stance.

![External prior diagnostic: first declared seed, fixed times, not learned cricket](cricket_g1_results/unitree_prior/stance_diagnostic.png)

### Wicket-complete collision contract v2

The original scene omitted bat/robot-to-wicket pairs and all bail contacts.
The current scene adds explicit pairs and per-physics-step contact sensors for
both bat geoms and every robot collider against all six stumps and both bails;
ball/bail pairs are included too. There are 232 new bat/robot wicket channels.
Any bat/robot wicket presence terminates the PPO/A2C task and prior audit, even
when the optional general bat-contact guard is disabled. Zero reported force
does not excuse contact. Stock robot self-collision pairs and inertias are unchanged.

The [new complete 48-row audit](cricket_g1_results/unitree_prior_wickets_v2/evaluation.json)
retains **24/24 imported-prior ten-second passes and 24/24 constant-target
failures**, with native/serial qpos/qvel equality at every 2 ms step. Each
bat-equipped episode includes the expanded wicket counts and peak normal loads.
Unit tests cover every explicit pair and 48 injected overlap cases across both
hands, both bat geoms, a representative robot collider and all eight wicket
geoms; native force records exactly match serial MuJoCo. Those penetrations are
synthetic fault tests, not realistic impact-force measurements.

The freshly rendered diagnostic is byte-identical to the retained figure above;
the duplicate image is omitted.

Earlier training and contact reports remain historical evidence under their
original scene, not validations of v2. This repair does not establish legal
bowling, a learned shot, deformable/breakable stumps, or calibrated contact loads.

Reproduce after obtaining the pinned files from the source above in a local
cache outside this checkout; pass that directory, containing `policy.onnx` and
`deploy.yaml`, as `--assets`. No checkpoint download occurs implicitly:

```sh
uv pip install --python .venv/bin/python onnxruntime==1.30.0 pyyaml==6.0.3
uv run --no-sync python examples/cricket_g1_prior.py --assets /path/to/local/cache \
  --output examples/cricket_g1_results/unitree_prior_wickets_v2/evaluation.json
uv run --no-sync python -m pytest tests/test_cricket_g1_prior.py -q
```

The optional integration tests use the macOS cache
`~/Library/Caches/unitree_rl_lab/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` and
skip if those external assets are absent. Report/schema and hash-rejection tests
remain separate. Physics-step sensor timing is unchanged; fixture/contact data
remain simulated, uncalibrated signals, not hardware tactile measurements.

## Height Observation Comparison

`--observe-root-height` appends pelvis height relative to the 0.78 m target,
changing the input from 117 to 118 values without changing physics or reward.
New checkpoints record this contract and restore it on resume; an incompatible
checkpoint/input shape is rejected rather than silently reinterpreted. Old
checkpoints and the default observation remain unchanged.

Two fresh seed-1 PPO runs used 524,288 transitions each, fixed action std 0.08,
target KL 0.02, learning rate 0.0003 and the same free-bat contact guard:

| Observation | All eight deterministic development trials |
| --- | --- |
| Original 117 inputs | Contact failure at 0.72-0.74 s; 0 successes |
| Height-aware 118 inputs | Contact failure at 0.70 s; 0 successes |

[Original evidence](cricket_g1_results/observation_comparison/original/evaluation.json)
and [height-aware evidence](cricket_g1_results/observation_comparison/height/evaluation.json)
retain every row and both training curves. This budget does not establish a
height-observation benefit or a solved stance. Different input widths also
change seeded network initialization; this is not an independently replicated
causal comparison. Neither candidate advanced to the ten-second gate.

Reproduce either arm with `examples/cricket_g1_train.py`, `--steps 524288`,
`--forbid-bat-contact --fixed-action-std .08 --target-kl .02 --learning-rate .0003`
and a distinct `--output`; add `--observe-root-height` only for the height arm.

## Contact Evidence

Explicit pairs cover ball/bat, robot, ground, stumps and bails, plus bat/robot,
bat/ground and bat/robot-to-wicket collisions. The fixture intentionally excludes bat collisions with its holding
palm/wrist. Original robot self-collision pairs are retained. Geometry-level
contact records report force, torque, distance, position and contact frame;
normal loads and contact-presence counts are sampled after every physics step.
Presence uses the sensor's `found` field, including zero-normal-load contacts;
it is not inferred from a positive force threshold.
Both feet expose support records. Fixture force/torque sensors include gravity
and inertial loads, not finger pressure or hardware tactile taxels.

Tests compare native stepping against serial MuJoCo exactly, preserve original
robot inertias/motor parameters, verify selective reset, demonstrate gravity
without actuation, and exercise a controlled incoming-ball contact. These tests
verify integration and instrumentation, **not realistic impact magnitudes**.
Timestep/contact-parameter convergence and calibration remain required.

The [16-case timestep audit](cricket_g1_results/contact_timestep_audit.json) retains
both hands at 2 and 8 m/s over four stepsizes from 2 ms to 0.25 ms. Coarse versus
finest peak normal loads differ by 4.84-9.20%; summed normal impulses differ by
0.35-2.11%. These are controlled short impacts on the free robot with constant
joint targets, not learned shots. The current critically damped contact has very
little rebound; restitution/material calibration is still a prerequisite for
credible batting dynamics. Numerical agreement alone does not supply it.

[Allen et al. (2014)](https://shura.shu.ac.uk/8205/) validated cricket ball/bat
impacts experimentally and found limitations in rigid-body predictions across
blade locations. This motivates a separate rebound/contact-duration calibration
study; their data are not a calibration of this wrist fixture or MuJoCo model.

## Results And Limits

**Current physical gate: not passed.** A later PPO checkpoint stayed upright for
all eight three-second development trials and all eight ten-second stress trials,
but the added bat-ground sensors exposed support loads up to 1,364 N. It was using
the bat as a crutch. This is rejected as a cricket stance, not a promoted result.

![Rejected stance on the first declared stress seed](cricket_g1_results/balance_ppo_fine/stance_diagnostic.png)

The [complete physical audit](cricket_g1_results/balance_ppo_fine/balance_validation.json)
separates the no-fall gate from the cricket-stance gate. The new free-bat curriculum
terminates on any incidental bat-ground or bat-body contact; the declared holding
wrist fixture is excluded geometrically. Ball/bat contact remains allowed.
Do not compare its returns directly with the older, unguarded curriculum.

| Development condition | Additional transitions | Deterministic outcome |
| --- | ---: | --- |
| Fixed action std 0.08 from bootstrap | 524,288 | 8/8 falls, 1.88-2.68 s |
| Same parent/std plus PPO target KL 0.02 | 524,288 | 5/8 falls; only 3/8 reach 3 s |
| Continue KL checkpoint at learning rate 0.0001 | 524,288 | 0/8 falls at 3 s and 10 s, rejected for bat support |
| Free-bat v2 continuation, same physics | 262,144 | 8/8 invalid bat contacts, 1.00-1.04 s; 0 successes |
| Continue free-bat v2, unchanged contract | 1,048,576 | 8/8 falls, 1.88-2.70 s; no incidental bat contacts, 0 successes |
| Further free-bat v2 continuation | 1,048,576 | 7/8 contact failures, 1/8 fall, 1.82-2.16 s; 0 successes |

The frozen-noise and KL settings use Stable-Baselines3's existing PPO policy and
update stopping logic. These are bounded development comparisons, not a final
tournament, independently seeded training replication or an algorithm ranking.
The second unchanged continuation regressed and is not promoted. More budget
has not established contact-free balance; the next investigation is the stance
observation/control contract rather than another unchanged continuation.
Both continuation evaluations replay exactly after the contact-presence guard
fix. Their saved training-source hashes describe the implementation at commit
`b487c9d`; the subsequent guard correction did not retrain the checkpoints.

The complete first-episode evaluations use development seeds 9001-9008, all eight
rows retained. These repeated development seeds are not an untouched final test
set. The constant-target baseline falls after about 0.9 s; the initial 524,288-step
PPO checkpoint lasts 2.02-2.12 s but still falls in all eight trials. See
[evaluation](cricket_g1_results/balance_ppo_right/bootstrap_evaluation.json) and
[development history](cricket_g1_results/experiment_history.md). Longer continuation
results are kept separately from that parent checkpoint.

The continuation to 1,572,864 total transitions regressed to 1.56-1.66 s with
eight falls out of eight, despite a training-episode average near 144 control
steps. The [complete continuation evaluation](cricket_g1_results/balance_ppo_right/evaluation.json)
is retained, not promoted. The action-sampling diagnostic below investigated
that training/evaluation mismatch.

A [fixed-seed action-sampling probe](cricket_g1_results/balance_ppo_right/action_sampling_probe.json)
reproduces both deterministic evaluations exactly. Sampled actions reduce falls
to 5/8 for the parent and 6/8 for the continuation, but still fail the balance
gate. It retains all 32 diagnostic episodes, not selected successful examples.

Before advertising learned cricket, the remaining gates are sustained balance,
both-handed bat interception and strike attribution, physical bowling/release,
full PPO/A2C comparison without selected episodes, joint/torque and foot-fault
audits, impact sensitivity checks, and videos of actual learned control.
The current UniLab prototype has a different 0.70 kg fixture and control period;
cross-framework policy/performance parity is not established.
Both handed scenes now align the bat, incoming ball and wicket line; the pitch,
creases and boundary center mirror with them. Tests cover this alignment and
the previous right-hand checkpoint evaluation replays exactly. Left-handed
learned batting is not established.
