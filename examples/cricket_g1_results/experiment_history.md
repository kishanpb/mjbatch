# G1 Development History

## Initial Balance Smoke

2026-09-23: Stable-Baselines3 PPO, seed 1, 32 native mjbatch environments,
32,768 control transitions, 7.98 s training wall time. Task: balance while carrying
a 1.12 kg rigid wrist-mounted bat; no scripted root stabilization. No batting or
bowling skill claim.

Fixed development seeds 9001-9008 all fell for both constant-target baseline and
trained policy. Baseline lengths: 43, 43, 44, 44, 44, 44, 44, 44 control steps.
Trained lengths: 44, 44, 45, 45, 45, 45, 44, 45. Control period: 0.02 s.
This run verifies the learning pipeline but does not solve balance. Its checkpoint
and report are superseded by the next bounded, from-scratch run in the same canonical
`balance_ppo_right` directory; no successful result was filtered away.

## First Balance Checkpoint

From-scratch PPO seed 1, 524,288 transitions, 135.90 s training time. All eight
development trials still fell, after 104, 103, 102, 103, 101, 101, 103, 106 control
steps (2.02-2.12 s). Training-episode survival improved, but no three-second
balance success was established. The bootstrap checkpoint, curve and evaluation
are retained because the continuation depends on this exact parent; its SHA-256
is `cd507310def331ca2ebbc398e9efaae700cd001bf127f2c3a04aca0169b7bda3`.

## Continuation Regressed

An additional 1,048,576 PPO transitions from that parent reached 1,572,864 total
transitions in 271.88 s of additional training. No physics, reward or optimizer
hyperparameter was changed. All eight deterministic development trials still
fell, now after 82, 80, 79, 78, 81, 79, 83, 79 control steps (1.56-1.66 s).
The final training rollout average was about 144 steps, so training averages
are not evidence of deterministic balance. This continuation is a regression,
not a promoted checkpoint. `policy.zip`, `progress.csv` and `evaluation.json`
retain it, including the exact parent hash, runtime versions and configuration.
The next design question is the stochastic-training/deterministic-control gap;
more transitions alone are not justified as a demonstrated solution.

## Action-Sampling Diagnostic

The same eight development episodes were replayed with deterministic actions and
sampled policy actions (Torch seed 42 per eight-row vector evaluation). Both
deterministic replays exactly reproduce the retained reports: 8/8 falls. Sampled
actions produce 5/8 falls for the bootstrap and 6/8 for the continuation. This
partly explains the training/evaluation gap, but still does not solve balance.
All 32 episodes are retained in `action_sampling_probe.json`; none is selected
for a skill or success-rate claim beyond this development diagnostic.

## Fixed Noise, KL And Physical Rejection

2026-09-23: from the same 524,288-transition bootstrap, fixed action std 0.08
for 524,288 further transitions produced eight falls at 1.88-2.68 s. Adding PPO
target KL 0.02 from that same parent/budget produced five falls and three
three-second completions. Continuing the KL checkpoint for 524,288 transitions
at learning rate 0.0001 produced eight three-second and eight ten-second
completions, but this apparent stability was rejected by physical inspection.

The first declared stress seed, 19101, visibly uses the bat on the floor.
New bat/ground and bat/body sensors expose up to 1,363.69 N normal load across
the complete eight-seed physical audit. Every old development episode fails
the no-incidental-bat-contact gate at 0.56-0.58 s. This is a reward shortcut,
not a valid cricket stance, despite passing the original no-fall gate.

The versioned free-bat v2 curriculum preserves physics and additionally
terminates on incidental bat-ground/body contact. Its 262,144-transition warm
start from the rejected checkpoint still has eight invalid contacts at
1.00-1.04 s and zero successes. All rows and the current v2 checkpoint remain.
Returns across the two termination contracts are not directly comparable.

The fixed-noise-only failed checkpoint is recoverable from commit `ed4981f`;
its current redundant binary is removed, while the complete evaluation and
training curve stay. Other retained checkpoints either provide the comparison
probe, a parent in the active curriculum lineage, or the current candidate.

## Guarded Continuation And Scene Alignment

From the 1,835,008-transition free-bat v2 checkpoint, another 1,048,576 transitions
at unchanged settings took 296.38 s. All eight development trials avoid incidental
bat contact but still fall, at 1.92, 1.96, 2.38, 2.42, 2.70, 2.12, 1.88 and 2.20 s.
This removes the observed support shortcut at evaluation, not the balance failure.
The first declared stress seed 19101 also falls (1.92 s); its diagnostic stops
at failure instead of continuing into post-fall motion. No ten-second gate was run.

The left-handed scene now mirrors ball, wickets, crease, pitch and boundary
alignment to the bat side. Right-hand physics is unchanged: all eight prior
free-bat v2 evaluation rows replay exactly. The 16-case timestep audit and
untrained physical baseline were regenerated for the corrected scenes.

A further 1,048,576 transitions reached 3,932,160 total in 300.15 s additional
training. The eight trials ended at 2.10, 1.82, 1.84, 2.00, 2.06, 1.88, 1.94 and
2.16 s: seven incidental-contact failures and one fall (seed 9004), zero successes.
This continuation regresses; both full reports/curves and their parent checkpoints
remain available, with no selected-row promotion or ten-second success claim.
Pause unchanged-budget scaling and inspect the stance observation/control contract.

The contact counter now uses sensor `found`, not positive normal load. A focused
zero-load-contact regression test passes, and both latest checkpoints reproduce
all 16 original development rows exactly under the stricter presence guard.
Training/evaluation source snapshots remain recoverable at `b487c9d`; no old
training hash was replaced with a post-training implementation hash.

Superseded `balance_ppo_fixed_noise_kl/policy.zip` and
`balance_free_bat_v2/policy.zip` binaries are removed from the current tree; both
are recoverable at `6b7961a`. Their complete evaluations, curves and descendant
parent hashes remain. Keep the bootstrap/action-sampling pair, the rejected
bat-crutch counterexample and the two latest guarded checkpoints for current
comparisons instead of accumulating every optimizer snapshot.
