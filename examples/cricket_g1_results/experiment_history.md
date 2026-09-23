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
