# G1 Balance Development Plan

The goal is stable free-floating control before cricket interception, not a
selected upright frame. Robot inertias, actuator limits, bat fixture, contact
parameters, reset jitter and reward stay unchanged for this experiment.

## Fixed Action Noise

- Parent: `balance_ppo_right/bootstrap_policy.zip`, 524,288 transitions,
  SHA-256 `cd507310def331ca2ebbc398e9efaae700cd001bf127f2c3a04aca0169b7bda3`.
- Change: hold all 29 Gaussian action standard deviations at 0.08, rather than
  learning the exploration variance. The mean policy is still trained with PPO.
- Budget: 524,288 additional transitions, 32 environments, training seed 1;
  remaining optimizer and environment settings unchanged.
- Primary diagnostic: all eight deterministic development episodes, seeds
  9001-9008, three seconds each; falls remain failures and are never filtered.
- Advancement requires all eight three-second episodes without a fall, followed
  by a separate ten-second check on seeds 19101-19108. Passing this is only a
  balance curriculum result, not batting, bowling or robot safety validation.
- If low-noise learning produces excessive PPO KL, the next separate comparison
  may bound update KL at 0.02 from the same parent and budget. Do not attribute a
  combined result to noise reduction alone.

Both unchanged and changed conditions use the same stock robot dynamics.
There is no root support, joint-pose trajectory overwrite or relaxed fall gate.
The existing longer learned-variance continuation is failed development context,
not an equal-budget causal control or a promoted policy.

## Lower-Step-Size Continuation

The fixed-noise/KL run produced three non-falling three-second episodes and five
falls. Continue that exact checkpoint for 524,288 transitions with learning rate
0.0001 instead of 0.0003, retaining fixed noise 0.08 and target KL 0.02. This
tests smaller updates because the early-stop mechanism can only detect an
oversized update after it occurs. Keep the same eight-episode development gate;
do not run the predeclared ten-second check unless all eight first pass.

## Free-Bat V2 Physical Gate

The smaller-step checkpoint passed both no-fall checks, but subsequent contact
instrumentation and first-seed rendering showed bat-floor support. Reject it
for cricket stance. Add the explicit no-incidental-bat-ground/body-contact gate
without removing those collision pairs or changing robot physics. Continue
from that checkpoint for 262,144 transitions at its saved optimizer settings.
Count every invalid contact as failure, not a timeout or successful survival;
this is a newly versioned termination/reward contract, not the old return scale.
