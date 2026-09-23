# SPDX-License-Identifier: Apache-2.0

"""Compare deterministic and sampled actions on the fixed development episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from cricket_g1_train import evaluate


class SampledPolicy:
  def __init__(self, policy):
    self.policy = policy

  def predict(self, observation, deterministic=True):
    return self.policy.predict(observation, deterministic=False)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--directory", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(2)
  rows = []
  for filename in ("bootstrap_policy.zip", "policy.zip"):
    path = args.directory / filename
    policy = PPO.load(path, device="cpu")
    for sampled in (False, True):
      torch.manual_seed(42)
      result = evaluate(SampledPolicy(policy) if sampled else policy, "right", "balance")
      rows.append({"checkpoint": filename, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "action_selection": "sampled" if sampled else "deterministic",
                   "torch_action_sampling_seed": 42 if sampled else None,
                   "falls": sum(r["fell"] for r in result), "episodes": result})
  report = {"scope": "Development diagnostic only; stochastic actions do not establish deterministic balance",
            "sampling_contract": "One Torch seed for each eight-row vector evaluation; all first episodes retained",
            "rows": rows}
  (args.directory / "action_sampling_probe.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps([{k: row[k] for k in ("checkpoint", "action_selection", "falls")} for row in rows]))


if __name__ == "__main__":
  main()
