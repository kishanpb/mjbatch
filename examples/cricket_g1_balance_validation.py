# SPDX-License-Identifier: Apache-2.0

"""Predeclared ten-second G1 stance checks after the short curriculum gate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from cricket_g1_train import CricketVecEnv, evaluate


def physical_rollout(policy, seed, horizon=500):
  env = CricketVecEnv(1, seed=seed)
  obs = env.reset()
  physics = env.physics
  initial = physics.qpos[0, :3].copy()
  max_drift, max_joint_error, force_peak, torque_peak = 0., 0., 0., 0.
  min_height = float(initial[2])
  for _step in range(horizon):
    actions = policy.predict(obs, deterministic=True)[0]
    physics.step(actions)
    env.last_action[:] = actions
    obs = env.observation()
    q = physics.qpos[0]
    min_height = min(min_height, float(q[2]))
    max_drift = max(max_drift, float(np.linalg.norm(q[:2] - initial[:2])))
    joints = q[physics.model.jnt_qposadr[physics.joints]]
    error = np.maximum(physics.limits[:, 0] - joints, 0) + np.maximum(joints - physics.limits[:, 1], 0)
    max_joint_error = max(max_joint_error, float(error.max()))
    force_peak = max(force_peak, float(np.linalg.norm(physics.batch.sensor("bat_fixture_force")[0])))
    torque_peak = max(torque_peak, float(np.linalg.norm(physics.batch.sensor("bat_fixture_torque")[0])))
    fallen = q[2] < .50 or 1 - 2 * (q[4]**2 + q[5]**2) < .65
    if fallen:
      break
  return {"seed": seed, "seconds": (_step + 1) * .02, "fell": bool(fallen),
          "minimum_pelvis_height_m": min_height, "maximum_xy_drift_m": max_drift,
          "maximum_joint_limit_excess_rad": max_joint_error,
          "fixture_force_control_snapshot_peak_n": force_peak,
          "fixture_torque_control_snapshot_peak_nm": torque_peak,
          "contact_active_substep_counts": dict(zip(physics.contact_names, physics.active_samples[0].tolist(), strict=True)),
          "contact_normal_substep_peaks_n": dict(zip(physics.contact_names, physics.peak_load[0].tolist(), strict=True))}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--directory", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(2)
  checkpoint = args.directory / "policy.zip"
  policy = PPO.load(checkpoint, device="cpu")
  short = evaluate(policy, "right", "balance", forbid_bat_contact=True)
  if not all(row["success"] for row in short):
    raise RuntimeError("the eight-episode contact-free short curriculum gate has not passed")
  stress = evaluate(policy, "right", "balance", horizon=500, seed=19101, forbid_bat_contact=True)
  physical = [physical_rollout(policy, seed) for seed in range(19101, 19109)]
  report = {
    "scope": "Balance only, right-hand wrist fixture, no learned batting/bowling or safety claim",
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "short_development": short, "ten_second_stress": stress,
    "physical_diagnostics": physical,
    "physical_diagnostics_scope": "All eight seeds, independently simulated single-row batches; fixture snapshots at control rate, contact peaks at physics rate",
    "ten_second_gate_passed": all(row["success"] for row in stress) and not any(row["fell"] for row in physical),
    "guarded_three_second_evaluation": short,
    "cricket_stance_gate_passed": all(row["success"] for row in stress) and not any(row["fell"] for row in physical) and
        not any(value > 0 for row in physical for name, value in row["contact_active_substep_counts"].items()
                if name.startswith("bat_")),
    "physical_gate": "No incidental bat-ground or bat-robot support/contact; existing wrist fixture remains declared",
    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__), Path(__file__).with_name("cricket_g1.py"), Path(__file__).with_name("cricket_g1_train.py"))},
  }
  (args.directory / "balance_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps({"ten_second_gate_passed": report["ten_second_gate_passed"],
                    "cricket_stance_gate_passed": report["cricket_stance_gate_passed"],
                    "stress": [(r["seed"], r["seconds"], r["fell"]) for r in stress]}))


if __name__ == "__main__":
  main()
