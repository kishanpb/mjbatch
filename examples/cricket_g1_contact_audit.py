# SPDX-License-Identifier: Apache-2.0

"""Short free-base G1 impact probes; numerical sensitivity, not calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cricket_g1 import G1Cricket


def impact(hand, timestep, speed, duration=.08):
  env = G1Cricket(hand=hand, timestep=timestep)
  ball = env.batch.joint("cricket_ball_joint")
  bat = env.batch.site("bat_sweet_spot")
  fixture_force = env.batch.sensor("bat_fixture_force")
  fixture_torque = env.batch.sensor("bat_fixture_torque")
  env.batch.forward()
  ball.qpos[0, :3] = bat.xpos[0] + [.038, 0, 0]
  ball.qvel[0, :3] = [-speed, 0, 0]
  env.batch.forward()
  sensor = env.contacts[env.contact_names.index("ball_bat_blade")]
  peak, impulse, force_peak, torque_peak = 0., 0., 0., 0.
  active = 0
  for _ in range(round(duration / timestep)):
    env.batch.step()
    if np.any(sensor[..., 0] > sensor.shape[1]):
      raise RuntimeError("impact sensor overflow")
    normal_load = float(sensor[0, :, 1].sum())
    peak = max(peak, normal_load)
    impulse += timestep * normal_load
    active += normal_load > 0
    force_peak = max(force_peak, float(np.linalg.norm(fixture_force[0])))
    torque_peak = max(torque_peak, float(np.linalg.norm(fixture_torque[0])))
  if active == 0 or env.warning[:, :, 1].any() or not np.isfinite(env.qpos).all():
    raise RuntimeError("invalid impact probe")
  return {
    "hand": hand, "physics_timestep_s": timestep, "incoming_speed_m_s": speed,
    "duration_s": duration, "active_contact_samples": active,
    "peak_normal_load_n": peak, "summed_normal_impulse_magnitude_ns": impulse,
    "peak_fixture_force_norm_n": force_peak, "peak_fixture_torque_norm_nm": torque_peak,
    "final_ball_velocity_m_s": ball.qvel[0, :3].tolist(),
    "final_pelvis_height_m": float(env.qpos[0, 2]), "provenance": env.provenance,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  rows = [impact(hand, dt, speed) for hand in ("right", "left")
          for speed in (2., 8.) for dt in (.002, .001, .0005, .00025)]
  comparisons = []
  for hand in ("right", "left"):
    for speed in (2., 8.):
      group = [row for row in rows if row["hand"] == hand and row["incoming_speed_m_s"] == speed]
      coarse, reference = group[0], group[-1]
      comparisons.append({"hand": hand, "incoming_speed_m_s": speed,
                          "2ms_vs_0_25ms_peak_relative_error": abs(coarse["peak_normal_load_n"] / reference["peak_normal_load_n"] - 1),
                          "2ms_vs_0_25ms_impulse_relative_error": abs(coarse["summed_normal_impulse_magnitude_ns"] / reference["summed_normal_impulse_magnitude_ns"] - 1)})
  sources = (Path(__file__), Path(__file__).with_name("cricket_g1.py"))
  report = {
    "scope": "Controlled reset-only incoming ball, constant motor targets, floating robot; not learned cricket or hardware calibration",
    "setup": "Ball starts 2 mm clear of the bat face; 80 ms simulated duration; all declared cases retained",
    "sensor_timing": "Integrated physics-step force stage, no post-step forward substitution",
    "impulse_scope": "Sum of normal-force magnitudes times timestep, not net vector impulse or a pressure estimate",
    "fixture_scope": "Rigid attachment wrench includes gravity/inertia, not finger taxels",
    "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    "rows": rows, "comparisons": comparisons,
  }
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps(comparisons))


if __name__ == "__main__":
  main()
