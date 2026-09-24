# SPDX-License-Identifier: Apache-2.0

"""Render the first declared stress seed through failure, not a cricket highlight."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from cricket_g1_train import CricketVecEnv


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--directory", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(2)
  policy = PPO.load(args.directory / "policy.zip", device="cpu")
  env = CricketVecEnv(1, seed=19101, observe_root_height=getattr(policy, "observe_root_height", False))
  obs = env.reset()
  physics = env.physics
  data = mujoco.MjData(physics.model)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [.2, 0, .65]
  camera.distance, camera.azimuth, camera.elevation = 3.4, 130, -16
  snapshots = []
  invalid_bat_contact = fallen = False
  with mujoco.Renderer(physics.model, height=360, width=640) as renderer:
    for step in range(501):
      q = physics.qpos[0]
      fallen = q[2] < .50 or 1 - 2 * (q[4]**2 + q[5]**2) < .65
      invalid_bat_contact = any(physics.active_samples[0, i] > 0
                                for i, name in enumerate(physics.contact_names) if name.startswith("bat_"))
      failed = fallen or invalid_bat_contact
      if step % 25 == 0 or failed:
        # Only the separate render data is forwarded; physics force samples are untouched.
        data.qpos[:], data.qvel[:] = physics.qpos[0], physics.qvel[0]
        mujoco.mj_forward(physics.model, data)
        renderer.update_scene(data, camera)
        snapshots.append((step, renderer.render().copy()))
      if step == 500 or failed:
        break
      action = policy.predict(obs, deterministic=True)[0]
      physics.step(action)
      env.last_action[:] = action
      obs = env.observation()
  fig, axes = plt.subplots(2, 3, figsize=(12, 6))
  selected = np.linspace(0, len(snapshots) - 1, min(6, len(snapshots)), dtype=int)
  for axis in axes.flat:
    axis.axis("off")
  for axis, index in zip(axes.flat, selected, strict=False):
    step, image = snapshots[index]
    axis.imshow(image)
    axis.set_title(f"{step * .02:g} s")
    axis.axis("off")
  outcome = "fall" if fallen else "incidental bat contact" if invalid_bat_contact else "10 s completed"
  fig.suptitle(f"G1 stance diagnostic: {outcome} | seed 19101 | not learned cricket")
  fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.5)
  fig.savefig(args.directory / "stance_diagnostic.png", dpi=140)
  plt.close(fig)
  print({name: float(physics.peak_load[0, i]) for i, name in enumerate(physics.contact_names)
         if name.startswith("bat_") and physics.peak_load[0, i] > 0})
  assert all(np.std(image) > 10 for _, image in snapshots)


if __name__ == "__main__":
  main()
