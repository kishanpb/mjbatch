# SPDX-License-Identifier: Apache-2.0

"""Render the first declared stress seed as a diagnostic, not a cricket highlight."""

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
  env = CricketVecEnv(1, seed=19101)
  obs = env.reset()
  physics = env.physics
  data = mujoco.MjData(physics.model)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [.2, 0, .65]
  camera.distance, camera.azimuth, camera.elevation = 3.4, 130, -16
  capture_steps = (0, 25, 50, 100, 250, 500)
  images = []
  with mujoco.Renderer(physics.model, height=360, width=640) as renderer:
    for step in range(501):
      if step in capture_steps:
        # Only the separate render data is forwarded; physics force samples are untouched.
        data.qpos[:], data.qvel[:] = physics.qpos[0], physics.qvel[0]
        mujoco.mj_forward(physics.model, data)
        renderer.update_scene(data, camera)
        images.append(renderer.render().copy())
      if step == 500:
        break
      action = policy.predict(obs, deterministic=True)[0]
      physics.step(action)
      env.last_action[:] = action
      obs = env.observation()
  fig, axes = plt.subplots(2, 3, figsize=(12, 6))
  for axis, image, step in zip(axes.flat, images, capture_steps, strict=True):
    axis.imshow(image)
    axis.set_title(f"{step * .02:g} s")
    axis.axis("off")
  fig.suptitle("Rejected G1 stance: bat-floor support | seed 19101 | not learned cricket")
  fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.5)
  fig.savefig(args.directory / "stance_diagnostic.png", dpi=140)
  plt.close(fig)
  print({name: float(physics.peak_load[0, i]) for i, name in enumerate(physics.contact_names)
         if name.startswith("bat_") and physics.peak_load[0, i] > 0})
  assert all(np.std(image) > 10 for image in images)


if __name__ == "__main__":
  main()
