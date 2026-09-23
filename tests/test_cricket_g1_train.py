# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_menagerie")
pytest.importorskip("stable_baselines3")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from cricket_g1_train import CricketVecEnv


def test_seeded_native_vector_observations_and_actions():
  a, b = CricketVecEnv(2, seed=7), CricketVecEnv(2, seed=7)
  np.testing.assert_array_equal(a.reset(), b.reset())
  action = np.full((2, 29), .01, dtype=np.float32)
  for _ in range(4):
    x, y = a.step(action), b.step(action)
    np.testing.assert_array_equal(x[0], y[0])
    np.testing.assert_array_equal(x[1], y[1])
    np.testing.assert_array_equal(a.physics.qpos, b.physics.qpos)
    assert all(a.observation_space.contains(obs) for obs in x[0])
  before = a.physics.qpos[1].copy()
  a._reset_rows(np.array([0]))
  np.testing.assert_array_equal(a.physics.qpos[1], before)


def test_timeout_keeps_terminal_observation_and_resets_episode():
  env = CricketVecEnv(2, seed=11)
  env.reset()
  env.steps[:] = 149
  obs, reward, done, infos = env.step(np.zeros((2, 29)))
  assert done.all()
  assert np.isfinite(reward).all()
  assert (env.steps == 0).all()
  for i, info in enumerate(infos):
    assert info["TimeLimit.truncated"]
    assert not info["fell"]
    assert info["episode"]["l"] == 150
    assert info["terminal_observation"].shape == obs[i].shape
    assert not np.array_equal(info["terminal_observation"], obs[i])
