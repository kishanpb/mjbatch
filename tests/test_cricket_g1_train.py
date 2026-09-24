# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco_menagerie")
pytest.importorskip("stable_baselines3")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from cricket_g1_train import CricketVecEnv


def test_fixed_noise_survives_optimizer_update_and_checkpoint(tmp_path):
  import torch
  from stable_baselines3 import PPO

  from cricket_g1_train import fix_action_noise

  torch.set_num_threads(2)
  env = CricketVecEnv(2)
  policy = PPO("MlpPolicy", env, n_steps=2, batch_size=4, n_epochs=1, seed=1, target_kl=.02,
               policy_kwargs={"net_arch": [16, 16]})
  fix_action_noise(policy, .08)
  before = policy.policy.log_std.detach().clone()
  policy.learn(8)
  torch.testing.assert_close(policy.policy.log_std, before)
  policy.save(tmp_path / "policy")
  loaded = PPO.load(tmp_path / "policy", env=env)
  assert loaded.fixed_action_std == .08
  assert loaded.target_kl == .02
  fix_action_noise(loaded, loaded.fixed_action_std)
  assert not loaded.policy.log_std.requires_grad
  for invalid in (0, -1, float("nan")):
    with pytest.raises(ValueError):
      fix_action_noise(loaded, invalid)


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


def test_incidental_bat_contact_is_failure_not_timeout():
  env = CricketVecEnv(2, forbid_bat_contact=True)
  env.reset()
  floor = env.physics.contact_names.index("bat_blade_floor")
  env.physics.active_samples[0, floor] = 1
  _, _, done, info = env.step(np.zeros((2, 29)))
  assert done.tolist() == [True, False]
  assert info[0]["invalid_bat_contact"]
  assert not info[0]["TimeLimit.truncated"]
  assert not info[0]["fell"]
  assert env.physics.active_samples[0, floor] == 0


def test_stress_audit_rejects_contact_failure_before_running(monkeypatch, tmp_path):
  import cricket_g1_balance_validation as validation

  monkeypatch.setattr(sys, "argv", ["validation", "--directory", str(tmp_path)])
  monkeypatch.setattr(validation.PPO, "load", lambda *args, **kwargs: object())

  def failed_evaluation(policy, hand, task, **kwargs):
    assert kwargs == {"forbid_bat_contact": True}
    return [{"fell": False, "invalid_bat_contact": True, "success": False}] * 8

  monkeypatch.setattr(validation, "evaluate", failed_evaluation)
  with pytest.raises(RuntimeError, match="contact-free short curriculum gate"):
    validation.main()
  assert not (tmp_path / "balance_validation.json").exists()
