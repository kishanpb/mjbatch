# SPDX-License-Identifier: Apache-2.0

"""CPU PPO/A2C curricula for the floating-base G1, using Stable-Baselines3."""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import VecEnv

from cricket_g1 import DECIMATION, TIMESTEP, G1Cricket


class CricketVecEnv(VecEnv):
  def __init__(self, count=32, seed=1, hand="right", task="balance", horizon=150, forbid_bat_contact=False,
               observe_root_height=False):
    self.physics = G1Cricket(count, hand)
    self.task, self.render_mode = task, None
    self.horizon = horizon
    self.forbid_bat_contact = forbid_bat_contact
    self.observe_root_height = observe_root_height
    self.rngs = [np.random.default_rng(seed + i) for i in range(count)]
    self.steps = np.zeros(count, dtype=int)
    self.returns = np.zeros(count)
    self.last_action = np.zeros((count, 29), dtype=np.float32)
    self.hit = np.zeros(count, dtype=bool)
    self.ball = self.physics.batch.joint("cricket_ball_joint")
    self.support_ids = [i for i, n in enumerate(self.physics.contact_names) if n.startswith("support_")]
    self.bat_ids = [self.physics.contact_names.index("ball_" + n) for n in ("bat_blade", "bat_handle")]
    self.external_bat_ids = [i for i, n in enumerate(self.physics.contact_names) if n.startswith("bat_")]
    obs = self.observation()
    super().__init__(count, gym.spaces.Box(-10, 10, obs.shape[1:], dtype=np.float32),
                     gym.spaces.Box(-1, 1, (29,), dtype=np.float32))

  def observation(self):
    env = self.physics
    q = env.qpos[:, :7]
    w, x, y, z = q[:, 3:].T
    up = np.column_stack([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)])
    support = np.stack([env.contacts[i][:, :, 1].sum(1) for i in self.support_ids], axis=1) / 400
    bat_pos = env.batch.site("bat_sweet_spot").xpos
    parts = [env.qpos[:, env.model.jnt_qposadr[env.joints]] - env.home,
             env.qvel[:, env.model.jnt_dofadr[env.joints]] * .1,
             env.qvel[:, :3], env.qvel[:, 3:6] * .2, up, self.last_action,
             self.ball.qpos[:, :3] - q[:, :3], self.ball.qvel[:, :3] * .1,
             bat_pos - q[:, :3], support,
             env.batch.sensor("bat_fixture_force") / 100,
             env.batch.sensor("bat_fixture_torque") / 10]
    if self.observe_root_height:
      parts.append(q[:, 2:3] - .78)
    return np.clip(np.concatenate(parts, axis=1), -10, 10).astype(np.float32)

  def _reset_rows(self, ids):
    env = self.physics
    env.reset(ids, launch=self.task == "batting")
    for i in ids:
      jitter = self.rngs[i].uniform(-.005, .005, 29)
      env.qpos[i, env.model.jnt_qposadr[env.joints]] += jitter
      if self.task == "batting":
        self.ball.qpos[i, 1] += self.rngs[i].uniform(-.03, .03)
        self.ball.qpos[i, 2] += self.rngs[i].uniform(-.03, .03)
    env.batch.forward(ids)
    self.steps[ids], self.returns[ids], self.last_action[ids], self.hit[ids] = 0, 0, 0, False

  def reset(self):
    for i, seed in enumerate(self._seeds):
      if seed is not None:
        self.rngs[i] = np.random.default_rng(seed)
    self._reset_rows(np.arange(self.num_envs))
    self._reset_seeds()
    self._reset_options()
    return self.observation()

  def step_async(self, actions):
    self.actions = np.asarray(actions, dtype=np.float32).copy()

  def step_wait(self):
    env = self.physics
    env.step(self.actions)
    self.steps += 1
    q = env.qpos
    up = 1 - 2 * (q[:, 4]**2 + q[:, 5]**2)
    height = q[:, 2]
    fallen = (height < .50) | (up < .65)
    invalid_bat_contact = self.forbid_bat_contact & (env.active_samples[:, self.external_bat_ids].sum(1) > 0)
    joint = q[:, env.model.jnt_qposadr[env.joints]]
    violation = np.maximum(env.limits[:, 0] - joint, 0) + np.maximum(joint - env.limits[:, 1], 0)
    reward = 2 * np.exp(-20 * (1 - up)**2) + np.exp(-100 * (height - .78)**2)
    reward -= .15 * np.square(env.qvel[:, :2]).sum(1)
    reward -= .01 * np.square(self.actions - self.last_action).sum(1)
    reward -= .01 * np.square(joint - env.home).sum(1) + 5 * violation.sum(1)
    reward -= 5 * (fallen | invalid_bat_contact)
    current_hit = (env.active_samples[:, self.bat_ids].sum(1) > 0) & (self.ball.qvel[:, 0] > 1)
    new_hit = current_hit & ~self.hit
    self.hit |= current_hit
    if self.task == "batting":
      distance = np.linalg.norm(self.ball.qpos[:, :3] - env.batch.site("bat_sweet_spot").xpos, axis=1)
      reward += .5 * np.exp(-distance**2 / .09) + 20 * new_hit
    self.last_action[:] = self.actions
    self.returns += reward
    obs = self.observation()
    done = fallen | invalid_bat_contact | (self.steps >= self.horizon)
    infos = [{} for _ in range(self.num_envs)]
    ids = np.flatnonzero(done)
    for i in ids:
      infos[i] = {"terminal_observation": obs[i].copy(), "TimeLimit.truncated": not bool(fallen[i] or invalid_bat_contact[i]),
                  "episode": {"r": float(self.returns[i]), "l": int(self.steps[i])},
                  "fell": bool(fallen[i]), "invalid_bat_contact": bool(invalid_bat_contact[i]),
                  "bat_contact_outgoing": bool(self.hit[i])}
    if len(ids):
      self._reset_rows(ids)
      obs[ids] = self.observation()[ids]
    return obs, reward.astype(np.float32), done, infos

  def close(self):
    pass  # Batch owns native memory and releases it with its Python object.

  def get_attr(self, name, indices=None):
    return [getattr(self, name) for _ in self._get_indices(indices)]

  def set_attr(self, name, value, indices=None):
    raise NotImplementedError("per-row attribute mutation is not supported")

  def env_method(self, method_name, *args, indices=None, **kwargs):
    raise NotImplementedError("use the native vector environment methods directly")

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False for _ in self._get_indices(indices)]


def evaluate(policy, hand, task, horizon=150, seed=9001, forbid_bat_contact=False,
             observe_root_height=None):
  if observe_root_height is None:
    observe_root_height = getattr(policy, "observe_root_height", False)
  env = CricketVecEnv(count=8, seed=seed, hand=hand, task=task, horizon=horizon,
                      forbid_bat_contact=forbid_bat_contact, observe_root_height=observe_root_height)
  obs = env.reset()
  rows, finished = [], np.zeros(8, dtype=bool)
  for _ in range(horizon):
    actions = np.zeros((8, 29)) if policy is None else policy.predict(obs, deterministic=True)[0]
    obs, _, done, infos = env.step(actions)
    for i in np.flatnonzero(done & ~finished):
      info = infos[i]
      rows.append({"seed": seed + int(i), "steps": info["episode"]["l"],
                   "seconds": info["episode"]["l"] * TIMESTEP * DECIMATION,
                   "return": info["episode"]["r"], "fell": info["fell"],
                   "invalid_bat_contact": info["invalid_bat_contact"],
                   "success": not (info["fell"] or info["invalid_bat_contact"]),
                   "bat_contact_outgoing": info["bat_contact_outgoing"]})
      finished[i] = True
    if finished.all():
      break
  if not finished.all():
    raise RuntimeError("evaluation did not complete all declared seeds")
  return sorted(rows, key=lambda row: row["seed"])


def fix_action_noise(model, std):
  if not np.isfinite(std) or std <= 0:
    raise ValueError("fixed action standard deviation must be finite and positive")
  with torch.no_grad():
    model.policy.log_std.fill_(np.log(std))
  model.policy.log_std.requires_grad_(False)
  model.fixed_action_std = std


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--algorithm", choices=("ppo", "a2c"), default="ppo")
  parser.add_argument("--task", choices=("balance", "batting"), default="balance")
  parser.add_argument("--hand", choices=("right", "left"), default="right")
  parser.add_argument("--steps", type=int, default=32768)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--envs", type=int, default=32)
  parser.add_argument("--load", type=Path)
  parser.add_argument("--fixed-action-std", type=float)
  parser.add_argument("--target-kl", type=float)
  parser.add_argument("--learning-rate", type=float)
  parser.add_argument("--forbid-bat-contact", action="store_true")
  parser.add_argument("--observe-root-height", action="store_true")
  args = parser.parse_args()
  if args.target_kl is not None and (args.algorithm != "ppo" or not np.isfinite(args.target_kl) or args.target_kl <= 0):
    parser.error("--target-kl requires PPO and a finite positive value")
  if args.learning_rate is not None and (not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
    parser.error("--learning-rate must be finite and positive")
  args.output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(2)
  cls = PPO if args.algorithm == "ppo" else A2C
  model = cls.load(args.load, device="cpu") if args.load else None
  observe_root_height = args.observe_root_height or getattr(model, "observe_root_height", False)
  env = CricketVecEnv(args.envs, args.seed, args.hand, args.task, observe_root_height=observe_root_height)
  if args.load:
    model.set_env(env)
  else:
    kwargs = dict(n_steps=64, learning_rate=3e-4, ent_coef=.005, gamma=.99,
                  seed=args.seed, device="cpu", verbose=1,
                  policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.))
    if args.algorithm == "ppo":
      kwargs.update(batch_size=256, n_epochs=5)
    model = cls("MlpPolicy", env, **kwargs)
  parent = None if args.load is None else {
    "sha256": hashlib.sha256(args.load.read_bytes()).hexdigest(),
    "timesteps": model.num_timesteps,
  }
  std = args.fixed_action_std if args.fixed_action_std is not None else getattr(model, "fixed_action_std", None)
  if std is not None:
    fix_action_noise(model, std)
  if args.target_kl is not None:
    model.target_kl = args.target_kl
  if args.learning_rate is not None:
    model.learning_rate = args.learning_rate
    model.lr_schedule = FloatSchedule(args.learning_rate)
  env.forbid_bat_contact = args.forbid_bat_contact or getattr(model, "forbid_bat_contact", False)
  model.forbid_bat_contact = env.forbid_bat_contact
  model.observe_root_height = observe_root_height
  model.verbose = 0
  start = time.monotonic()
  model.set_logger(configure(str(args.output), ["csv"]))
  model.learn(total_timesteps=args.steps, reset_num_timesteps=args.load is None)
  model.save(args.output / "policy.zip")
  report = {"task": args.task, "hand": args.hand, "algorithm": args.algorithm,
            "training_seed": args.seed, "timesteps": model.num_timesteps,
            "parent_checkpoint": parent,
            "runtime_versions": {name: version(name) for name in
                                 ("mujoco", "mujoco-menagerie", "stable-baselines3", "torch", "numpy")},
            "training_config": {"envs": args.envs, "requested_additional_steps": args.steps,
                                "n_steps": model.n_steps, "learning_rate": model.learning_rate,
                                "gamma": model.gamma, "ent_coef": model.ent_coef,
                                "fixed_action_std": std,
                                "target_kl": getattr(model, "target_kl", None),
                                "forbid_bat_contact": env.forbid_bat_contact,
                                "observe_root_height": observe_root_height,
                                "policy_kwargs": model.policy_kwargs},
            "evaluation_scope": "Fixed development seeds 9001-9008; not an untouched final test set",
            "seconds": time.monotonic() - start, "provenance": env.physics.provenance,
            "observation_contract": "simulator proprioception, privileged ball state, geometry-level loads",
            "observation_version": "height_v3_118" if observe_root_height else "original_117",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), Path(__file__).with_name("cricket_g1.py"))},
            "baseline": evaluate(None, args.hand, args.task, forbid_bat_contact=env.forbid_bat_contact,
                                 observe_root_height=observe_root_height),
            "trained": evaluate(model, args.hand, args.task, forbid_bat_contact=env.forbid_bat_contact)}
  (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(json.dumps({"baseline_falls": sum(r["fell"] for r in report["baseline"]),
                    "trained_falls": sum(r["fell"] for r in report["trained"]),
                    "trained_invalid_bat_contacts": sum(r["invalid_bat_contact"] for r in report["trained"]),
                    "trained_successes": sum(r["success"] for r in report["trained"])}))


if __name__ == "__main__":
  main()
