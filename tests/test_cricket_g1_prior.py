"""Contract checks for the optional, externally trained G1 locomotion prior."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("yaml")
pytest.importorskip("mujoco_menagerie")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))

from cricket_g1_prior import (
  ASSET_HASHES,
  REVISION,
  SDK_JOINTS,
  SEEDS,
  UnitreePrior,
  prior_model,
  rollout,
  sha256,
)


@pytest.fixture
def prior():
  directory = Path.home() / "Library/Caches/unitree_rl_lab" / REVISION
  if not all((directory / name).exists() for name in ASSET_HASHES):
    pytest.skip("download the pinned optional Unitree policy/config to run the integration audit")
  return UnitreePrior(directory)


def test_hash_guard(tmp_path):
  (tmp_path / "policy.onnx").write_bytes(b"wrong checkpoint")
  with pytest.raises(ValueError, match="unexpected Unitree asset hash"):
    UnitreePrior(tmp_path)


def test_history_order_scale_reset_and_raw_action(prior):
  q = prior.sdk_default.copy()
  q[prior.mapping] += np.arange(29) / 100
  dq, gyro = np.arange(29), np.array([1., 2., 3.])
  command = np.array([.1, -.2, .3])
  obs = prior.observation(q, dq, gyro, np.array([1, 0, 0, 0]), command)
  expected = np.concatenate([np.tile(x, 5) for x in
                             (gyro*.2, [0, 0, -1], command, np.arange(29)/100,
                              dq[prior.mapping]*.05, np.zeros(29))])
  np.testing.assert_allclose(obs[0], expected, atol=1e-7)
  prior.last_action[:] = 2
  obs = prior.observation(q, dq, gyro*2, np.array([1, 0, 0, 0]), command)[0]
  np.testing.assert_allclose(obs[:15].reshape(5, 3), np.vstack([np.tile(gyro*.2, (4, 1)), gyro*.4]))
  np.testing.assert_array_equal(obs[-145:].reshape(5, 29)[-1], np.full(29, 2))
  np.testing.assert_array_equal(obs[-145:].reshape(5, 29)[:-1], np.zeros((4, 29)))
  prior.reset()
  obs = prior.observation(q, dq, gyro*2, np.array([1, 0, 0, 0]), command)[0]
  np.testing.assert_array_equal(obs[-145:], np.zeros(145))


@pytest.mark.parametrize("roll,pitch,yaw", [(0, 10, 0), (0, 10, 60), (-10, 0, 0), (10, 0, 60)])
def test_body_gravity_not_world_upvector(prior, roll, pitch, yaw):
  parts = []
  for axis, angle in zip(np.eye(3), (roll, pitch, yaw), strict=True):
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, axis, np.deg2rad(angle))
    parts.append(q)
  q = np.empty(4)
  mujoco.mju_mulQuat(q, parts[1], parts[0])
  mujoco.mju_mulQuat(q, parts[2], q.copy())
  obs = prior.observation(prior.sdk_default, np.zeros(29), np.zeros(3), q, np.zeros(3))[0]
  r, p = np.deg2rad([roll, pitch])
  expected = [np.sin(p), -np.sin(r)*np.cos(p), -np.cos(r)*np.cos(p)]
  np.testing.assert_allclose(obs[15:30].reshape(5, 3), np.tile(expected, (5, 1)), atol=1e-7)


def test_policy_sdk_mapping_and_no_raw_action_clip(prior):
  action = np.linspace(-2, 2, 29, dtype=np.float32)
  prior.session = SimpleNamespace(run=lambda *args: [action[None]])
  target = prior.target(np.zeros((1, 480), dtype=np.float32))
  np.testing.assert_allclose(target[prior.mapping], prior.default + action * .25)
  np.testing.assert_array_equal(prior.last_action, action)
  assert SDK_JOINTS[prior.mapping[1]] == "right_hip_pitch_joint"
  assert SDK_JOINTS[prior.mapping[12]] == "right_shoulder_pitch_joint"


@pytest.mark.parametrize("hand", ["none", "right", "left"])
def test_native_prior_preserves_unsupported_model_limits(prior, hand):
  model = prior_model(prior, hand)
  assert model.nu == 29 and model.neq == 0
  assert np.all(model.body_gravcomp == 0)
  assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
  np.testing.assert_array_equal(model.actuator_gainprm[:, 0], prior.config["stiffness"])
  np.testing.assert_array_equal(model.actuator_biasprm[:, 2], -np.array(prior.config["damping"]))
  assert np.all(model.jnt_actfrclimited[model.actuator_trnid[:, 0]])
  first = rollout(prior, hand, SEEDS[0], "unitree_onnx", seconds=.04)
  second = rollout(prior, hand, SEEDS[0], "unitree_onnx", seconds=.04)
  assert first == second
  assert first["completed"]
  assert first["maximum_applied_joint_torque_fraction"] <= 1
  assert first["native_serial_parity"] == "exact_qpos_qvel_every_physics_step"


def test_retained_prior_complete_pool_and_provenance():
  root = Path(__file__).parents[1]
  report = json.loads((root / "examples/cricket_g1_results/unitree_prior/evaluation.json").read_text())
  expected = {(hand, controller, seed) for hand in ("none", "right", "left")
              for controller in ("constant_target", "unitree_onnx") for seed in SEEDS}
  assert {(r["hand"], r["controller"], r["seed"]) for r in report["rows"]} == expected
  assert len(report["rows"]) == 48
  assert report["external_asset_sha256"] == ASSET_HASHES
  for name, checksum in report["source_sha256"].items():
    assert sha256(root / "examples" / name) == checksum
  for row in report["rows"]:
    assert row["native_serial_parity"] == "exact_qpos_qvel_every_physics_step"
    if row["controller"] == "unitree_onnx":
      assert row["completed"] and row["seconds"] == 10
      assert not row["nonfoot_ground_contact_geoms"]
      assert row["maximum_applied_joint_torque_fraction"] <= 1
      assert row["maximum_joint_limit_excess_rad"] == 0
      assert not any(v for k, v in row["contact_active_substep_counts"].items() if k.startswith("bat_"))
