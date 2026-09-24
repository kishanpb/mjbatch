"""Explicit wicket collisions must be physical, observable, and terminal."""

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_menagerie")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from cricket_g1 import CONTACT_WIDTH, SLOTS, scene
from mjbatch import Batch

WICKETS = [f"{end}_stump_{i}" for end in ("striker", "bowler") for i in range(3)]
WICKETS += [f"{end}_bail" for end in ("striker", "bowler")]


@pytest.mark.parametrize("hand", ["right", "left"])
def test_all_collision_geometries_have_pairs_and_sensors(hand):
  model, provenance = scene(hand)
  assert provenance["collision_contract"] == "v2_ball_bat_robot_vs_both_wickets_and_bails"
  colliders = [model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.endswith("_collision")]
  pairs = {frozenset((int(a), int(b))) for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)}
  for geom in ["cricket_ball_geom", "bat_handle", "bat_blade"] + colliders:
    for wicket in WICKETS:
      assert frozenset((model.geom(geom).id, model.geom(wicket).id)) in pairs
      name = f"ball_{wicket}" if geom == "cricket_ball_geom" else f"wicket_{geom}_{wicket}"
      assert model.sensor(name).dim[0] == SLOTS * CONTACT_WIDTH


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("target", ["bat_handle", "bat_blade", "robot"])
@pytest.mark.parametrize("wicket", WICKETS)
def test_injected_overlap_has_native_serial_contact_and_force(hand, target, wicket):
  model, _ = scene(hand)
  if target == "robot":
    target = next(model.geom(i).name for i in range(model.ngeom) if model.geom(i).name.endswith("_collision"))
  serial = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, serial, 0)
  mujoco.mj_forward(model, serial)
  serial.qpos[:3] += serial.geom(wicket).xpos - serial.geom(target).xpos
  batch = Batch(model, 1, num_threads=1)
  name = f"wicket_{target}_{wicket}"
  native = batch.sensor(name).reshape(SLOTS, CONTACT_WIDTH)
  batch.reset(keyframe=0)
  batch.bind("qpos")[0] = serial.qpos
  batch.forward()
  mujoco.mj_forward(model, serial)
  batch.step()
  mujoco.mj_step(model, serial)
  np.testing.assert_array_equal(native.ravel(), serial.sensor(name).data)
  np.testing.assert_array_equal(batch.bind("qpos")[0], serial.qpos)
  assert (native[:, 0] > 0).any()
  assert native[:, 1].sum() > 0
  assert np.isfinite(native).all()


def test_wicket_presence_fails_even_when_optional_bat_guard_is_off(monkeypatch):
  pytest.importorskip("stable_baselines3")
  from cricket_g1_train import CricketVecEnv

  env = CricketVecEnv(count=2, forbid_bat_contact=False)
  env.reset()
  channel = env.physics.contact_names.index("wicket_bat_blade_striker_stump_1")
  env.physics.active_samples[0, channel] = 1
  monkeypatch.setattr(env.physics, "step", lambda _: None)
  _, _, done, info = env.step(np.zeros((2, 29)))
  assert done.tolist() == [True, False]
  assert info[0]["invalid_wicket_contact"]
  assert not info[0]["invalid_bat_contact"] and not info[0]["TimeLimit.truncated"]


def test_wicket_complete_stance_report_covers_all_rows_and_current_sources():
  root = Path(__file__).parents[1]
  report = json.loads((root / "examples/cricket_g1_results/unitree_prior_wickets_v2/evaluation.json").read_text())
  old = json.loads((root / "examples/cricket_g1_results/unitree_prior/evaluation.json").read_text())
  assert len(report["rows"]) == 48
  assert report["contract"]["wicket_contact_guard"].startswith("v2 explicit")
  for name, expected in report["source_sha256"].items():
    assert hashlib.sha256((root / "examples" / name).read_bytes()).hexdigest() == expected
  for row, parent in zip(report["rows"], old["rows"], strict=True):
    assert all(row[k] == parent[k] for k in ("hand", "controller", "seed"))
    assert row["native_serial_parity"] == "exact_qpos_qvel_every_physics_step"
    if row["hand"] == "none":
      assert row == parent
    else:
      wicket = {k: v for k, v in row["contact_active_substep_counts"].items() if k.startswith("wicket_")}
      assert len(wicket) == 232
    if row["controller"] == "unitree_onnx":
      assert row["completed"] and row["seconds"] == 10 and row["failure"] is None
      assert not row["nonfoot_ground_contact_geoms"]
      assert row["maximum_joint_limit_excess_rad"] == 0
      assert row["maximum_applied_joint_torque_fraction"] <= 1
      assert not any(v for k, v in row["contact_active_substep_counts"].items() if k.startswith(("bat_", "wicket_")))
    else:
      assert not row["completed"]
