"""Optional G1 asset and floating-base native batch checks."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_menagerie")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
import mujoco_menagerie as mm

from cricket_g1 import DECIMATION, G1Cricket


@pytest.mark.parametrize("hand", ["right", "left"])
def test_robot_contract_and_native_serial_parity(hand, monkeypatch):
  env = G1Cricket(count=2, hand=hand)
  model = env.model
  assert model.nu == 29
  assert (model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).sum() == 2
  assert model.neq == 0
  assert model.body("cricket_bat").mass[0] == pytest.approx(1.12)
  assert model.body("cricket_ball").mass[0] == pytest.approx(0.156)
  assert np.all(model.jnt_actfrclimited[env.joints])
  assert np.all(model.body_gravcomp == 0)
  assert model.sensor("bat_fixture_force").dim[0] == 3
  env.reset(launch=True)
  refs = [mujoco.MjData(model) for _ in range(2)]
  for data in refs:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.joint("cricket_ball_joint").qvel[0] = -8
    mujoco.mj_forward(model, data)
  action = np.random.default_rng(17).uniform(-0.1, 0.1, (2, 29))
  for _ in range(10):
    with monkeypatch.context() as context:
      context.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Python integration fallback"))
      env.step(action)
    for row, data in enumerate(refs):
      data.ctrl[:] = env.ctrl[row]
      for _ in range(DECIMATION):
        mujoco.mj_step(model, data)
      np.testing.assert_array_equal(data.qpos, env.qpos[row])
      np.testing.assert_array_equal(data.qvel, env.qvel[row])
      np.testing.assert_array_equal(data.sensordata, env.batch.bind("sensordata")[row])
  retained = env.qpos[1].copy()
  env.reset(np.array([0]))
  np.testing.assert_array_equal(env.qpos[1], retained)
  assert env.peak_load[0].sum() == 0
  assert env.peak_load[1].sum() > 0


def test_no_actuation_exposes_free_base_gravity():
  env = G1Cricket()
  initial = env.qpos[0, 2]
  env.batch.expand("disableflags")[:] = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
  env.batch.step(nstep=300)
  assert env.qpos[0, 2] < initial - 0.15


def test_robot_inertias_and_motor_limits_match_source():
  env = G1Cricket()
  original = mm.get("unitree_g1").model("scene_mjx")
  for index in range(1, original.nbody):
    source = original.body(index)
    target = env.model.body(source.name)
    for field in ("mass", "inertia", "ipos", "iquat"):
      np.testing.assert_array_equal(getattr(target, field), getattr(source, field))
  for index in range(original.nu):
    source = original.actuator(index)
    target = env.model.actuator(source.name)
    for field in ("gainprm", "biasprm", "ctrlrange", "forcerange"):
      np.testing.assert_array_equal(getattr(target, field), getattr(source, field))


def test_reset_only_incoming_ball_produces_measured_bat_contact():
  env = G1Cricket()
  ball = env.batch.joint("cricket_ball_joint")
  bat = env.batch.site("bat_sweet_spot")
  env.batch.forward()
  ball.qpos[0, :3] = bat.xpos[0] + [.038, 0, 0]
  ball.qvel[0, :3] = [-2, 0, 0]
  env.batch.forward()
  for _ in range(3):
    env.step(np.zeros((1, 29)))
  sensor = env.contact_names.index("ball_bat_blade")
  assert env.active_samples[0, sensor] > 0
  assert env.peak_load[0, sensor] > 0
  assert np.isfinite(env.peak_load).all()
