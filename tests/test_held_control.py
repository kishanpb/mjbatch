"""Native Batch intervals match the official rollout state and sensor phases."""

import mujoco
import numpy as np
import pytest
from mujoco.rollout import Rollout

from mjbatch.held_control import FULL, HeldControlRollout

XML = """
<mujoco><option integrator="implicitfast" timestep=".00025" gravity="0 0 0"/>
  <worldbody>
    <body name="blade"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="blade" type="box" size=".02 .2 .2" mass=".7"/>
    </body>
    <body name="ball" pos=".055 0 0"><freejoint/>
      <geom name="ball_geom" type="sphere" size=".036" mass=".156"/>
    </body>
  </worldbody>
  <actuator><position joint="slide" kp="30" kv="2"/></actuator>
  <contact><pair geom1="blade" geom2="ball_geom" solref=".004 1"/></contact>
  <sensor><contact name="hit" geom1="ball_geom" geom2="blade" num="4"
    data="found force torque dist pos normal tangent"/>
    <framelinvel name="velocity" objtype="body" objname="ball"/>
  </sensor>
</mujoco>
"""


@pytest.mark.parametrize("nstep", [1, 80, 160])
def test_all_substeps_with_variants_wrench_and_partial_reset(nstep):
  first = mujoco.MjModel.from_xml_string(XML)
  second = mujoco.MjModel.from_xml_string(XML)
  second.body_mass[2] *= 1.7
  mujoco.mj_setConst(second, mujoco.MjData(second))
  models = [first, first, second]
  initial = np.empty((3, mujoco.mj_stateSize(first, FULL)))
  for row, model in enumerate(models):
    data = mujoco.MjData(model)
    data.qvel[1] = -2.5
    mujoco.mj_getState(model, data, initial[row], FULL)
  reset = initial.copy()
  batch = HeldControlRollout(models, num_threads=2)
  official = Rollout(nthread=1)
  scratch = mujoco.MjData(first)
  scratch.qacc_warmstart[:] = 123
  scratch.ctrl[:] = 91
  scratch.xfrc_applied[:] = 37
  contact_seen = False
  try:
    for tick in range(4):
      spec = int(mujoco.mjtState.mjSTATE_CTRL)
      if tick == 1:
        spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
      if tick == 2:
        initial[1] = reset[1]
      control = np.zeros((3, nstep, mujoco.mj_stateSize(first, spec)))
      control[:, :, 0] = 0.01
      if tick == 1:
        control[:, :, 1:] = 0.03
      expected = official.rollout(models, [scratch], initial, control, control_spec=spec, nstep=nstep)
      actual = batch.rollout(models, [scratch], initial, control, control_spec=spec, nstep=nstep)
      for observed, reference in zip(actual, expected, strict=True):
        assert observed.dtype == mujoco.MJTNUM_DTYPE
        np.testing.assert_array_equal(observed, reference)
      contact_seen |= bool(np.any(actual[1][:, :, 0] > 0))
      initial = expected[0][:, -1].astype(np.float32)
    assert contact_seen
  finally:
    official.close()
    batch.close()


def test_rejects_other_control_contracts_and_closed_recorder():
  model = mujoco.MjModel.from_xml_string(XML)
  recorder = HeldControlRollout([model])
  initial = np.zeros((1, mujoco.mj_stateSize(model, FULL)))
  control = np.zeros((1, 2, 1))
  control[0, 1] = 1
  with pytest.raises(ValueError, match="held throughout"):
    recorder.rollout([model], [], initial, control, control_spec=int(mujoco.mjtState.mjSTATE_CTRL), nstep=2)
  with pytest.raises(ValueError, match="only CTRL"):
    recorder.rollout(
      [model], [], initial, control, control_spec=int(mujoco.mjtState.mjSTATE_QFRC_APPLIED), nstep=2
    )
  recorder.close()
  recorder.close()
  with pytest.raises(RuntimeError, match="closed"):
    recorder.rollout([model], [], initial, control, control_spec=0, nstep=2)


def test_divergence_fails_before_auto_reset_can_look_healthy():
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)
  initial = np.empty((2, mujoco.mj_stateSize(model, FULL)))
  for row in initial:
    mujoco.mj_getState(model, data, row, FULL)
  invalid = initial.copy()
  invalid[1, 1] = np.nan
  control = np.zeros((2, 4, model.nu))
  kwargs = dict(control_spec=int(mujoco.mjtState.mjSTATE_CTRL), nstep=4)
  recorder = HeldControlRollout([model, model], num_threads=2)
  official = Rollout(nthread=1)
  try:
    with pytest.raises(RuntimeError, match="mjWARN_BADQPOS in row 1 at substep 0"):
      recorder.rollout([model, model], [data], invalid, control, **kwargs)
    actual = recorder.rollout([model, model], [data], initial, control, **kwargs)
    expected = official.rollout([model, model], [data], initial, control, **kwargs)
    for observed, reference in zip(actual, expected, strict=True):
      np.testing.assert_array_equal(observed, reference)
  finally:
    recorder.close()
    official.close()
