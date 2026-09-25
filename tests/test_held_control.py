"""Native Batch intervals match the official rollout state and sensor phases."""

import copy

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
@pytest.mark.parametrize("group_identical_models", [False, True])
@pytest.mark.parametrize("num_threads", [1, 8])
@pytest.mark.parametrize("selection", ["all", "subset", "empty"])
def test_all_substeps_with_variants_wrench_and_partial_reset(
  nstep, group_identical_models, num_threads, selection
):
  first = mujoco.MjModel.from_xml_string(XML)
  second = mujoco.MjModel.from_xml_string(XML)
  second.body_mass[2] *= 1.7
  mujoco.mj_setConst(second, mujoco.MjData(second))
  models = [first, second, copy.copy(first), first, copy.copy(second)]
  initial = np.empty((len(models), mujoco.mj_stateSize(first, FULL)))
  for row, model in enumerate(models):
    data = mujoco.MjData(model)
    data.qvel[1] = -2.5 - 0.1 * row
    mujoco.mj_getState(model, data, initial[row], FULL)
  reset = initial.copy()
  batch = HeldControlRollout(models, num_threads=num_threads, group_identical_models=group_identical_models)
  assert [g[0].tolist() for g in batch.groups] == (
    [[0, 2, 3], [1, 4]] if group_identical_models else [[0, 3], [1], [2], [4]]
  )
  official = Rollout(nthread=1)
  scratch = mujoco.MjData(first)
  scratch.qacc_warmstart[:] = 123
  scratch.ctrl[:] = 91
  scratch.xfrc_applied[:] = 37
  columns = {
    "all": None,
    "subset": np.array([first.nsensordata - 1, 0, 2, 0]),
    "empty": np.empty(0, dtype=int),
  }[selection]
  contact_seen = False
  try:
    for tick in range(4):
      spec = int(mujoco.mjtState.mjSTATE_CTRL)
      if tick == 1:
        spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
      if tick == 2:
        initial[1] = reset[1]
      control = np.zeros((len(models), nstep, mujoco.mj_stateSize(first, spec)))
      control[:, :, 0] = 0.01
      if tick == 1:
        control[:, :, 1:] = 0.03
      expected = official.rollout(models, [scratch], initial, control, control_spec=spec, nstep=nstep)
      actual = batch.rollout(
        models, [scratch], initial, control, control_spec=spec, nstep=nstep, sensor_indices=columns
      )
      np.testing.assert_array_equal(batch.final_sensors, expected[1][:, -1])
      selected = expected if columns is None else (expected[0], expected[1][:, :, columns])
      for observed, reference in zip(actual, selected, strict=True):
        assert observed.dtype == mujoco.MJTNUM_DTYPE
        np.testing.assert_array_equal(observed, reference)
      contact_seen |= bool(np.any(expected[1][:, :, 0] > 0))
      initial = expected[0][:, -1].astype(np.float32)
    assert contact_seen
  finally:
    official.close()
    batch.close()
    assert batch.final_sensors is None


@pytest.mark.parametrize("columns", [[-1], [71], [0.5], [[0]], [True]])
def test_invalid_sensor_columns_clear_final_cache(columns):
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)
  initial = np.empty((1, mujoco.mj_stateSize(model, FULL)))
  mujoco.mj_getState(model, data, initial[0], FULL)
  recorder = HeldControlRollout([model])
  control = np.zeros((1, 1, model.nu))
  kwargs = dict(control_spec=int(mujoco.mjtState.mjSTATE_CTRL), nstep=1)
  try:
    recorder.rollout([model], [data], initial, control, **kwargs)
    assert recorder.final_sensors is not None
    with pytest.raises(ValueError, match="sensor_indices"):
      recorder.rollout([model], [data], initial, control, sensor_indices=columns, **kwargs)
    assert recorder.final_sensors is None
  finally:
    recorder.close()


def test_compiled_grouping_keeps_same_shape_option_and_contact_variants_separate():
  first = mujoco.MjModel.from_xml_string(XML)
  timestep, friction, actuator = (copy.copy(first) for _ in range(3))
  timestep.opt.timestep *= 2
  friction.pair_friction[0, 0] *= 2
  actuator.actuator_gainprm[0, 0] *= 2
  models = [first, timestep, friction, actuator, copy.copy(first)]
  recorder = HeldControlRollout(models, group_identical_models=True)
  try:
    assert [group[0].tolist() for group in recorder.groups] == [[0, 4], [1], [2], [3]]
  finally:
    recorder.close()


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


@pytest.mark.parametrize("group_identical_models", [False, True])
def test_divergence_fails_before_auto_reset_can_look_healthy(group_identical_models):
  model = mujoco.MjModel.from_xml_string(XML)
  data = mujoco.MjData(model)
  initial = np.empty((2, mujoco.mj_stateSize(model, FULL)))
  for row in initial:
    mujoco.mj_getState(model, data, row, FULL)
  invalid = initial.copy()
  invalid[1, 1] = np.nan
  control = np.zeros((2, 4, model.nu))
  kwargs = dict(control_spec=int(mujoco.mjtState.mjSTATE_CTRL), nstep=4)
  models = [model, copy.copy(model)]
  recorder = HeldControlRollout(models, num_threads=2, group_identical_models=group_identical_models)
  official = Rollout(nthread=1)
  try:
    with pytest.raises(RuntimeError, match="mjWARN_BADQPOS in row 1 at substep 0"):
      recorder.rollout(models, [data], invalid, control, **kwargs)
    actual = recorder.rollout(models, [data], initial, control, **kwargs)
    expected = official.rollout(models, [data], initial, control, **kwargs)
    for observed, reference in zip(actual, expected, strict=True):
      np.testing.assert_array_equal(observed, reference)
    np.testing.assert_array_equal(recorder.final_sensors, expected[1][:, -1])
    with pytest.raises(RuntimeError, match="mjWARN_BADQPOS"):
      recorder.rollout(models, [data], invalid, control, sensor_indices=np.array([0]), **kwargs)
    assert recorder.final_sensors is None
  finally:
    recorder.close()
    official.close()


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("group_identical_models", [False, True])
def test_moving_holder_release_persists_and_reset_is_per_environment(side, group_identical_models):
  model = mujoco.MjModel.from_xml_string(f"""
  <mujoco><option timestep=".001" integrator="implicitfast"/>
    <worldbody>
      <body name="wrist" pos="0 {side * 0.08} 1.2">
        <joint type="slide" axis="1 0 0"/>
        <geom type="box" size=".04 .03 .03" mass=".7" contype="0" conaffinity="0"/>
      </body>
      <body name="ball" pos="0 {side * 0.08} 1.11"><freejoint/>
        <geom type="sphere" size=".036" mass=".156"/>
      </body>
    </worldbody>
    <equality><weld name="holder" body1="wrist" body2="ball" solref=".004 1"/></equality>
    <sensor><framelinvel name="velocity" objtype="body" objname="ball"/></sensor>
  </mujoco>
  """)
  data = mujoco.MjData(model)
  data.qvel[[0, 1]] = 1.2
  reset = np.empty(mujoco.mj_stateSize(model, FULL))
  mujoco.mj_getState(model, data, reset, FULL)
  initial = np.tile(reset, (3, 1))
  active = np.ones(3)
  models = [copy.copy(model) for _ in range(3)]
  native = HeldControlRollout(models, num_threads=2, group_identical_models=group_identical_models)
  official = Rollout(nthread=1)
  spec = int(mujoco.mjtState.mjSTATE_EQ_ACTIVE | mujoco.mjtState.mjSTATE_XFRC_APPLIED)
  try:
    for tick in range(6):
      if tick == 1:
        active[1:] = 0
      if tick == 3:
        initial[1] = reset
        active[1] = 1
      control = np.zeros((3, 40, mujoco.mj_stateSize(model, spec)))
      for row in range(3):
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, initial[row], FULL)
        before = np.concatenate((data.qpos, data.qvel))
        data.eq_active[:] = active[row]
        data.xfrc_applied[1, 0] = 0.1
        np.testing.assert_array_equal(np.concatenate((data.qpos, data.qvel)), before)
        mujoco.mj_getState(model, data, control[row, 0], spec)
        control[row] = control[row, 0]
      actual = native.rollout(models, [data], initial, control, control_spec=spec, nstep=40)
      expected = official.rollout(models, [data], initial, control, control_spec=spec, nstep=40)
      for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed, reference)
      for row in range(3):
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, initial[row], FULL)
        mujoco.mj_setState(model, data, control[row, 0], spec)
        for step in range(40):
          mujoco.mj_step(model, data)
          state = np.empty_like(reset)
          mujoco.mj_getState(model, data, state, FULL)
          np.testing.assert_array_equal(actual[0][row, step], state)
          np.testing.assert_array_equal(actual[1][row, step], data.sensordata)
          if not active[row]:
            assert data.nefc == 0
            np.testing.assert_array_equal(data.qfrc_constraint[1:], 0)
        if active[row]:
          assert abs(data.qpos[3] - 1.11) < 0.001
        else:
          assert data.qvel[3] < -0.3
          assert data.qvel[1] > 1
      initial = actual[0][:, -1].copy()
  finally:
    native.close()
    official.close()


def test_omitting_equality_input_restores_model_defaults():
  model = mujoco.MjModel.from_xml_string("""
  <mujoco><worldbody><body><joint name="slide" type="slide"/>
    <geom type="sphere" size=".05" mass="1"/>
  </body></worldbody><equality>
    <joint name="held" joint1="slide"/>
    <joint name="unused" joint1="slide" active="false"/>
  </equality></mujoco>
  """)
  data = mujoco.MjData(model)
  initial = np.empty((1, mujoco.mj_stateSize(model, FULL)))
  mujoco.mj_getState(model, data, initial[0], FULL)
  recorder = HeldControlRollout([model])
  official = Rollout(nthread=1)
  try:
    released = recorder.rollout(
      [model],
      [data],
      initial,
      np.zeros((1, 5, 2)),
      control_spec=int(mujoco.mjtState.mjSTATE_EQ_ACTIVE),
      nstep=5,
    )
    control = np.zeros((1, 5, 0))
    actual = recorder.rollout([model], [data], initial, control, control_spec=0, nstep=5)
    expected = official.rollout([model], [data], initial, control, control_spec=0, nstep=5)
    for observed, reference in zip(actual, expected, strict=True):
      np.testing.assert_array_equal(observed, reference)
    assert not np.array_equal(actual[0], released[0])
  finally:
    recorder.close()
    official.close()
