"""Full substep trajectories for fixed-model, held-control MuJoCo intervals."""

from collections.abc import Sequence

import mujoco
import numpy as np

from mjbatch import Batch

FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION
CONTROLS = int(
  mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED | mujoco.mjtState.mjSTATE_EQ_ACTIVE
)


class HeldControlRollout:
  """A narrow rollout adapter, not a replacement for mujoco.rollout.Rollout.

  Models are copied at construction and must stay fixed. Each call starts from
  FULLPHYSICS with reset auxiliary state, then preserves warmstart within the
  interval. Sensors are mj_step's solved-phase values, without an extra forward.
  Models are grouped by identity unless group_identical_models is enabled,
  which groups byte-identical compiled models. Groups step sequentially.
  EQ_ACTIVE is explicit held
  input: callers must supply it each interval to preserve released constraints.
  """

  def __init__(
    self,
    models: Sequence[mujoco.MjModel],
    num_threads: int = 1,
    *,
    group_identical_models: bool = False,
  ):
    if not models:
      raise ValueError("at least one model is required")
    self.models = tuple(models)
    self.nstate = mujoco.mj_stateSize(models[0], FULL)
    self.nsensor = models[0].nsensordata
    ncontrol = mujoco.mj_stateSize(models[0], CONTROLS)
    if any(
      mujoco.mj_stateSize(m, FULL) != self.nstate
      or m.nsensordata != self.nsensor
      or mujoco.mj_stateSize(m, CONTROLS) != ncontrol
      for m in models
    ):
      raise ValueError("models must have matching state, sensor and control dimensions")
    grouped = {}
    for row, model in enumerate(models):
      if group_identical_models:
        compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=compiled)
        key = compiled.tobytes()
      else:
        key = id(model)
      grouped.setdefault(key, []).append(row)
    self.groups = []
    for rows in grouped.values():
      indices = np.array(rows)
      model = models[indices[0]]
      batch = Batch(model, len(indices), num_threads=num_threads, forward=False)
      self.groups.append(
        (
          indices,
          model,
          batch,
          batch.bind("state"),
          batch.bind("sensordata"),
          batch.bind("warning"),
          mujoco.MjData(model),
        )
      )

  def rollout(self, model, data, initial_state, control, *, control_spec, nstep, chunk_size=None):
    if not self.groups:
      raise RuntimeError("recorder is closed")
    if len(model) != len(self.models) or any(a is not b for a, b in zip(model, self.models, strict=True)):
      raise ValueError("recorder models differ from construction")
    if control_spec & ~CONTROLS:
      raise ValueError("only CTRL, XFRC_APPLIED and EQ_ACTIVE control fields are supported")
    width = mujoco.mj_stateSize(self.models[0], control_spec)
    expected = (len(self.models), nstep, width)
    if nstep < 1 or np.shape(control) != expected or np.shape(initial_state) != (len(model), self.nstate):
      raise ValueError("invalid held-control interval shape")
    if not np.array_equal(control, np.broadcast_to(control[:, :1], expected)):
      raise ValueError("control must be held throughout the interval")
    states = np.empty((len(model), nstep, self.nstate), dtype=mujoco.MJTNUM_DTYPE)
    sensors = np.empty((len(model), nstep, self.nsensor), dtype=mujoco.MJTNUM_DTYPE)
    for indices, source, batch, integration, sensed, warning, scratch in self.groups:
      warning[:] = 0
      for local, row in enumerate(indices):
        mujoco.mj_resetData(source, scratch)
        mujoco.mj_setState(source, scratch, initial_state[row], FULL)
        mujoco.mj_setState(source, scratch, control[row, 0], control_spec)
        mujoco.mj_getState(source, scratch, integration[local], INTEGRATION)
      for step in range(nstep):
        batch.step()
        failed = np.argwhere(warning[:, :, 1] > 0)
        if len(failed):
          local, code = failed[0]
          raise RuntimeError(
            f"MuJoCo warning {mujoco.mjtWarning(int(code)).name} in row {indices[local]} at substep {step}"
          )
        sensors[indices, step] = sensed
        for local, row in enumerate(indices):
          mujoco.mj_setState(source, scratch, integration[local], INTEGRATION)
          mujoco.mj_getState(source, scratch, states[row, step], FULL)
    return states, sensors

  def close(self):
    self.groups.clear()
