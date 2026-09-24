# SPDX-License-Identifier: Apache-2.0

"""Audit an externally trained Unitree locomotion prior, not a cricket policy."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
import onnxruntime as ort
import yaml

from cricket_g1 import CONTACT_WIDTH, ROBOT_TREE, SLOTS, scene
from mjbatch import Batch

REVISION = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
ASSET_HASHES = {
  "policy.onnx": "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f",
  "deploy.yaml": "64b04c0596a7010f39f8ac6e9ec46dc750141063cdcf2d42c83ecc444e57bc63",
}
SDK_JOINTS = (
  [f"{side}_{joint}_joint" for side in ("left", "right") for joint in
   ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
  + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
  + [f"{side}_{joint}_joint" for side in ("left", "right") for joint in
     ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
SEEDS = tuple(range(4201, 4209))


def sha256(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


class UnitreePrior:
  def __init__(self, directory):
    for name, expected in ASSET_HASHES.items():
      if sha256(directory / name) != expected:
        raise ValueError(f"unexpected Unitree asset hash: {name}")
    self.config = yaml.safe_load((directory / "deploy.yaml").read_text())
    self.mapping = np.array(self.config["joint_ids_map"])
    self.default = np.array(self.config["default_joint_pos"])
    self.sdk_default = np.empty(29)
    self.sdk_default[self.mapping] = self.default
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(str(directory / "policy.onnx"), options,
                                       providers=["CPUExecutionProvider"])
    if self.session.get_inputs()[0].shape != [1, 480] or self.session.get_outputs()[0].shape != [1, 29]:
      raise ValueError("unexpected Unitree network shape")
    self.history = None
    self.last_action = np.zeros(29)

  def reset(self):
    self.history = None
    self.last_action[:] = 0

  def observation(self, q, dq, gyro, quat, command):
    w, x, y, z = quat
    gravity = np.array([2 * (w*y - x*z), -2 * (y*z + w*x), 2 * (x*x + y*y) - 1])
    terms = [gyro, gravity, command, q[self.mapping] - self.default,
             dq[self.mapping], self.last_action]
    terms = [np.asarray(term, dtype=np.float32) * cfg["scale"]
             for term, cfg in zip(terms, self.config["observations"].values(), strict=True)]
    if self.history is None:
      self.history = [np.tile(term, (5, 1)) for term in terms]
    else:
      for history, term in zip(self.history, terms, strict=True):
        history[:-1] = history[1:]
        history[-1] = term
    return np.concatenate([history.ravel() for history in self.history]).astype(np.float32)[None]

  def target(self, obs):
    self.last_action = self.session.run(["actions"], {"obs": obs})[0][0]
    if not np.isfinite(self.last_action).all():
      raise RuntimeError("non-finite Unitree action")
    action = self.config["actions"]["JointPositionAction"]
    target = np.empty(29)
    target[self.mapping] = self.last_action * action["scale"] + action["offset"]
    return target


def prior_model(prior, hand):
  robot = mm.get("unitree_g1")
  if robot.oid != ROBOT_TREE:
    raise ValueError("G1 asset revision changed")
  model = robot.model("scene_mjx") if hand == "none" else scene(hand)[0]
  names = [model.joint(int(i)).name for i in model.actuator_trnid[:, 0]]
  if names != SDK_JOINTS:
    raise ValueError("native actuator order does not match the verified SDK joint names")
  model.opt.timestep = .002
  model.opt.iterations = 50
  model.opt.ls_iterations = 20
  model.actuator_gainprm[:, 0] = prior.config["stiffness"]
  model.actuator_biasprm[:, 1] = -np.array(prior.config["stiffness"])
  model.actuator_biasprm[:, 2] = -np.array(prior.config["damping"])
  return model


def rollout(prior, hand, seed, controller, seconds=10.0, frames=None):
  model = prior_model(prior, hand)
  batch = Batch(model, 1, num_threads=1)
  qpos, qvel, ctrl, warning, torques = (batch.bind(n) for n in
                                     ("qpos", "qvel", "ctrl", "warning", "qfrc_actuator"))
  gyro, quat = batch.sensor("gyro_pelvis"), batch.sensor("orientation_pelvis")
  joints = model.actuator_trnid[:, 0]
  qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
  limits, force_limits = model.jnt_range[joints], model.jnt_actfrcrange[joints]
  names = [model.sensor(i).name for i in range(model.nsensor)
           if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_CONTACT]
  contacts = [batch.sensor(name).reshape(SLOTS, CONTACT_WIDTH) for name in names]
  counts, peaks = np.zeros(len(names), dtype=int), np.zeros(len(names))
  batch.reset(keyframe=0)
  qpos[0, qids] = prior.sdk_default + np.random.default_rng(seed).uniform(-.005, .005, 29)
  qpos[0, 2] = .8
  ctrl[0] = prior.sdk_default
  batch.forward()
  shadow = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, shadow, 0)
  shadow.qpos[:], shadow.qvel[:], shadow.ctrl[:] = qpos[0], qvel[0], ctrl[0]
  mujoco.mj_forward(model, shadow)
  floor = model.geom("floor").id
  nonfoot_ground = set()
  if frames is not None:
    frames.append((hand, 0., qpos[0].copy(), qvel[0].copy()))
  prior.reset()
  initial = qpos[0, :3].copy()
  min_height, max_drift, max_limit, max_torque_fraction, max_raw_action = .8, 0., 0., 0., 0.
  target_clip_steps, failure, elapsed = 0, None, 0.
  for step in range(round(seconds / prior.config["step_dt"])):
    obs = prior.observation(qpos[0, qids], qvel[0, vids], gyro[0], quat[0], np.zeros(3))
    target = prior.target(obs) if controller == "unitree_onnx" else prior.sdk_default
    max_raw_action = max(max_raw_action, float(np.abs(prior.last_action).max()))
    target_clip_steps += int(np.any((target < limits[:, 0]) | (target > limits[:, 1])))
    ctrl[0] = np.clip(target, limits[:, 0], limits[:, 1])
    shadow.ctrl[:] = ctrl[0]
    for substep in range(10):
      batch.step()
      mujoco.mj_step(model, shadow)
      if not np.array_equal(shadow.qpos, qpos[0]) or not np.array_equal(shadow.qvel, qvel[0]):
        raise RuntimeError("native/serial shadow state mismatch")
      for contact in shadow.contact:
        if floor not in contact.geom:
          continue
        other = int(contact.geom[1] if contact.geom[0] == floor else contact.geom[0])
        name = model.geom(other).name
        if name.endswith("_collision") and "_foot" not in name:
          nonfoot_ground.add(name)
      elapsed = (step * 10 + substep + 1) * model.opt.timestep
      if warning[:, :, 1].any() or not np.isfinite(qpos).all() or not np.isfinite(torques).all():
        raise RuntimeError("invalid native G1 physics state")
      for i, sensor in enumerate(contacts):
        if not np.isfinite(sensor).all() or np.any(sensor[:, 0] > SLOTS):
          raise RuntimeError("invalid or overflowing contact channel")
        counts[i] += int(np.any(sensor[:, 0] > 0))
        peaks[i] = max(peaks[i], float(sensor[:, 1].sum()))
      min_height = min(min_height, float(qpos[0, 2]))
      max_drift = max(max_drift, float(np.linalg.norm(qpos[0, :2] - initial[:2])))
      joint_pos = qpos[0, qids]
      max_limit = max(max_limit, float(np.maximum(limits[:, 0] - joint_pos, joint_pos - limits[:, 1]).max()))
      torque = torques[0, vids]
      max_torque_fraction = max(max_torque_fraction, float(np.max(np.abs(torque) / force_limits[:, 1])))
      if qpos[0, 2] < .5 or 1 - 2 * (qpos[0, 4]**2 + qpos[0, 5]**2) < .65:
        failure = "fall"
      invalid = [name for name, count in zip(names, counts, strict=True) if name.startswith("bat_") and count]
      if invalid:
        failure = "incidental_bat_contact"
      if nonfoot_ground:
        failure = "nonfoot_robot_ground_contact"
      if failure:
        break
    if failure:
      break
    if frames is not None and step + 1 in (100, 250, 500):
      frames.append((hand, elapsed, qpos[0].copy(), qvel[0].copy()))
  return {
    "hand": hand, "seed": seed, "controller": controller, "seconds": elapsed,
    "failure": failure, "completed": failure is None,
    "minimum_pelvis_height_m": min_height, "maximum_xy_drift_m": max_drift,
    "maximum_joint_limit_excess_rad": max_limit, "maximum_applied_joint_torque_fraction": max_torque_fraction,
    "maximum_raw_action": max_raw_action, "target_clipped_control_steps": target_clip_steps,
    "native_serial_parity": "exact_qpos_qvel_every_physics_step",
    "nonfoot_ground_contact_geoms": sorted(nonfoot_ground),
    "final_root_pose": qpos[0, :7].tolist(),
    "contact_active_substep_counts": dict(zip(names, counts.tolist(), strict=True)),
    "contact_normal_substep_peaks_n": dict(zip(names, peaks.tolist(), strict=True)),
  }


def render_frames(prior, frames, path):
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(3, 4, figsize=(16, 9))
  for axis in axes.flat:
    axis.axis("off")
  for row, hand in enumerate(("none", "right", "left")):
    model = prior_model(prior, hand)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.15, 0, .7]
    camera.distance, camera.azimuth, camera.elevation = 2.8, 125, -12
    with mujoco.Renderer(model, height=360, width=640) as renderer:
      for col, (_, time, q, dq) in enumerate(frame for frame in frames if frame[0] == hand):
        data.qpos[:], data.qvel[:] = q, dq
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        image = renderer.render().copy()
        if np.std(image) < 10:
          raise RuntimeError("blank stance diagnostic")
        axes[row, col].imshow(image)
        axes[row, col].set_title(f"{hand} bat | {time:g} s")
  fig.suptitle("External Unitree locomotion prior | seed 4201, predeclared times | NOT learned cricket")
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--assets", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  prior = UnitreePrior(args.assets)
  frames = []
  rows = [rollout(prior, hand, seed, controller,
                  frames=frames if seed == SEEDS[0] and controller == "unitree_onnx" else None)
          for hand in ("none", "right", "left")
          for controller in ("constant_target", "unitree_onnx") for seed in SEEDS]
  report = {
    "scope": "Externally trained locomotion transfer audit; no local cricket learning, hardware validation or showcase claim",
    "source": f"https://github.com/unitreerobotics/unitree_rl_lab/tree/{REVISION}",
    "external_asset_sha256": ASSET_HASHES,
    "external_asset_distribution": "Local cache only; upstream README advertises Apache-2.0, no root license file; checkpoint redistribution not cleared",
    "robot_tree": ROBOT_TREE,
    "contract": {"history": "term-major, oldest-to-newest, repeat-first reset", "input_shape": [1, 480],
                 "imu_frame": "pelvis", "policy_to_sdk": prior.mapping.tolist(), "seconds": 10,
                 "root_reset_height_m": .8, "joint_reset_jitter_rad": .005, "seeds": list(SEEDS),
                 "control_dt_s": .02, "physics_dt_s": .002, "pd_gains": "official deploy.yaml SDK order",
                 "target_clipping": "native joint limits, count every affected control step",
                 "unsupported": True, "pose_overwrite_after_reset": False,
                 "bat_contact_guard": "any sensor presence, every physics substep; fixed wrist fixture excepted",
                 "ground_contact_guard": "every-substep independent serial shadow, exact native state parity, robot feet only",
                 "sensor_timing": "MuJoCo step sensor stage, one 2 ms step behind returned generalized state; no post-step forward",
                 "no_bat_model": "Menagerie stock scene_mjx, original collisions/inertias/torque limits",
                 "bat_model": "Existing cricket scene, original rigid fixture geometry, no new pose fitting"},
    "runtime": {p: version(p) for p in ("mujoco", "mjbatch", "mujoco-menagerie", "onnxruntime", "numpy")},
    "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("cricket_g1.py"))},
    "rows": rows,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  render_frames(prior, frames, args.output.with_name("stance_diagnostic.png"))
  print(json.dumps([{k: r[k] for k in ("hand", "controller", "seed", "seconds", "failure")} for r in rows]))


if __name__ == "__main__":
  main()
