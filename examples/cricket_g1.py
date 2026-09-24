# SPDX-License-Identifier: Apache-2.0

"""Floating-base Unitree G1 cricket foundation with experimental stance policies."""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np

from mjbatch import Batch

TIMESTEP = 0.002
DECIMATION = 10
BALL_RADIUS, BALL_MASS = 0.036, 0.156
SLOTS, CONTACT_WIDTH = 4, 17
ROBOT_TREE = "57c00d310bfd8ae7d5676c64b959c86fbdd61d20"


def scene(hand="right", timestep=TIMESTEP):
  """Extend the stock 29-DoF scene without changing robot inertias or motor limits."""
  if hand not in ("left", "right"):
    raise ValueError("hand must be left or right")
  mirror = 1 if hand == "right" else -1
  line_y = -.20 * mirror
  robot = mm.get("unitree_g1")
  if robot.oid != ROBOT_TREE:
    raise ValueError("G1 asset revision changed; audit the new model before use")
  root = ET.parse(robot.xml("g1_mjx")).getroot()
  reference = ET.parse(robot.xml("scene_mjx")).getroot()
  for child in reference:
    if child.tag != "include":
      root.append(copy.deepcopy(child))
  root.find("option").set("timestep", str(timestep))
  root.find("option").set("iterations", "50")
  root.find("option").set("ls_iterations", "20")
  root.find("visual/global").set("offwidth", "1280")
  root.find("visual/global").set("offheight", "720")

  # A raised-arm reset pose keeps the full-size bat clear of the floor.
  original = robot.model("scene_mjx")
  pose = original.key_qpos[0].copy()
  for side in ("left", "right"):
    pose[original.joint(f"{side}_shoulder_pitch_joint").qposadr[0]] = -0.8
  data = mujoco.MjData(original)
  data.qpos[:] = pose
  mujoco.mj_forward(original, data)
  wrist_quat = data.body(f"{hand}_wrist_yaw_link").xquat.copy() * [1, -1, -1, -1]
  wrist = root.find(f".//body[@name='{hand}_wrist_yaw_link']")
  bat = ET.SubElement(wrist, "body", name="cricket_bat", pos="0.08 0 0", quat=" ".join(map(str, wrist_quat)))
  ET.SubElement(bat, "site", name="bat_fixture", pos="0 0 0", size=".012")
  ET.SubElement(
    bat,
    "geom",
    name="bat_handle",
    type="capsule",
    fromto="0 0 .10 0 0 -.16",
    size=".017",
    mass=".12",
    rgba=".12 .16 .19 1",
    contype="0",
    conaffinity="0",
  )
  ET.SubElement(
    bat,
    "geom",
    name="bat_blade",
    type="box",
    pos="0 0 -.405",
    size=".018 .054 .245",
    mass="1.0",
    rgba=".79 .73 .51 1",
    contype="0",
    conaffinity="0",
  )
  ET.SubElement(bat, "site", name="bat_sweet_spot", pos=".018 0 -.42", size=".012")

  world = root.findall("worldbody")[-1]
  floor = root.find(".//geom[@name='floor']")
  floor.attrib.pop("material", None)
  floor.set("rgba", ".12 .32 .17 1")
  ET.SubElement(
    world,
    "geom",
    name="pitch_marking",
    type="box",
    pos=f"9.46 {line_y} -.001",
    size="10.66 1.52 .002",
    rgba=".49 .51 .38 1",
    contype="0",
    conaffinity="0",
  )
  for end, x in (("striker", -0.6), ("bowler", 19.52)):
    crease = x + (1.22 if end == "striker" else -1.22)
    ET.SubElement(
      world,
      "geom",
      name=f"{end}_popping_crease",
      type="box",
      pos=f"{crease} {line_y} .003",
      size=".025 1.83 .002",
      rgba=".95 .95 .95 1",
      contype="0",
      conaffinity="0",
    )
    for i, y in enumerate((-0.29, -0.20, -0.11)):
      y *= mirror
      ET.SubElement(
        world,
        "geom",
        name=f"{end}_stump_{i}",
        type="capsule",
        fromto=f"{x} {y} .018 {x} {y} .70",
        size=".018",
        rgba=".92 .9 .76 1",
        contype="0",
        conaffinity="0",
      )
    ET.SubElement(
      world,
      "geom",
      name=f"{end}_bail",
      type="capsule",
      fromto=f"{x} {-.31 * mirror} .72 {x} {-.09 * mirror} .72",
      size=".008",
      rgba=".92 .9 .76 1",
      contype="0",
      conaffinity="0",
    )
  for i in range(96):
    a, b = 2 * np.pi * np.array([i, i + 1]) / 96
    ends = [
      9.46 + 24 * np.cos(a),
      line_y + 24 * np.sin(a),
      0.008,
      9.46 + 24 * np.cos(b),
      line_y + 24 * np.sin(b),
      0.008,
    ]
    ET.SubElement(
      world,
      "geom",
      name=f"boundary_{i}",
      type="capsule",
      fromto=" ".join(map(str, ends)),
      size=".022",
      rgba=".94 .94 .94 1",
      contype="0",
      conaffinity="0",
    )
  ET.SubElement(
    world, "light", name="cricket_sun", directional="true", pos="0 0 8", dir=".2 -.3 -1", diffuse=".7 .7 .7"
  )
  ball = ET.SubElement(world, "body", name="cricket_ball", pos=f"3 {line_y} 1.1")
  ET.SubElement(ball, "freejoint", name="cricket_ball_joint")
  ET.SubElement(
    ball,
    "geom",
    name="cricket_ball_geom",
    type="sphere",
    size=str(BALL_RADIUS),
    mass=str(BALL_MASS),
    rgba=".7 .035 .045 1",
    contype="0",
    conaffinity="0",
  )

  contacts, sensors = root.find("contact"), root.find("sensor")
  robot_colliders = [
    geom.get("name") for geom in root.iter("geom") if geom.get("name", "").endswith("_collision")
  ]
  ball_surfaces = ["floor", "bat_handle", "bat_blade"] + robot_colliders
  ball_surfaces += [f"{end}_stump_{i}" for end in ("striker", "bowler") for i in range(3)]
  for surface in ball_surfaces:
    ET.SubElement(
      contacts,
      "pair",
      geom1="cricket_ball_geom",
      geom2=surface,
      condim="3",
      friction=".4 .4 .005 .0001 .0001",
      solref=".008 1",
    )
    ET.SubElement(
      sensors,
      "contact",
      name=f"ball_{surface}",
      geom1="cricket_ball_geom",
      geom2=surface,
      num=str(SLOTS),
      reduce="none",
      data="found force torque dist pos normal tangent",
    )
  for geom in ("bat_handle", "bat_blade"):
    for surface in ["floor"] + robot_colliders:
      if surface in (f"{hand}_hand_collision", f"{hand}_wrist_collision"):
        continue  # The declared rigid fixture occupies the holding palm/wrist.
      ET.SubElement(contacts, "pair", geom1=geom, geom2=surface, condim="3", solref=".008 1")
      ET.SubElement(sensors, "contact", name=f"{geom}_{surface}", geom1=geom, geom2=surface,
                    num=str(SLOTS), reduce="none", data="found force torque dist pos normal tangent")
  for side in ("left", "right"):
    for i in range(1, 4):
      ET.SubElement(
        sensors,
        "contact",
        name=f"support_{side}_{i}",
        geom1=f"{side}_foot{i}_collision",
        geom2="floor",
        num=str(SLOTS),
        reduce="none",
        data="found force torque dist pos normal tangent",
      )
  ET.SubElement(sensors, "force", name="bat_fixture_force", site="bat_fixture")
  ET.SubElement(sensors, "torque", name="bat_fixture_torque", site="bat_fixture")
  root.remove(root.find("keyframe"))
  keys = ET.SubElement(root, "keyframe")
  ET.SubElement(
    keys,
    "key",
    name="cricket_ready",
    qpos=" ".join(map(str, np.r_[pose, [3, line_y, 1.1, 1, 0, 0, 0]])),
    ctrl=" ".join(map(str, pose[7:])),
  )
  xml = ET.tostring(root, encoding="unicode")
  model = mujoco.MjModel.from_xml_string(xml, assets=robot.assets("scene_mjx"))
  provenance = {
    "robot": robot.name,
    "model": "g1_mjx",
    "tree_id": robot.oid,
    "archive_sha256": robot.sha256,
    "license": robot.license,
    "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
    "grip": "single_hand_rigid_fixture_not_dexterous_grasp",
    "robot_control": "29_joint_position_targets_floating_base_no_pose_overwrite",
  }
  return model, provenance


class G1Cricket:
  """Native batch physics; reset is the only place that writes generalized state."""

  def __init__(self, count=1, hand="right", timestep=TIMESTEP):
    self.model, self.provenance = scene(hand, timestep)
    self.batch = Batch(self.model, count, num_threads=min(count, 4))
    self.qpos, self.qvel, self.ctrl = (self.batch.bind(n) for n in ("qpos", "qvel", "ctrl"))
    self.home = self.model.key_ctrl[0].copy()
    self.joints = self.model.actuator_trnid[:, 0]
    self.limits = self.model.jnt_range[self.joints]
    self.contact_names = [
      self.model.sensor(i).name
      for i in range(self.model.nsensor)
      if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_CONTACT
    ]
    self.contacts = [
      self.batch.sensor(name).reshape(count, SLOTS, CONTACT_WIDTH) for name in self.contact_names
    ]
    self.warning = self.batch.bind("warning")
    self.peak_load = np.zeros((count, len(self.contacts)))
    self.active_samples = np.zeros_like(self.peak_load, dtype=int)
    self.reset()

  def reset(self, ids=None, launch=False):
    if ids is None:
      ids = np.arange(len(self.qpos))
    self.batch.reset(ids, keyframe=0)
    if launch:
      self.batch.joint("cricket_ball_joint").qvel[ids, 0] = -8
    self.batch.forward(ids)
    self.peak_load[ids] = 0
    self.active_samples[ids] = 0
    self.elapsed = 0.0

  def step(self, action):
    action = np.asarray(action)
    if action.shape != self.ctrl.shape or not np.isfinite(action).all():
      raise ValueError(f"expected finite actions of shape {self.ctrl.shape}")
    self.ctrl[:] = np.clip(self.home + 0.5 * np.clip(action, -1, 1), self.limits[:, 0], self.limits[:, 1])
    for _ in range(DECIMATION):
      self.batch.step()
      loads = np.stack([slots[:, :, 1].sum(axis=1) for slots in self.contacts], axis=1)
      if not np.isfinite(loads).all() or any(np.any(slots[:, :, 0] > SLOTS) for slots in self.contacts):
        raise RuntimeError("invalid or overflowing G1 contact sensor")
      self.peak_load = np.maximum(self.peak_load, loads)
      self.active_samples += np.stack([(slots[:, :, 0] > 0).any(axis=1) for slots in self.contacts], axis=1)
    self.elapsed += DECIMATION * self.model.opt.timestep
    if self.warning[:, :, 1].any() or not np.isfinite(self.qpos).all():
      raise RuntimeError("invalid G1 simulation state")

  def render(self, path):
    from PIL import Image

    data = mujoco.MjData(self.model)
    data.qpos[:], data.qvel[:] = self.qpos[0], self.qvel[0]
    mujoco.mj_forward(self.model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.2, 0, 0.7]
    camera.distance, camera.azimuth, camera.elevation = 3.5, 130, -16
    with mujoco.Renderer(self.model, height=720, width=1280) as renderer:
      renderer.update_scene(data, camera)
      Image.fromarray(renderer.render()).save(path)


def audit(seconds=3.0):
  rows = []
  for hand in ("right", "left"):
    env = G1Cricket(hand=hand)
    initial = env.qpos[0].copy()
    minimum_height = initial[2]
    max_joint_error = 0.0
    for _ in range(round(seconds / (TIMESTEP * DECIMATION))):
      env.step(np.zeros((1, 29)))
      minimum_height = min(minimum_height, float(env.qpos[0, 2]))
      q = env.qpos[0, env.model.jnt_qposadr[env.joints]]
      violation = np.maximum(env.limits[:, 0] - q, 0) + np.maximum(q - env.limits[:, 1], 0)
      max_joint_error = max(max_joint_error, float(violation.max()))
    rows.append(
      {
        "hand": hand,
        "seconds": env.elapsed,
        "initial_root": initial[:7].tolist(),
        "final_root": env.qpos[0, :7].tolist(),
        "minimum_root_height_m": minimum_height,
        "fell": minimum_height < 0.45,
        "max_joint_limit_violation_rad": max_joint_error,
        "peak_sample_normal_load_n": dict(zip(env.contact_names, env.peak_load[0].tolist(), strict=True)),
        "provenance": env.provenance,
      }
    )
  return {
    "scope": "Untrained constant-joint-target physical baseline; not a learned cricket result",
    "force_scope": "Every physics-step sensor stage; not impulse or hardware tactile measurements",
    "runtime": {p: version(p) for p in ("mujoco", "mjbatch", "mujoco-menagerie")},
    "rows": rows,
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--audit", type=Path, required=True)
  parser.add_argument("--seconds", type=float, default=3.0)
  args = parser.parse_args()
  report = audit(args.seconds)
  args.audit.parent.mkdir(parents=True, exist_ok=True)
  args.audit.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(
    json.dumps(
      {"rows": [{k: r[k] for k in ("hand", "fell", "minimum_root_height_m")} for r in report["rows"]]}
    )
  )
