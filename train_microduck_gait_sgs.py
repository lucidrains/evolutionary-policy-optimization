#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "mujoco", "mlx", "numpy", "torch", "tqdm", "einx", "imageio",
#     "evolutionary-policy-optimization", "mujoco_warp_lucidrains_fork[mlx,nim]==0.0.3", "env-ssl-wrapper>=0.5.13", "x-ppo==0.0.12", "nimporter-plus", "setuptools"]
# [tool.uv.sources]
# evolutionary-policy-optimization = { path = ".", editable = true }
# ///
"""Single microduck gait, sparse gait rewards, SGS-chosen initial states.

Success = crossing the cell's target distance on foot while upright; the
sampler concentrates training on the frontier cells. With --distill, a no-CPG
student learns from the CPG teacher's labels on its own visited states (online
DAgger via x-ppo 0.0.12), optionally with representation alignment.

Dependencies come from PyPI - uv resolves them, including the fast MLX
training stack from lucidrains' mujoco_warp fork, published as
mujoco_warp_lucidrains_fork; the robot model repo (pollen-robotics/microduck_rl)
is git cloned next to this repo on first run if not already there. The CPG gait
pattern it distills from (the leg-pattern table, and the clock it rides) lives
in this file. On a Mac the run ends by rendering the gait it found into
<run>/videos/{teacher,student}.mp4.

  uv run train_microduck_gait_sgs.py --total_env_steps 2000000
"""


from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import fire
import imageio
import mlx.core as mx
import mujoco
import numpy as np
import torch
from tqdm import tqdm

from evolutionary_policy_optimization import (
    EPO,
    create_agent,
    rescale_from_to,
)

from env_ssl_wrapper import SGSSampler, SuccessGuidedSamplingWrapper, fourier_time
from mujoco_warp.mps.robot import BatchedEngine


# prerequisites: the robot model repo is required for the MJCF; clone it next
# to this repo if not already there

def ensure_deps():
    root = Path(__file__).resolve().parent.parent
    path = root / 'microduck_rl'

    if path.exists():
        return

    url = 'https://github.com/pollen-robotics/microduck_rl'
    print(f'[deps] cloning {url} -> {path}')
    subprocess.run(['git', 'clone', '--depth', '1', url, str(path)], check = True)

ensure_deps()

# constants

MJCF = Path(__file__).resolve().parent.parent / 'microduck_rl' / 'src' / 'mjlab_microduck' / 'robot' / 'microduck' / 'scene_walk.xml'

CONTROL_HZ = 50
N_SUBSTEPS = 5
PHYSICS_DT = 1.0 / (CONTROL_HZ * N_SUBSTEPS)
ACTION_SCALE = 0.5
POLICY_RANGE = (-1., 1.)
ACTION_MARGIN = 1.5
ACTION_INTERFACE_RANGE = (-ACTION_MARGIN, ACTION_MARGIN)

OBS_CPG_DIM = 2

CPG_FREQ = 2.0
CPG_SWING_MAX = 0.22
CPG_KNEE_SCALE = 1.6
CPG_ANKLE_SCALE = 0.55
CPG_ROLL_AMP = 0.035
CPG_RAMP_CYCLES = 1.0
CPG_TURN_GAIN = 0.1
COMMAND_NOMINAL_SPEED = 0.15

LIN_VEL_RANGE = (-0.3, 0.3)
ANG_VEL_RANGE = (-0.5, 0.5)
COMMAND_DIM = 2
COMMAND = (0.15, 0.0)
COMMAND_EPS = 0.01

TRACK_WEIGHT = 2.0
TRACK_STD = (0.02, 0.15)
PROGRESS_WEIGHT = 5.0
UPRIGHT_WEIGHT = 2.0
UPRIGHT_STD = 0.05
POSE_WEIGHT = 1.0
ALIVE_BONUS = 0.0
ACTION_RATE_WEIGHT = -0.1

AIR_TIME_MIN = 0.10
AIR_TIME_MAX = 0.25
AIR_TIME_WEIGHT = 3.0
AIR_TIME_RAMP_EPISODES = 960

NAN_PENALTY = -10.0
FLOOR_CLIP_PENALTY = -5.0
BODY_ANG_VEL_WEIGHT = -0.05
ANGULAR_MOMENTUM_WEIGHT = -0.02
FOOT_SLIP_WEIGHT = -0.1
FOOT_CLEARANCE_WEIGHT = -0.5
FOOT_CLEARANCE_TARGET = 0.02
SELF_COLLISION_WEIGHT = -1.0

FOOT_HEIGHT_WEIGHT = 3.0
FOOT_HEIGHT_CLEAR = 0.01
FOOT_HEIGHT_CAP = 0.02

ZMP_WEIGHT = 2.0
ZMP_STD = 0.05
ZMP_EMA = 0.7

POSE_STDS = np.array([
    0.30, 0.05, 0.40, 0.40, 0.25,
    0.35, 0.35, 0.35, 0.35,
    0.30, 0.05, 0.40, 0.40, 0.25,
])
LEG_MASK = np.array([True] * 5 + [False] * 4 + [True] * 5)
POSE_STD_STANDING_SCALE = 0.5
POSE_STD_WALKING_SCALE = 1.5

RESET_Z_RANGE = (0.0, 0.01)
RESET_JOINT_NOISE = 0.05
RESET_VEL_NOISE = 0.1
MIN_TRUNK_Z = 0.06
MIN_UPRIGHTNESS = 0.5

TRUNK_BODY = 1
FREE_QPOS_DIM = 7
FREE_QVEL_DIM = 6
QUAT_OFFSET = 3
SPATIAL_DIM = 3
BODY_VEL_DIM = 6
NUM_FEET = 2
FEET = ('left', 'right')
WORLD_UP = np.array([0., 0., 1.])

OBS_GRAVITY_DIM = 3
OBS_ANG_VEL_DIM = 3

KP_MULT = 3.0
FORCED_SWING = 0.25

RENDER_CMD = (0.15, 0.0)
CAMERA_LOOKAT = (0.3, 0.0, 0.15)
CAMERA_DISTANCE = 1.0
CAMERA_AZIMUTH = 150
CAMERA_ELEVATION = -15
VIDEO_SIZE = (480, 640)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def to_phase(x):
    """Python numbers become float64 (a clock should not lose precision to a
    default dtype); tensors and arrays keep their own dtype."""
    return torch.tensor(x, dtype = torch.float64) if isinstance(x, (int, float)) else torch.as_tensor(x)

class CPG:
    def __init__(self, freq = 2., dt = .02, phase_reset_gain = 0.):
        self.freq, self.dt, self.phase_reset_gain = freq, dt, phase_reset_gain

    def waveform(self, phi, amp, phase = 0., rectify = False, bias = 0.):
        """`(k, dim)` offsets - slot `i` gets `bias[i] + amp[i] * sin(phi + phase[i])`,
        half-wave rectified where `rectify[i]`; a scalar `phi` gives `(dim,)`."""

        phi = to_phase(phi)

        amp, phase, bias = (torch.as_tensor(x, dtype = phi.dtype, device = phi.device) for x in (amp, phase, bias))
        rectify = torch.as_tensor(rectify).to(device = phi.device)

        s = torch.sin(phi[..., None] + phase)
        s = torch.where(rectify, s.clamp(min = 0.), s)

        return bias + amp * s

    def phase_at(self, t_seconds, phi0 = 0., freq = None):
        """The phase, `phi0` at reset, at episode time `t_seconds`."""

        return (2. * math.pi * default(freq, self.freq) * to_phase(t_seconds) + phi0) % (2. * math.pi)

    def advance(self, phi, freq = None):
        """One control step of the free-running phase."""

        return (to_phase(phi) + 2. * math.pi * default(freq, self.freq) * self.dt) % (2. * math.pi)

    def phase_reset(self, phi, contact, prev_contact, targets = (math.pi, 0.), gain = None):
        """A fresh strike nudges the phase toward that foot's canonical touchdown
        phase (simultaneous strikes: the earliest foot's); contacts come back so
        the caller keeps the state."""

        gain = default(gain, self.phase_reset_gain)
        phi = to_phase(phi)

        if gain <= 0.:
            return phi, prev_contact

        newly = torch.as_tensor(contact, dtype = torch.bool) & ~torch.as_tensor(prev_contact, dtype = torch.bool)
        target = torch.as_tensor(targets, dtype = phi.dtype, device = phi.device)[newly.int().argmax(dim = -1)]
        err = (target - phi + math.pi) % (2. * math.pi) - math.pi

        return torch.where(newly.any(dim = -1), (phi + gain * err) % (2. * math.pi), phi), contact

def cpg_waveform(cpg, phi, steps, command, joints, dim, swing = None):
    """The duck leg pattern as a table on the CPG's sine basis: the model's
    left/right joints are axis-mirrored, so hips and ankles ride the base phase
    with the same sign, knees alternate as half-waves off their own phase, hip
    rolls sit at quarter phase; amplitudes ramp in, the turn command offsets
    both hip pitches."""

    swing = FORCED_SWING if swing is None else swing
    k = len(phi)
    command = np.asarray(command, np.float64)

    t = np.asarray(steps, np.float64) * (PHYSICS_DT * N_SUBSTEPS)
    ramp = np.minimum(1., t * CPG_FREQ / CPG_RAMP_CYCLES)
    speed = np.clip(np.abs(command[:, 0]) / COMMAND_NOMINAL_SPEED, 0., 1.5)

    swing_amp = swing * ramp
    knee_amp = swing_amp * CPG_KNEE_SCALE
    ankle_amp = swing_amp * CPG_ANKLE_SCALE
    roll_amp = CPG_ROLL_AMP * speed * ramp
    turn_bias = command[:, 1] * CPG_TURN_GAIN

    amp = np.zeros((k, dim))
    bias = np.zeros((k, dim))
    phase = np.zeros(dim)
    rectify = np.zeros(dim, bool)

    for slot, a, p, rect in (
        (joints['l_hip_pitch'], -swing_amp, 0., False),
        (joints['r_hip_pitch'], swing_amp, np.pi, False),
        (joints['l_knee'], knee_amp, 0., True),
        (joints['r_knee'], -knee_amp, np.pi, True),
        (joints['l_ankle'], ankle_amp, 0., False),
        (joints['r_ankle'], -ankle_amp, np.pi, False),
        (joints['l_hip_roll'], -roll_amp, np.pi / 2, False),
        (joints['r_hip_roll'], -roll_amp, np.pi / 2, False),
    ):
        amp[:, slot], phase[slot], rectify[slot] = a, p, rect

    bias[:, joints['l_hip_pitch']] = turn_bias
    bias[:, joints['r_hip_pitch']] = turn_bias

    return cpg.waveform(phi, amp, phase, rectify, bias).numpy()


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
  """Quaternion (w, x, y, z) -> row-major rotation matrix."""
  q = np.asarray(q, np.float64)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return np.stack([
    np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
    np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
    np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
  ], -2)

def _quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
  return np.stack([
    u[..., 0] * v[..., 0] - u[..., 1] * v[..., 1] - u[..., 2] * v[..., 2] - u[..., 3] * v[..., 3],
    u[..., 0] * v[..., 1] + u[..., 1] * v[..., 0] + u[..., 2] * v[..., 3] - u[..., 3] * v[..., 2],
    u[..., 0] * v[..., 2] - u[..., 1] * v[..., 3] + u[..., 2] * v[..., 0] + u[..., 3] * v[..., 1],
    u[..., 0] * v[..., 3] + u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1] + u[..., 3] * v[..., 0],
  ], -1)

def _matmul33(m: np.ndarray, v: np.ndarray) -> np.ndarray:
  """(..., 3, 3) @ (..., 3)."""
  return np.einsum("...ij,...j->...i", m, v)

def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return np.cross(a, b)

def _quat_mul_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Hamilton product (w, x, y, z), broadcasting over leading dims."""
  w1, x1, y1, z1 = np.moveaxis(np.asarray(u, np.float64), -1, 0)
  w2, x2, y2, z2 = np.moveaxis(np.asarray(v, np.float64), -1, 0)
  return np.stack([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ], -1)

def _terrain_frame(slope_deg, direction_deg = 0.0):
  """Ramp frame: R = Rz(psi) Ry(-theta). Returns (up, uphill, quat).

  `up` = plane normal (third column of R), `uphill` = in-plane steepest-ascent
  unit vector (first column of R), `quat` = R as a (w, x, y, z) quaternion, so
  a body spawned with it stands on the ramp facing uphill.
  """
  th = np.deg2rad(np.asarray(slope_deg, np.float64))
  ps = np.deg2rad(np.asarray(direction_deg, np.float64))
  ct, st, cp, sp = np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)

  up = np.stack([-st * cp, -st * sp, ct * np.ones_like(cp)], -1)
  uphill = np.stack([ct * cp, ct * sp, st * np.ones_like(cp)], -1)

  # q = qz(psi) [*] qy(-theta)  (w, x, y, z), MuJoCo convention
  qz = np.stack([np.cos(ps / 2), np.zeros_like(ps), np.zeros_like(ps), np.sin(ps / 2)], -1)
  qy = np.stack([np.cos(th / 2), np.zeros_like(th), -np.sin(th / 2), np.zeros_like(th)], -1)
  return up, uphill, _quat_mul_np(qz, qy)

class MicroduckMlxBatchedEnv:
  def __init__(
    self,
    mjcf_path: str | Path,
    *,
    num_envs: int = 512,
    kp_mult: float = KP_MULT,
    kv: float = 0.1,
    horizon: int = 400,
    seed: int = 0,
    cpg: bool = True,
    forced_swing: float | None = FORCED_SWING,
    fixed_command: tuple | None = None,
    rescale_to=ACTION_INTERFACE_RANGE,
    coop: bool = True,
  ):
    import mujoco

    self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    self.model.opt.timestep = PHYSICS_DT
    self.model.actuator_gainprm[:, 0] *= kp_mult
    self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]
    self.model.actuator_biasprm[:, 2] = -kv

    self.eng = BatchedEngine(self.model, nworld=num_envs, coop=coop)
    self.num_envs = num_envs
    self.nv = self.model.nv
    self.action_dim = self.model.nu
    self.horizon = horizon
    self.max_episode_length = horizon
    self.cpg = cpg
    self.obs_dim = (
      OBS_GRAVITY_DIM + OBS_ANG_VEL_DIM + 3 * self.action_dim + COMMAND_DIM + (OBS_CPG_DIM if self.cpg else 0)
    )
    self.rescale_to = rescale_to
    self.forced_swing = forced_swing
    self.fixed_command = np.asarray(fixed_command, dtype=np.float64) if fixed_command is not None else None
    self._random_commands = self.fixed_command is None

    stand = self.model.key("STAND")
    self.home_qpos = stand.qpos.copy()
    self.home_ctrl = stand.ctrl.copy()
    self._Riq = _quat_to_mat(self.model.body_iquat)
    self._trunk_iquat = self.model.body_iquat[TRUNK_BODY]

    self._foot_geoms = np.array([
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision") for side in FEET
    ])
    self._foot_bodies = np.array([self.model.geom_bodyid[g] for g in self._foot_geoms])
    self._non_foot_bodies = np.array(
      [i for i in range(self.model.nbody) if i not in self._foot_bodies and i != 0]
    )
    self._floor_clip_eps = 0.02

    joint_id = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) - 1  # noqa: E731
    self.l_hip_pitch = joint_id("left_hip_pitch")
    self.r_hip_pitch = joint_id("right_hip_pitch")
    self.l_knee = joint_id("left_knee")
    self.r_knee = joint_id("right_knee")
    self.l_ankle = joint_id("left_ankle")
    self.r_ankle = joint_id("right_ankle")
    self.l_hip_roll = joint_id("left_hip_roll")
    self.r_hip_roll = joint_id("right_hip_roll")

    # the phase clock is the CPG class above; the leg pattern below is the
    # duck's table on that sine basis

    self.cpg_gen = CPG(freq = CPG_FREQ, dt = PHYSICS_DT * N_SUBSTEPS)
    self.cpg_joints = dict(
      l_hip_pitch = self.l_hip_pitch, r_hip_pitch = self.r_hip_pitch, l_knee = self.l_knee, r_knee = self.r_knee,
      l_ankle = self.l_ankle, r_ankle = self.r_ankle, l_hip_roll = self.l_hip_roll, r_hip_roll = self.r_hip_roll,
    )

    g1, g2 = self.eng.rig.pair_g1, self.eng.rig.pair_g2
    self._foot_pair_idx = [
      np.array([p for p in range(len(g1)) if g in (int(g1[p]), int(g2[p]))], np.int64) for g in self._foot_geoms
    ]
    self._foot_pair_indices = self._foot_pair_idx

    self._last_action = np.zeros((num_envs, self.action_dim), np.float64)
    self._ctrl_np = np.zeros((num_envs, self.action_dim), np.float32)
    self._command = np.zeros((num_envs, COMMAND_DIM), np.float64)
    self._steps = np.zeros(num_envs, np.int64)
    self._episodes = np.zeros(num_envs, np.int64)
    self._foot_air = np.zeros((num_envs, NUM_FEET), np.float64)
    self._phi = np.zeros(num_envs, np.float64)
    self._rngs = [np.random.default_rng(seed + i) for i in range(num_envs)]

    # per-world ground plane (ramps): the floor geom is geom 0 of the scene;
    # `up` is its normal, `uphill` the in-plane ascent direction, `spawn_quat`
    # rotates the STAND keyframe onto the plane. Defaults are the flat floor.
    # NOTE: the floor geom must stay geom 0 for the packed-table write below.

    self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert self.floor_geom == 0, "floor geom must be geom 0 for per-world plane tables"

    self._up = np.tile(np.array([0.0, 0.0, 1.0]), (num_envs, 1))
    self._uphill = np.tile(np.array([1.0, 0.0, 0.0]), (num_envs, 1))
    self._spawn_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (num_envs, 1))
    self._slope_deg = np.zeros(num_envs)
    self._pos0 = np.zeros((num_envs, 3))
    self._reset_scale = np.ones(num_envs)
    self._reset_split = np.zeros(num_envs)
    self._reset_phase = np.zeros(num_envs)
    self._reset_tilt = np.zeros((num_envs, 2))
    self._reset_vel = np.zeros(num_envs)
    self._reset_cmd = np.full((num_envs, COMMAND_DIM), np.nan)
    self._phase_reset_gain = 0.0
    self._phase_prev_contact = np.zeros((num_envs, NUM_FEET), bool)
    self._reset_cache_qpos = np.zeros((num_envs, self.model.nq))
    self._reset_cache_qvel = np.zeros((num_envs, self.nv))
    self._reset_cache_valid = np.zeros(num_envs, bool)
    self._reset_cache_action = np.zeros((num_envs, self.action_dim))
    self._reset_cache_has_action = np.zeros(num_envs, bool)

    for i in range(num_envs):
      self.reset_one(i, seed=seed + i)

  @property
  def progress(self):
    """Per-world in-plane progress (m) along the uphill direction since the reset."""
    qpos = np.asarray(self.eng.qpos)
    return ((qpos[:, : SPATIAL_DIM] - self._pos0) * self._uphill).sum(-1)

  # ---------------------------------------------------------------- terrain
  def set_terrain(self, rows, slope_deg, direction_deg = 0.0):
    """Set the per-world ground plane: a ramp of `slope_deg` whose in-plane
    steepest-ascent direction is `direction_deg` (yaw about world z).

    The plane passes through the origin; the duck spawned by `reset_one` stands
    on it facing uphill. The new plane reaches the physics on the next step
    (per-world plane tables are uploaded lazily).
    """
    idx = np.asarray(rows, np.int64).reshape(-1)
    up, uphill, quat = _terrain_frame(np.asarray(slope_deg, np.float64), np.asarray(direction_deg, np.float64))

    up = np.broadcast_to(up, (len(idx), 3)) if up.ndim == 1 else up
    uphill = np.broadcast_to(uphill, (len(idx), 3)) if uphill.ndim == 1 else uphill
    quat = np.broadcast_to(quat, (len(idx), 4)) if quat.ndim == 1 else quat

    self._up[idx] = np.asarray(up, np.float64)
    self._uphill[idx] = np.asarray(uphill, np.float64)
    self._spawn_quat[idx] = np.asarray(quat, np.float64)
    self._slope_deg[idx] = np.asarray(slope_deg, np.float64) if np.ndim(slope_deg) else float(slope_deg)

    # plane normal + point live in the floor geom's rows of the packed tables

    self.eng.set_world_tables(
      idx,
      geom_quat = np.asarray(quat, np.float64).reshape(len(idx), 4),
      geom_pos = np.zeros((len(idx), 3)),
    )

  def set_reset_scale(self, rows, scale):
    """Set the per-world reset perturbation scale (0 = exact home pose at rest)."""
    idx = np.asarray(rows, np.int64).reshape(-1)
    self._reset_scale[idx] = np.asarray(scale, np.float64)

  def set_reset_pose(self, rows, split = 0.0, phase = 0.0):
    """Prime the reset pose: a mid-step hip split (rad) and the initial clock
    phase (rad). `split` puts the left leg forward and the right leg back with
    matching knee flex, i.e. the body starts inside a step cycle; 0 = home."""
    idx = np.asarray(rows, np.int64).reshape(-1)
    self._reset_split[idx] = np.asarray(split, np.float64)
    self._reset_phase[idx] = np.asarray(phase, np.float64)

  def set_reset_state(self, rows, qpos, qvel = None, last_action = None):
    """Set a cached reset state (qpos, qvel) to be written on the next reset_one."""
    idx = np.asarray(rows, np.int64).reshape(-1)
    qpos = np.asarray(qpos, np.float64).reshape(len(idx), -1)
    self._reset_cache_qpos[idx] = qpos

    if qvel is None:
        self._reset_cache_qvel[idx] = 0.0
    else:
        self._reset_cache_qvel[idx] = np.asarray(qvel, np.float64).reshape(len(idx), -1)

    if last_action is not None:
        self._reset_cache_action[idx] = np.asarray(last_action, np.float64).reshape(len(idx), -1)
        self._reset_cache_has_action[idx] = True
    else:
        self._reset_cache_action[idx] = 0.0
        self._reset_cache_has_action[idx] = False

    self._reset_cache_valid[idx] = True

  def set_reset_tilt(self, rows, pitch = 0.0, roll = 0.0):
    """Tilt the spawned trunk by (pitch, roll) rad in addition to the terrain frame."""
    idx = np.asarray(rows, np.int64).reshape(-1)
    self._reset_tilt[idx, 0] = np.asarray(pitch, np.float64)
    self._reset_tilt[idx, 1] = np.asarray(roll, np.float64)

  def set_reset_vel(self, rows, vx = 0.0):
    """Give the spawned trunk a forward (world x) velocity at reset.

    A mid-stride start with momentum cannot be survived by staying put: the
    trunk carries the duck over its feet and something has to catch it - the
    initial-state-distribution channel (RSI / OmniReset) for stepping."""
    idx = np.asarray(rows, np.int64).reshape(-1)
    self._reset_vel[idx] = np.asarray(vx, np.float64)

  def set_reset_command(self, rows, cmd):
    """Pin the per-episode velocity command (NaN clears back to the sampler)."""
    idx = np.asarray(rows, np.int64).reshape(-1)
    self._reset_cmd[idx] = np.asarray(cmd, np.float64).reshape(len(idx), COMMAND_DIM)

  def _cpg_waveform(self, phi, steps, command):
    """The duck's leg pattern on the CPG clock (same math as the step-time CPG)."""
    return cpg_waveform(self.cpg_gen, phi, steps, command, self.cpg_joints, self.action_dim, self.forced_swing)

  def cpg_waveform_at(self, t_seconds, command, phi0 = 0.0):
    """CPG waveform at episode time `t_seconds` for a reset-started episode."""
    t = np.asarray(t_seconds, np.float64)
    return self._cpg_waveform(self.cpg_gen.phase_at(t, phi0).numpy(), t / (PHYSICS_DT * N_SUBSTEPS), command)

  def _phase_reset(self, phi, active):
    """Contact-driven phase reset (Taga & Yamaguchi 1991; Kimura et al. 2007).

    The reset law lives in the CPG class above; the env owns the contact
    state it needs - which feet touched on the previous step - and reads the
    fresh contacts off the engine. Gain 0 disables the entrainment.
    """
    gain = self._phase_reset_gain

    if gain <= 0.0 or not hasattr(self.eng, 'pair_conct'):
      return phi

    pair_conct = np.asarray(self.eng.pair_conct)[active]
    contact = np.stack([
      (pair_conct[:, ids].sum(axis = 1) > 0) if len(ids) else np.zeros(len(active), bool)
      for ids in self._foot_pair_indices
    ], axis = 1)
    prev = self._phase_prev_contact[active]
    phi, contact = self.cpg_gen.phase_reset(phi, contact, prev, gain = gain)
    self._phase_prev_contact[active] = contact
    return phi.numpy()

  @property
  def foot_contacts(self):
    """(num_envs, NUM_FEET) bool: which feet are touching the ground right now."""
    pair_conct = np.asarray(self.eng.pair_conct)
    return np.stack([
      (pair_conct[:, ids].sum(axis = 1) > 0) if len(ids) else np.zeros(pair_conct.shape[0], bool)
      for ids in self._foot_pair_indices
    ], axis = 1)

  # ------------------------------------------------------------------ helpers
  def _write_state(self, i: int, qpos: np.ndarray, qvel: np.ndarray):
    self.eng.qpos = self.eng.qpos.at[i].subtract(self.eng.qpos[i]).at[i].add(mx.array(qpos.astype(np.float32)))
    self.eng.qvel = self.eng.qvel.at[i].subtract(self.eng.qvel[i]).at[i].add(mx.array(qvel.astype(np.float32)))
    self.eng.warm = self.eng.warm.at[i].subtract(self.eng.warm[i])

  def _object_velocity(self, cvel, xipos, subtree_com, body: int, local: bool):
    """Batched mj_objectVelocity."""
    ang = cvel[:, body, :3]
    lin = cvel[:, body, 3:] + _cross(ang, xipos[:, body] - subtree_com[:, int(self.model.body_rootid[body])])
    if local:
      ximat = np.einsum("kij,jl->kil", self._xmat[:, body], self._Riq[body])
      ang = _matmul33(np.swapaxes(ximat, -1, -2), ang)
      lin = _matmul33(np.swapaxes(ximat, -1, -2), lin)
    return ang, lin

  def _observe(self, qpos, qvel, xmat_trunk, ang_local, command, last_action, phi):
    g = -_matmul33(np.swapaxes(xmat_trunk, -1, -2), np.broadcast_to(WORLD_UP, (len(qpos), 3)))
    cpg_obs = np.stack([np.sin(phi), np.cos(phi)], -1) if self.cpg else np.empty((len(qpos), 0))
    return np.concatenate([
      g,
      ang_local,
      qpos[:, FREE_QPOS_DIM:] - self.home_qpos[FREE_QPOS_DIM:],
      qvel[:, FREE_QVEL_DIM:],
      last_action,
      command,
      cpg_obs,
    ], -1).astype(np.float32)

  def _safe_obs(self, command, last_action, phi):
    n = len(command)
    if self.cpg:
      cpg_obs = np.stack([np.sin(phi), np.cos(phi)], -1)
    else:
      cpg_obs = np.zeros((n, 0))
    return np.concatenate([
      np.zeros((n, 3)),
      np.zeros((n, 3)),
      np.zeros((n, self.action_dim)),
      np.zeros((n, self.action_dim)),
      last_action,
      command,
      cpg_obs,
    ], -1).astype(np.float32)

  # --------------------------------------------------------------------- env
  def reset_one(self, worker: int, seed: int | None = None):
    i = int(worker)
    rng = self._rngs[i] if seed is None else np.random.default_rng(seed)
    if seed is not None:
      self._rngs[i] = rng

    up = self._up[i]

    # spawn on the current ground plane: anchor along the plane normal above
    # the origin (the plane passes through it), trunk rotated onto the plane

    qpos = self.home_qpos.copy()

    # per-worker reset perturbation scale: 0 = exact home pose at rest (the
    # easiest initial state), 1 = the stock reset noise. SGS can ride a bank of
    # these scales (initial-state distribution curriculum).

    s = self._reset_scale[i]
    anchor = up * (self.home_qpos[SPATIAL_DIM - 1] + s * rng.uniform(*RESET_Z_RANGE))
    qpos[: SPATIAL_DIM] = anchor
    spawn_quat = self._spawn_quat[i]
    pitch, roll = self._reset_tilt[i]

    if pitch != 0.0 or roll != 0.0:
        # local-frame tilt: q_tilt (pitch about y, roll about x)
        half = 0.5 * pitch
        qp = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
        half = 0.5 * roll
        qr = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
        spawn_quat = _quat_mul_np(spawn_quat, _quat_mul_np(qp, qr))

    qpos[QUAT_OFFSET : QUAT_OFFSET + SPATIAL_DIM + 1] = _quat_mul_np(
      spawn_quat, qpos[QUAT_OFFSET : QUAT_OFFSET + SPATIAL_DIM + 1]
    )
    qpos[FREE_QPOS_DIM:] += s * rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, size=self.action_dim)
    qvel = s * rng.uniform(-RESET_VEL_NOISE, RESET_VEL_NOISE, size=self.nv)

    # mid-step priming: left leg forward, right leg back, knees flexed to keep
    # the feet under the trunk (a configuration from inside a step cycle)

    split = self._reset_split[i]

    if split != 0.0:
        lead = split
        trail = -split
        qpos[FREE_QPOS_DIM + self.l_hip_pitch] += lead
        qpos[FREE_QPOS_DIM + self.r_hip_pitch] += trail
        qpos[FREE_QPOS_DIM + self.l_knee] += 0.6 * abs(lead)
        qpos[FREE_QPOS_DIM + self.r_knee] += 0.6 * abs(trail)
        qpos[FREE_QPOS_DIM + self.l_ankle] -= 0.5 * lead
        qpos[FREE_QPOS_DIM + self.r_ankle] -= 0.5 * trail

    if self._random_commands and rng.uniform() < 0.25:
      command = np.zeros(COMMAND_DIM)
    elif self._random_commands:
      command = np.array([rng.uniform(*LIN_VEL_RANGE), rng.uniform(*ANG_VEL_RANGE)])
    else:
      command = self.fixed_command.copy()

    # a bank cell may pin the episode's command (the commanded-goal axis)

    if np.isfinite(self._reset_cmd[i]).all():
      command = self._reset_cmd[i].copy()

    if self._reset_cache_valid[i]:
        qpos = self._reset_cache_qpos[i].copy()
        qvel = self._reset_cache_qvel[i].copy()

    # mid-stride momentum prime: forward trunk velocity on top of the spawned
    # state (world-x is the free joint's first velocity dof)

    qvel[0] += self._reset_vel[i]

    self._write_state(i, qpos, qvel)
    self._last_action[i] = self._reset_cache_action[i] if self._reset_cache_has_action[i] else 0.0
    self._command[i] = command
    self._steps[i] = 0
    self._foot_air[i] = 0.0
    self._episodes[i] += 1
    self._phi[i] = self._reset_phase[i]
    self._pos0[i] = qpos[: SPATIAL_DIM]

    xquat = qpos[QUAT_OFFSET:QUAT_OFFSET + SPATIAL_DIM + 1]
    xquat = xquat / np.linalg.norm(xquat)
    xmat = _quat_to_mat(xquat)
    ximat = xmat @ _quat_to_mat(self._trunk_iquat)
    g = -xmat.T @ WORLD_UP
    ang_local = ximat.T @ qvel[SPATIAL_DIM:2 * SPATIAL_DIM]
    cpg_obs = np.array([np.sin(0.0), np.cos(0.0)]) if self.cpg else np.empty(0)
    obs = np.concatenate([
      g,
      ang_local,
      qpos[FREE_QPOS_DIM:] - self.home_qpos[FREE_QPOS_DIM:],
      qvel[FREE_QVEL_DIM:],
      np.zeros(self.action_dim),
      command,
      cpg_obs,
    ]).astype(np.float32)
    return obs, {}

  def reset(self, seed: int | None = None, **kwargs):
    obs = np.stack([self.reset_one(i, seed=seed)[0] for i in range(self.num_envs)])
    return obs, {}

  def reset_batch(self, seeds):
    return np.stack([self.reset_one(i, seed=s)[0] for i, s in enumerate(seeds)])

  def step_batch(self, actions, worker_ids=None):
    active = np.arange(self.num_envs) if worker_ids is None else np.asarray(list(worker_ids), np.int64)
    k = len(active)

    # worlds outside the active set are not advanced by the caller: snapshot
    # their state and restore it after the whole-batch engine step, so idle
    # slots do not decay under a stale ctrl (they come back mid-episode intact)

    idle = np.setdiff1d(np.arange(self.num_envs, dtype = np.int64), active) if k < self.num_envs else np.empty(0, np.int64)
    keep = None
    if len(idle):
      keep = np.zeros(self.num_envs, bool)
      keep[idle] = True
      qpos_keep = np.asarray(self.eng.qpos).astype(np.float32)
      qvel_keep = np.asarray(self.eng.qvel).astype(np.float32)

    a = np.asarray(actions, np.float32).reshape(k, self.action_dim)
    if self.rescale_to is not None:
      a = rescale_from_to(a, to_range=self.rescale_to)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a_eff = np.clip(a, *POLICY_RANGE)
    action_rate = ((a_eff - np.clip(self._last_action[active], *POLICY_RANGE)) ** 2).mean(axis=1)

    command = self._command[active]
    phi = self._phi[active]
    steps = self._steps[active]

    ctrl = self.home_ctrl[None] + ACTION_SCALE * a_eff
    if self.cpg:
      phi = self._phase_reset(phi, active)
      cpg_target = self._cpg_waveform(phi, steps, command)
      ctrl = ctrl + cpg_target
      phi = self.cpg_gen.advance(phi).numpy()

    self._ctrl_np[active] = ctrl.astype(np.float32)
    self.eng.ctrl = mx.array(self._ctrl_np)
    for s in range(N_SUBSTEPS):
      self.eng.step(kin=(s == N_SUBSTEPS - 1))
    mx.eval(
      self.eng.qpos, self.eng.qvel, self.eng.xmat, self.eng.xpos, self.eng.cvel, self.eng.xipos,
      self.eng.cinert, self.eng.geom_xpos, self.eng.subtree_com, self.eng.pair_conct,
    )

    if keep is not None:
      mask = mx.array(keep)[:, None]
      self.eng.qpos = mx.where(mask, mx.array(qpos_keep), self.eng.qpos)
      self.eng.qvel = mx.where(mask, mx.array(qvel_keep), self.eng.qvel)

    qpos = np.asarray(self.eng.qpos)[active].astype(np.float64)
    qvel = np.asarray(self.eng.qvel)[active].astype(np.float64)
    self._xmat = np.asarray(self.eng.xmat)[active].astype(np.float64).reshape(k, self.model.nbody, 3, 3)
    xpos = np.asarray(self.eng.xpos)[active].astype(np.float64)
    cvel = np.asarray(self.eng.cvel)[active].astype(np.float64)
    xipos = np.asarray(self.eng.xipos)[active].astype(np.float64)
    cinert = np.asarray(self.eng.cinert)[active].astype(np.float64)
    subtree_com = np.asarray(self.eng.subtree_com)[active].astype(np.float64)
    geom_xpos = np.asarray(self.eng.geom_xpos)[active].astype(np.float64)
    pair_conct = np.asarray(self.eng.pair_conct)[active]

    self._steps[active] = steps + 1
    self._last_action[active] = a
    self._phi[active] = phi

    self._xmat_trunk = self._xmat[:, TRUNK_BODY]
    ang_local, _ = self._object_velocity(cvel, xipos, subtree_com, TRUNK_BODY, local=True)
    _, lin_world = self._object_velocity(cvel, xipos, subtree_com, TRUNK_BODY, local=False)

    qw, qx, qy, qz = qpos[:, QUAT_OFFSET:QUAT_OFFSET + SPATIAL_DIM + 1].T
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    vx = np.cos(yaw) * lin_world[:, 0] + np.sin(yaw) * lin_world[:, 1]
    wz = ang_local[:, 2]

    # terrain-relative geometry: heights and uprightness are measured against
    # the per-world ground plane (up == world z on the flat floor)

    up = self._up[active]
    trunk_up = self._xmat_trunk[:, :, SPATIAL_DIM - 1]
    uprightness = (trunk_up * up).sum(-1)
    trunk_h = (xpos[:, TRUNK_BODY] * up).sum(-1)

    finite = np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1)
    terminated = (uprightness < MIN_UPRIGHTNESS) | (trunk_h < MIN_TRUNK_Z)
    truncated = (steps + 1) >= self.horizon
    truncated = truncated & ~terminated

    body_h = (xpos[:, self._non_foot_bodies] * up[:, None, :]).sum(-1).min(axis=1)
    floor_clip_depth = np.maximum(0.0, -body_h)
    terminated = terminated | (floor_clip_depth > self._floor_clip_eps)

    reward = np.full(k, float(ALIVE_BONUS))
    reward += np.where(floor_clip_depth > 0.0, FLOOR_CLIP_PENALTY * floor_clip_depth * 100.0, 0.0)
    reward += TRACK_WEIGHT * np.exp(-((vx - command[:, 0]) ** 2) / TRACK_STD[0])
    reward += TRACK_WEIGHT * np.exp(-((wz - command[:, 1]) ** 2) / TRACK_STD[1])
    reward += UPRIGHT_WEIGHT * np.exp(-((1.0 - uprightness) ** 2) / UPRIGHT_STD)
    pose_scale = np.where(
      (np.abs(command[:, 0]) > COMMAND_EPS) | (np.abs(command[:, 1]) > COMMAND_EPS),
      POSE_STD_WALKING_SCALE, POSE_STD_STANDING_SCALE,
    )
    pose_err = (qpos[:, FREE_QPOS_DIM:] - self.home_qpos[FREE_QPOS_DIM:])[:, LEG_MASK] / (
      POSE_STDS[LEG_MASK] * pose_scale[:, None]
    )
    reward += POSE_WEIGHT * np.exp(-(pose_err ** 2).mean(axis=1))
    reward += ACTION_RATE_WEIGHT * action_rate
    moving = np.abs(command[:, 0]) > COMMAND_EPS
    reward += np.where(moving, PROGRESS_WEIGHT * vx * np.sign(command[:, 0]), 0.0)

    is_standing_gait = (uprightness >= 0.85) & (trunk_h >= 0.095) & (floor_clip_depth == 0.0)
    moving_any = (np.abs(command[:, 0]) > COMMAND_EPS) | (np.abs(command[:, 1]) > COMMAND_EPS)
    gait_active = is_standing_gait & moving_any

    foot_contact = np.stack([
      (pair_conct[:, idx].sum(axis=1) > 0) if len(idx) else np.zeros(k, bool) for idx in self._foot_pair_idx
    ], axis=1)
    air = np.where(gait_active[:, None], np.where(foot_contact, 0.0, self._foot_air[active] + 1.0), 0.0)
    self._foot_air[active] = air
    air_time = air * (PHYSICS_DT * N_SUBSTEPS)
    ramp_ep = np.minimum(1.0, self._episodes[active] / AIR_TIME_RAMP_EPISODES)
    reward += np.where(
      gait_active, AIR_TIME_WEIGHT * ramp_ep * ((air_time > AIR_TIME_MIN) & (air_time < AIR_TIME_MAX)).sum(axis=1), 0.0
    )

    foot_h = (geom_xpos[:, self._foot_geoms] * up[:, None, :]).sum(-1)
    reward += np.where(
      is_standing_gait,
      FOOT_HEIGHT_WEIGHT * np.clip((foot_h - FOOT_HEIGHT_CLEAR) / FOOT_HEIGHT_CAP, 0.0, 1.0).sum(axis=1),
      0.0,
    )
    reward += BODY_ANG_VEL_WEIGHT * (ang_local ** 2).sum(axis=1)

    i_comp = cinert[:, TRUNK_BODY, :6]
    inertia = np.stack([
      np.stack([i_comp[:, 0], i_comp[:, 3], i_comp[:, 4]], -1),
      np.stack([i_comp[:, 3], i_comp[:, 1], i_comp[:, 5]], -1),
      np.stack([i_comp[:, 4], i_comp[:, 5], i_comp[:, 2]], -1),
    ], -2)
    angmom = np.einsum("kij,kj->ki", inertia, ang_local)
    reward += ANGULAR_MOMENTUM_WEIGHT * (angmom ** 2).sum(axis=1)

    slip = np.zeros(k)
    for body in self._foot_bodies:
      _, v_foot = self._object_velocity(cvel, xipos, subtree_com, int(body), local=False)
      slip += v_foot[:, 0] ** 2 + v_foot[:, 1] ** 2
    reward += np.where(
      moving_any,
      FOOT_SLIP_WEIGHT * slip
      + FOOT_CLEARANCE_WEIGHT * np.clip((FOOT_CLEARANCE_TARGET - foot_h) / FOOT_CLEARANCE_TARGET, 0.0, 1.0).mean(axis=1),
      0.0,
    )

    obs = self._observe(qpos, qvel, self._xmat_trunk, ang_local, command, self._last_action[active], phi)
    bad = ~finite
    if bad.any():
      obs = np.where(bad[:, None], self._safe_obs(command, self._last_action[active], phi), obs)
      reward = np.where(bad, NAN_PENALTY, reward)
      terminated = terminated | bad

    return obs, reward.astype(np.float32), terminated, np.where(bad, False, truncated), {}

  def set_attr(self, key: str, value):
    setattr(self, key, value)

  def to_epo_agent(self, **kwargs):
    return build_epo_agent(self.obs_dim, self.action_dim, kwargs)

  def seed(self, seed: int):
    self._rngs = [np.random.default_rng(seed + i) for i in range(self.num_envs)]

  def close(self):
    pass

# gait banks

DEFAULT_BANK = ('tier', 'target')
SPLITS = (-0.3, -0.15, 0.0, 0.15, 0.3)
PHASES = (0.0, 1.5708, 3.1416, 4.7124)
TILTS = (0.0, 0.05, -0.05)
SCALES = (0.0, 0.5, 1.0)
TARGETS = (0.05, 0.15, 0.3, 0.5, 0.8)

def make_bank(
    splits = SPLITS,
    phases = PHASES,
    tilts = TILTS,
    scales = SCALES,
    targets = TARGETS,
    tiers: int = 6,
):
    """The SGS bank: difficulty tier x target distance, one cell per combination.

    The state grid (foot split x clock phase x trunk tilt x reset-noise scale)
    is ranked by a difficulty proxy and cut into `tiers` chunks, so a cell is
    "a start this hard, this far to walk": SGS rides the tier x target frontier
    while each cell keeps drawing diverse states from its own chunk. Coarse
    cells give per-cell success rates that stabilise within a cycle, which is
    what an initial-state curriculum needs to climb.
    """

    grid = [(sp, ph, ti, sc) for sp in splits for ph in phases for ti in tilts for sc in scales]
    span_split = max(abs(v) for v in splits) or 1.0
    span_tilt = max(abs(v) for v in tilts) or 1.0

    # difficulty proxy: a noiseless home-pose start (scale 0) is easiest, then
    # reset noise, then a mid-step split, then a tilted trunk - so tier 0 is
    # "stand still at rest" and the last tier is "spawn mid-fall"

    diff = np.array([
        abs(sp) / span_split + 0.5 * abs(ti) / span_tilt + 0.5 * float(sc)
        for sp, _, ti, sc in grid
    ])
    tier_of = np.empty(len(grid), np.int64)

    for t, chunk in enumerate(np.array_split(np.argsort(diff), min(tiers, len(grid)))):
        tier_of[chunk] = t

    bank = []

    for t in range(tier_of.max() + 1):
        pool = [grid[k] for k in np.flatnonzero(tier_of == t)]

        for tg in targets:
            def draw(rng, idx, step, pool = pool, target = float(tg)):
                sp, ph, ti, sc = pool[int(rng.integers(len(pool)))]
                return dict(split = sp, phase = ph, tilt = ti, scale = sc, target = target)

            bank.append(draw)

    return bank
def make_walk_bank(
    vxs = (0.0, 0.1, 0.2),
    phases = PHASES,
    split_amp: float = 0.15,
    targets = TARGETS,
):
    """The walking-pathway bank (OmniReset-style reset coverage of the gait).

    Each cell is a mid-stride start - forward trunk velocity x clock phase x hip
    split amplitude - crossed with a target distance. The grid puts the bank's
    mass on the *pathway* of the behavior itself (mid-step poses, weight
    transfer, forward momentum), which is where RSI (DeepMimic) and OmniReset
    put the initial-state distribution: from a momentum start the duck cannot
    freeze back to a stand, so the reachable first move is a catch step, and
    SGS then rides the frontier of (momentum x distance) as steps stabilise.
    Per-draw jitter turns each cell into a region, keeping per-cell success
    rates stable enough for the curriculum to climb.
    """

    bank = []

    for vx in vxs:
        for ph in phases:
            sp = split_amp * np.sin(float(ph))

            for tg in targets:
                def draw(rng, idx, step, vx = float(vx), ph = float(ph), sp = float(sp), tg = float(tg)):
                    return dict(
                        vx = vx + rng.uniform(-0.02, 0.02),
                        split = sp + rng.uniform(-0.03, 0.03),
                        phase = ph + rng.uniform(-0.3, 0.3),
                        tilt = 0.0,
                        scale = 0.5,
                        target = tg,
                    )

                bank.append(draw)

    return bank
def make_stand_bank(
    scales = (0.0, 0.5, 1.0),
    targets = TARGETS,
    commands = None,
):
    """The home-pose bank: reset-noise scale x target distance (x commanded goal).

    No momentum, no stride primes - the regime in which this duck's from-scratch
    locomotion was first found (a slow upright drift): difficulty is the reset
    noise x how far the episode has to travel. SGS rides that frontier first;
    the stride / momentum primes re-enter once the drift is stable.

    Pass `commands` (a sequence of (lin, ang)) to add the commanded-goal axis:
    each cell pins its episode's velocity command, so the policy learns
    "stand still" / "walk this fast" per command rather than one fixed command.
    """

    bank = []

    for sc in scales:
        for cmd in (commands if commands is not None else (None,)):
            for tg in targets:
                def draw(rng, idx, step, sc = float(sc), tg = float(tg), cmd = cmd):
                    state = dict(split = 0.0, phase = 0.0, tilt = 0.0, scale = sc + rng.uniform(-0.05, 0.05), vx = 0.0, target = tg)

                    if cmd is not None:
                        state['cmd'] = (cmd[0] + rng.uniform(-0.02, 0.02), cmd[1] + rng.uniform(-0.05, 0.05))

                    return state

                bank.append(draw)

    return bank
def _floats(v):
    """Fire hands comma-separated CLI values as a tuple, a string or one scalar."""
    if isinstance(v, (int, float)):
        return (float(v),)

    parts = v.split(',') if isinstance(v, str) else v
    return tuple(float(x) for x in parts if str(x).strip() != '')
def _pairs(v):
    """Parse 'lin[:ang],...' into a tuple of (lin, ang) velocity commands."""
    if not isinstance(v, str) or v.strip() == '':
        return ()

    out = []

    for part in v.split(','):
        if part.strip() == '':
            continue
        bits = part.split(':')
        out.append((float(bits[0]), float(bits[1]) if len(bits) > 1 else 0.0))

    return tuple(out)
def make_gait_bank(kind: str, targets = '0.05,0.15,0.3,0.5,0.8', stand_scales = '0.0,0.5,1.0', stand_commands = '',
                   primed_splits = '0.0,0.15,0.3', bank_tiers: int = 6, walk_vxs = '0.0,0.05,0.10',
                   walk_phases = '0,1.5708,3.1416,4.7124', walk_split_amp: float = 0.15):
    """The bank for a training run, by kind: 'stand' (reset-noise scale x
    commanded goal x target ladder), 'walk' (mid-stride starts x targets),
    'primed' (mid-step primes), or the difficulty-tiered default grid."""

    if kind == 'walk':
        return make_walk_bank(vxs = _floats(walk_vxs), phases = _floats(walk_phases),
                              split_amp = walk_split_amp, targets = _floats(targets))

    if kind == 'stand':
        return make_stand_bank(scales = _floats(stand_scales), targets = _floats(targets),
                               commands = _pairs(stand_commands) or None)

    if kind == 'primed':
        splits = _floats(primed_splits)
        return make_bank(splits = splits, phases = (0.0,), tilts = (0.0,), scales = (1.0,),
                         targets = _floats(targets)[:1], tiers = len(splits))

    return make_bank(targets = _floats(targets), tiers = bank_tiers)

@torch.no_grad()
def evaluate(env: DuckGaitVecEnv, agent, num_episodes = 128, horizon = 400, latent_id = 0, reach = 0.35):
    """Greedy eval from the clean stand: distance, lines, foot steps, falls.

    `ev_lines` is standing-gated progress (same credit the env pays), and
    `ev_reach` the fraction of eval episodes that walk `reach` metres upright.
    """

    n = min(num_episodes, env.num_envs)
    rows = np.arange(n, dtype = np.int64)
    env._active = rows
    env._apply_state(rows, dict(split = 0.0, phase = 0.0, tilt = 0.0, scale = 0.0, target = 0.0))

    alive = np.ones(n, bool)
    x_max, x_end, lines = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    speed_sum, n_steps, surv_steps, fell = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, bool)
    prev_contact = np.zeros((n, len(env.env._foot_pair_indices)), bool)
    strike_errs = []
    prev_x = np.asarray(env.env.eng.qpos)[rows, 0].astype(np.float64)

    agent.eval()
    t_eval = time.perf_counter()

    for _ in range(horizon):
        idx = np.flatnonzero(alive)

        if len(idx) == 0:
            break

        state = torch.from_numpy(np.ascontiguousarray(env._obs[idx])).float()
        action = agent.get_actor_actions(state, latent_id = latent_id, sample = False, use_unwrapped_model = True)
        joint_targets = env._synthesize(action.numpy().astype(np.float32), idx) if env.learned_clock else action.numpy()

        obs, _, term, trunc, _ = env.env.step_batch(joint_targets, worker_ids = idx)

        # advance the per-episode time (the wrapper's step does this; the eval
        # steps the inner env directly, so mirror it or time features freeze)

        env._time[idx] += 1

        contact = env.env.foot_contacts[idx]
        newly = (~prev_contact[idx]) & contact

        # strike-phase error: wrapped distance from the canonical touchdown
        # phase (left pi, right 0) at each fresh strike

        struck = newly.any(axis = 1)

        if struck.any():
            phi_now = np.asarray(env.env._phi)[idx][struck]
            tgt = np.where(newly[struck][:, 0], np.pi, 0.0)
            strike_errs.extend(np.abs((tgt - phi_now + np.pi) % (2.0 * np.pi) - np.pi).tolist())

        n_steps[idx] += newly.sum(axis = 1)
        prev_contact[idx] = contact

        trunk_up_z = np.asarray(env.env._xmat_trunk)[:, -1, -1]
        qpos_active = np.asarray(env.env.eng.qpos)[idx]
        standing = (trunk_up_z >= 0.85) & (qpos_active[:, SPATIAL_DIM - 1] >= 0.095)

        x = qpos_active[:, 0].astype(np.float64)
        forward = np.floor(np.maximum(x, 0.0) / env.line_step)
        prev_forward = np.floor(np.maximum(prev_x[idx], 0.0) / env.line_step)
        lines[idx] += ((forward > prev_forward) & standing).astype(np.int64)
        speed_sum[idx] += np.abs(x - prev_x[idx]) / (PHYSICS_DT * N_SUBSTEPS)
        prev_x[idx] = x
        x_max[idx] = np.maximum(x_max[idx], x)
        x_end[idx] = x
        surv_steps[idx] += 1

        env._obs[idx] = env._augment_batch(obs, idx)

        done = np.asarray(term) | np.asarray(trunc)
        fell[idx[done]] = np.asarray(term, bool)[done]
        alive[idx[done]] = False

    agent.train()
    print(f'[eval] {n} episodes x {horizon} steps in {time.perf_counter() - t_eval:.2f}s')

    return dict(
        ev_x = float(x_end.mean()),
        ev_strike_err = float(np.mean(strike_errs)) if strike_errs else float('nan'),
        ev_strike_err_std = float(np.std(strike_errs)) if strike_errs else float('nan'),
        ev_x_max = float(x_max.max()),
        ev_speed = float(speed_sum.sum() / max(1, horizon * n)),
        ev_lines = float(lines.mean()),
        ev_reach = float((lines * env.line_step >= reach).mean()),
        ev_steps = float(n_steps.mean()),
        ev_survival = float(surv_steps.mean() / horizon),
        ev_falls = float(fell.mean()),
    )

class DuckGaitVecEnv:
    """Batched microduck sparse-gait task; SGS decides which state each episode starts in."""

    autoreset = True
    is_vector = True

    def __init__(
        self,
        num_envs: int = 1024,
        horizon: int = 400,
        mjcf_path = None,
        seed: int = 0,
        command: float = 0.15,
        learned_clock: bool = True,
        clock_band = (0.3, 3.0),
        line_step: float = 0.05,
        line_bonus: float = 0.1,
        target_bonus: float = 1.0,
        fall_penalty: float = 0.0,
        min_steps_success: int = 3,
        min_sustain_steps: int = 50,
        stance_soft: bool = True,
        forward_pull: float = 0.0,
        time_features: str = 'off',
        cpg: bool = False,
        forced_swing: float | None = 0.25,
        phase_reset: float = 0.0,
        shaping: float = 1.0,
        bank = None,
        sgs: dict | None = None,
    ):
        self.num_envs = int(num_envs)

        self._ctor_kwargs = dict(
            num_envs = num_envs, horizon = horizon, mjcf_path = mjcf_path, seed = seed, command = command,
            learned_clock = learned_clock, clock_band = clock_band, line_step = line_step, line_bonus = line_bonus,
            target_bonus = target_bonus, fall_penalty = fall_penalty, min_steps_success = min_steps_success,
            min_sustain_steps = min_sustain_steps, stance_soft = stance_soft, forward_pull = forward_pull,
            time_features = time_features, cpg = cpg, forced_swing = forced_swing, phase_reset = phase_reset,
            shaping = shaping, bank = bank, sgs = sgs,
        )
        self.horizon = int(horizon)
        self.line_step = float(line_step)
        self.line_bonus = float(line_bonus)
        self.target_bonus = float(target_bonus)
        self.fall_penalty = float(fall_penalty)
        self.min_steps_success = int(min_steps_success)

        # an arrival only counts (target bonus + SGS success) once the duck has
        # *kept* walking for this many control steps after crossing, so a
        # forward topple past the line earns neither - starts that reward
        # falling-forward no longer look like easy cells to the sampler

        self.min_sustain_steps = int(min_sustain_steps)

        # line-credit stance factor: graded by default (a gradient into
        # standing); `stance_soft = False` restores the hard 0.85 / 0.095 gate
        # of the in-house milestone recipe

        self.stance_soft = bool(stance_soft)

        # optional dense forward pull (the in-house rfwd term): upright forward
        # velocity pays every step, so sustained upright motion beats both the
        # statue (vx = 0) and the lunge (the uprightness gate drops). 0 = off.

        self.forward_pull = float(forward_pull)

        # per-episode time, TimeWrapper semantics (env_ssl_wrapper.context):
        # 0 on reset, +1 per control step, restarted on autoreset. Appended to
        # the flat obs as raw or fourier features (the wrapper's own
        # `fourier_time`), since EPO stacks flat state tensors.

        assert time_features in ('off', 'raw', 'fourier'), f"time_features must be off / raw / fourier, got {time_features}"
        self.time_features = time_features

        # auxiliary dense walk shaping from the inner env (alive, velocity
        # tracking, air time, foot clearance, slip): the sparse milestones pay
        # for the *task*, the shaping teaches *how* to keep stepping. Weight 0
        # restores the pure sparse task.

        self.shaping = float(shaping)

        self.env = MicroduckMlxBatchedEnv(
            Path(mjcf_path) if mjcf_path else MJCF,
            num_envs = self.num_envs,
            horizon = self.horizon,
            seed = seed,
            cpg = cpg,
            forced_swing = forced_swing,
            fixed_command = (command, 0.0),
        )
        self.env._phase_reset_gain = float(phase_reset)
        self.command = self.env._command.copy()
        self.home_ctrl = self.env.home_ctrl.copy()
        self.action_dim = self.env.action_dim

        self.learned_clock = bool(learned_clock)
        self.clock_band = (float(clock_band[0]), float(clock_band[1]))
        time_dim = {'off': 0, 'raw': 1, 'fourier': 8}[self.time_features]
        self.obs_dim = self.env.obs_dim + (4 if self.learned_clock else 0) + time_dim

        # learned clock: the policy outputs a cadence plus per-joint bias / sin /
        # cos amplitudes for the 1st and 2nd harmonics; the env integrates the
        # phase and synthesizes the joint targets from it

        self.coeff_dim = 1 + self.env.action_dim * 5 if self.learned_clock else self.env.action_dim
        self.action_dim = self.coeff_dim
        self.single_action_space = SimpleNamespace(
            shape = (self.action_dim,),
            low = np.full(self.action_dim, -1.0, np.float32),
            high = np.full(self.action_dim, 1.0, np.float32),
        )

        # per-slot episode bookkeeping
        self._steps = np.zeros(self.num_envs, np.int64)
        self._time = np.zeros(self.num_envs, np.int64)
        self._lines = np.zeros(self.num_envs, np.int64)
        self._foot_steps = np.zeros(self.num_envs, np.int64)
        self._target = np.zeros(self.num_envs)
        self._reached = np.zeros(self.num_envs, bool)
        self._cross_at = np.full(self.num_envs, -1, np.int64)
        self._bonus_paid = np.zeros(self.num_envs, bool)
        self._prev_contact = np.zeros((self.num_envs, len(self.env._foot_pair_indices)), bool)
        self._prev_x = np.zeros(self.num_envs)
        self._phase = np.zeros(self.num_envs)
        self._t = np.zeros(self.num_envs)
        self._active = np.arange(self.num_envs, dtype = np.int64)
        self._obs = np.zeros((self.num_envs, self.obs_dim), np.float32)

        # rolling stats of finished training episodes (eval does not touch these)

        self._ep_lengths: deque[int] = deque(maxlen = 8192)
        self._ep_successes: deque[int] = deque(maxlen = 8192)

        kwargs = dict(
            target = 0.66, kappa = 5.0, eps = 1e-4, window = 100, temperature = 2.0,
            uniform_prob = 0.0, log_every = 128,
        )
        kwargs.update(sgs or {})

        # the SGS wrapper is the only thing that ever picks a reset state

        self.sgs = SuccessGuidedSamplingWrapper(
            self,
            reset_configs = bank if bank is not None else make_bank(),
            success_fn = self._success,
            **kwargs,
        )
        self.bank = self.sgs.sampler

    # ------------------------------------------------------------- duck internals
    def _apply_state(self, rows, state):
        """Write one drawn (split, phase, tilt, scale, vx, target) spec onto the given rows."""
        rows = np.asarray(rows, np.int64).reshape(-1)
        split = np.asarray(state['split'], np.float64)
        phase = np.asarray(state['phase'], np.float64)
        tilt = np.asarray(state['tilt'], np.float64)
        scale = np.asarray(state['scale'], np.float64)
        vx = np.asarray(state.get('vx', 0.0), np.float64)
        target = np.asarray(state.get('target', 0.0), np.float64)
        cmd = state.get('cmd', None)

        self.env.set_reset_scale(rows, np.broadcast_to(scale, len(rows)))
        self.env.set_reset_pose(rows, split = split, phase = phase)
        self.env.set_reset_tilt(rows, pitch = tilt, roll = 0.0)
        self.env.set_reset_vel(rows, np.broadcast_to(vx, len(rows)))

        if cmd is None:
            self.env.set_reset_command(rows, np.full((len(rows), COMMAND_DIM), np.nan))
        else:
            self.env.set_reset_command(rows, np.broadcast_to(np.asarray(cmd, np.float64), (len(rows), COMMAND_DIM)))

        for i in rows:
            obs, _ = self.env.reset_one(int(i), seed = None)
            self._steps[i] = 0
            self._time[i] = 0
            self._lines[i] = 0
            self._foot_steps[i] = 0
            self._reached[i] = False
            self._cross_at[i] = -1
            self._bonus_paid[i] = False
            self._t[i] = 0.0
            self._prev_contact[i] = False
            self._prev_x[i] = float(np.asarray(self.env.eng.qpos)[i, 0])
            self._phase[i] = self.env._phi[i]
            self._obs[i] = self._augment(obs, int(i))

        self._target[rows] = np.broadcast_to(target, len(rows))

    def _augment(self, obs, i):
        return self._augment_batch(np.asarray(obs, np.float32)[None], np.asarray([i], np.int64))[0]

    def _augment_batch(self, obs, rows):
        """Proprio obs + per-row clock / time features -> (k, obs_dim), vectorized
        (the per-slot loop was the wrapper's per-step hot path)."""
        parts = [np.asarray(obs, np.float32)]

        if self.learned_clock:
            phi = self._phase[rows]
            parts.append(np.stack([np.sin(phi), np.cos(phi), np.sin(2.0 * phi), np.cos(2.0 * phi)], -1).astype(np.float32))

        if self.time_features != 'off':
            t = self._time[rows].astype(np.float32)[:, None]

            if self.time_features == 'raw':
                parts.append(t)
            else:
                # same features as env_ssl_wrapper.fourier_time (4 log-spaced freqs)
                freqs = np.exp(-np.log(10000.0) * np.arange(4, dtype = np.float32) / 4)
                args = t * freqs[None, :]
                parts.append(np.concatenate([np.sin(args), np.cos(args)], -1).astype(np.float32))

        if len(parts) == 1:
            return parts[0].astype(np.float32)

        return np.concatenate(parts, -1).astype(np.float32)

    def _standing(self, rows):
        # `_xmat_trunk` is already rolled down to the active rows, so it aligns
        # with `rows`; the trunk height comes from qpos and needs indexing

        rows = np.asarray(rows, np.int64).reshape(-1)
        trunk_up_z = np.asarray(self.env._xmat_trunk)[:, -1, -1]
        z = np.asarray(self.env.eng.qpos)[rows, SPATIAL_DIM - 1]
        return (trunk_up_z >= 0.85) & (z >= 0.095)

    def _success(self, obs, reward, terminated, truncated, info):
        return info.get('episode_success', np.zeros(self.num_envs, bool))

    # ------------------------------------------------------------- gym-style API
    def reset(self, **kwargs):
        options = kwargs.get('options') or {}
        states = options.get('reset_to_state')

        if states is None:
            # the wrapper always injects states; this arm is for direct use
            self._apply_state(np.arange(self.num_envs), dict(split = 0.0, phase = 0.0, tilt = 0.0, scale = 1.0))
        else:
            if isinstance(states, dict):
                states = [states] * self.num_envs

            for i, st in enumerate(states):
                if st is not None:
                    self._apply_state([i], st)

        return self._obs.copy(), {}

    def reset_at(self, i, options = None):
        """Slot reseed used by the SGS wrapper's autoreset path."""
        states = (options or {}).get('reset_to_state')

        if states is not None:
            self._apply_state([int(i)], states)

        return self._obs[int(i)].copy(), {}

    def _synthesize(self, coeffs, rows):
        """Oscillator coefficients -> joint targets at the current phase."""
        k = len(rows)
        d = self.env.action_dim
        c = np.asarray(coeffs, np.float64)
        lo, hi = self.clock_band
        f = lo + (hi - lo) * (np.clip(c[:, 0], -1.0, 1.0) + 1.0) / 2.0

        phi = self._phase[rows]
        cc = c[:, 1:].reshape(k, 5, d)
        a = cc[:, 0, :] + cc[:, 1, :] * np.sin(phi)[:, None] + cc[:, 2, :] * np.cos(phi)[:, None]
        a = a + cc[:, 3, :] * np.sin(2.0 * phi)[:, None] + cc[:, 4, :] * np.cos(2.0 * phi)[:, None]
        self._phase[rows] = (phi + 2.0 * np.pi * f / CONTROL_HZ) % (2.0 * np.pi)

        return a.astype(np.float32)

    def step(self, actions):
        """Full-batch gym-style step; only `self._active` rows are advanced.

        The SGS wrapper always hands a full-batch action array (one row per
        world); EPO only rolls a subset of workers, so the inactive rows are
        left untouched (their obs, reward 0, no termination).
        """
        rows = np.asarray(self._active, dtype = np.int64).reshape(-1)
        actions = np.asarray(actions, np.float32)

        if actions.ndim == 1:
            actions = actions[None]

        acts = actions[rows] if actions.shape[0] == self.num_envs else actions.reshape(len(rows), self.action_dim)

        joint_targets = self._synthesize(acts, rows) if self.learned_clock else acts
        obs, env_rew, terminated, truncated, _ = self.env.step_batch(joint_targets, worker_ids = rows)
        env_rew = np.asarray(env_rew, np.float32)

        x = np.asarray(self.env.eng.qpos)[rows, 0].astype(np.float64)
        standing = self._standing(rows)
        qpos = np.asarray(self.env.eng.qpos)[rows]
        qvel = np.asarray(self.env.eng.qvel)[rows]
        finite = np.isfinite(qpos).all(axis = 1) & np.isfinite(qvel).all(axis = 1)

        fell = np.asarray(terminated, bool) & finite
        bad = np.asarray(terminated, bool) & ~finite

        # standing-gated progress: +1 line per `line_step` metres crossed while
        # upright (and ~alive), never negative - the graded micro-credit that
        # pays for approaching the target

        forward = np.floor(np.maximum(x, 0.0) / self.line_step)
        prev_forward = np.floor(np.maximum(self._prev_x[rows], 0.0) / self.line_step)
        gained = ((forward > prev_forward) & standing).astype(np.int64)
        self._lines[rows] += gained
        self._prev_x[rows] = x

        # target crossing: the first step where standing-gated progress reaches
        # the episode's target; a capability marker, not an episode end

        progress = self._lines[rows] * self.line_step
        newly = (~self._reached[rows]) & (progress >= self._target[rows])
        self._reached[rows] |= newly

        # crossing step index (`self._steps` is bumped to the current step right
        # below, so a crossing on this step reads back as `steps - cross_at == 0`)

        self._cross_at[rows] = np.where(newly, self._steps[rows] + 1, self._cross_at[rows])

        # foot steps: air -> contact transitions, used to keep dives from
        # counting as arrivals

        contact = np.asarray(self.env.foot_contacts)[rows]
        self._foot_steps[rows] += ((~self._prev_contact[rows]) & contact).sum(axis = 1)
        self._prev_contact[rows] = contact

        self._steps[rows] += 1
        self._time[rows] += 1
        self._t[rows] += 1.0 / CONTROL_HZ

        self._obs[rows] = self._augment_batch(obs, rows)

        # gait gate: reward is only paid once the episode has taken at least
        # `min_steps_success` real foot steps, so a ballistic lunge (upright at
        # launch, airborne over the lines) earns nothing

        on_foot = self._foot_steps[rows] >= self.min_steps_success

        # graded stance credit: raw crossing events (which include the pitched
        # driving that the strict 0.85 / 0.095 gate rejects - measured ~2/3 of
        # crossings) are paid in proportion to how upright the trunk is, so the
        # line bonus is a gradient into standing instead of a cliff. The strict
        # gate still owns `_lines` / `_reached` / success accounting.

        trunk_up_z = np.asarray(self.env._xmat_trunk)[:, -1, -1]
        trunk_z = qpos[:, SPATIAL_DIM - 1]

        if self.stance_soft:
            stance = np.clip((trunk_up_z - 0.45) / 0.40, 0.0, 1.0) * np.clip((trunk_z - 0.055) / 0.040, 0.0, 1.0)
        else:
            stance = ((trunk_up_z >= 0.85) & (trunk_z >= 0.095)).astype(np.float64)

        # sustained arrival: the crossing has held for `min_sustain_steps`

        held = (self._cross_at[rows] >= 0) & (self._steps[rows] - self._cross_at[rows] >= self.min_sustain_steps)
        due = held & ~self._bonus_paid[rows] & ~fell & ~bad
        self._bonus_paid[rows] |= due

        # optional dense forward pull: upright forward velocity (world-x) pays
        # every step; the gate drops as soon as the trunk pitches

        vx = qvel[:, 0]
        rfwd = np.where(trunk_up_z >= 0.85, self.forward_pull * np.clip(vx, 0.0, None), 0.0)

        reward_rows = np.where(on_foot, (self.line_bonus * gained * stance).astype(np.float32), 0.0).astype(np.float32) \
            + np.where(due, self.target_bonus, 0.0).astype(np.float32) \
            + rfwd.astype(np.float32) \
            + np.where(fell, self.fall_penalty, 0.0).astype(np.float32) \
            + self.shaping * env_rew
        term_rows = fell | bad
        trunc_rows = (self._steps[rows] >= self.horizon) & ~term_rows

        # episode success (credited to the SGS bank cell): the target was
        # crossed on foot, at least `min_steps_success` real steps taken, and
        # the duck kept walking `min_sustain_steps` after the crossing

        done_rows = term_rows | trunc_rows
        success_rows = done_rows & held & (self._foot_steps[rows] >= self.min_steps_success) & ~bad

        for j in np.flatnonzero(done_rows):
            self._ep_lengths.append(int(self._steps[rows[j]]))
            self._ep_successes.append(int(success_rows[j]))

        # full-batch outputs (inactive rows untouched)

        reward = np.zeros(self.num_envs, np.float32); reward[rows] = reward_rows
        term = np.zeros(self.num_envs, bool); term[rows] = term_rows
        trunc = np.zeros(self.num_envs, bool); trunc[rows] = trunc_rows
        done = np.zeros(self.num_envs, bool); done[rows] = term_rows | trunc_rows
        success = np.zeros(self.num_envs, bool); success[rows] = success_rows
        prog_out = np.zeros(self.num_envs, np.float32); prog_out[rows] = progress
        steps_out = np.zeros(self.num_envs, np.int64); steps_out[rows] = self._steps[rows]

        info = dict(
            episode_done = done,
            episode_success = success,
            episode_progress = prog_out,
            episode_target = self._target.copy(),
            episode_steps = steps_out,
        )

        return self._obs.copy(), reward, term, trunc, info

    def spawn(self, **overrides):
        """A sibling env sharing this one's task config - e.g. the no-CPG student:
        `env.spawn(num_envs = 256, cpg = False, learned_clock = False)`."""

        return DuckGaitVecEnv(**{**self._ctor_kwargs, **overrides})

    def close(self):
        self.env.close()

    # ------------------------------------------------------------- EPO interface
    def reset_one(self, worker, seed = None):
        return self._obs[int(worker)].copy(), {}

    def step_batch(self, actions, worker_ids = None):
        self._active = np.arange(self.num_envs, dtype = np.int64) if worker_ids is None else np.asarray(list(worker_ids), np.int64)
        full = np.zeros((self.num_envs, self.action_dim), np.float32)
        full[self._active] = np.asarray(actions, np.float32).reshape(len(self._active), self.action_dim)
        obs, reward, terminated, truncated, info = self.sgs.step(full)
        rows = self._active
        return obs, reward[rows], terminated[rows], truncated[rows], {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == self.num_envs else v) for k, v in info.items()}

    # ------------------------------------------------------------- helpers
    def new_cycle(self):
        pass

    def set_attr(self, key, value):
        setattr(self, key, value)

    def to_epo_agent(self, **kwargs):
        return build_epo_agent(self.obs_dim, self.action_dim, kwargs)

    @property
    def ep_stats(self):
        """(mean steps, success rate) over recently finished training episodes."""

        if not self._ep_lengths:
            return dict(ep_steps = float(self.horizon), ep_success = 0.0)

        return dict(
            ep_steps = float(np.mean(self._ep_lengths)),
            ep_success = float(np.mean(self._ep_successes)),
        )

    @property
    def probs(self):
        return self.sgs.probs

    @property
    def success_rates(self):
        return self.sgs.success_rates

    @property
    def counts(self):
        return self.sgs.counts

# agent

def build_epo_agent(obs_dim, action_dim, kwargs = None):
    kwargs = dict(kwargs or {})
    critic_kwargs = kwargs.pop('critic_kwargs', dict(use_regression = True))

    defaults = dict(
        num_latents = 8,
        dim_latent = 32,
        actor_dim = 128,
        actor_mlp_depth = 2,
        critic_dim = 128,
        critic_mlp_depth = 2,
        use_state_norm = False,
        wrap_with_accelerate = 'RANK' in os.environ,
        # a slight overshoot on the beta mean -> env action transform (mean-conc-beta 0.2.7),
        # so a saturated mean can actually reach the action bounds
        actor_kwargs = dict(beta_kwargs = dict(overshoot = 0.1)),
        latent_gene_pool_kwargs = dict(
            frac_natural_selected = 0.5,
            frac_tournaments = 0.5,
            frac_elitism = 0.25,
            frozen_latents = True,
        ),
        critic_kwargs = critic_kwargs,
    )
    defaults.update(kwargs)

    return create_agent(
        dim_state = obs_dim,
        actor_num_actions = action_dim,
        action_is_continuous = True,
        **defaults
    )

def init_gait_agent(agent, resume_from = ''):
    """Resume actor (and critic / latents) from a checkpoint, or zero the action
    head (no bias) so the initial *mean* policy is exactly the home pose."""

    if resume_from:
        pkg = torch.load(str(resume_from), map_location = 'cpu', weights_only = True)
        agent.unwrap_model(agent.actor).load_state_dict(pkg['actor'])

        for key, name in (('critic', 'critic'), ('latents', 'latent_gene_pool')):
            module = getattr(agent, name, None)

            if pkg.get(key) is not None and module is not None:
                agent.unwrap_model(module).load_state_dict(pkg[key])

        return f'resumed from {resume_from}'

    for module in agent.unwrap_model(agent.actor).to_out.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)

    return 'fresh (zeroed action head)'

# rendering

def load_agent(checkpoint: str, num_latents: int = 0):

    pkg = torch.load(str(checkpoint), map_location = 'cpu', weights_only = True)
    obs_dim = int(pkg['actor']['init_layer.0.weight'].shape[1])
    num_actions = int(pkg['actor']['to_out.1.weight'].shape[0]) // 2

    if num_latents <= 0 and pkg.get('latents') is not None:
        num_latents = int(pkg['latents']['latents'].shape[0])

    num_latents = num_latents if num_latents > 0 else 32

    agent = build_epo_agent(
        obs_dim,
        num_actions,
        dict(num_latents = num_latents, dim_latent = 32, use_critic_ema = False, use_hidden_spr = False, use_diversity_discr = False, quiet = True),
    )

    agent.unwrap_model(agent.actor).load_state_dict(pkg['actor'])
    agent.unwrap_model(agent.critic).load_state_dict(pkg['critic'])

    if pkg.get('latents') is not None and agent.has_latent_genes:
        agent.unwrap_model(agent.latent_gene_pool).load_state_dict(pkg['latents'])

    agent.eval()
    return agent, obs_dim

def render_gait(
    checkpoint: str = 'experiments/gait_sgs_wrapped/ckpt.pt',
    out_path: str = 'experiments/gait_sgs_wrapped/videos/gait.mp4',
    start: str = '0@0',
    tilt: float = 0.0,
    scale: float = 0.0,
    clock_band = '0.3,3.0',
    horizon: int = 400,
    seed: int = 0,
    latent_id: int = 0,
    num_latents: int = 0,
    sample: bool = False,
    learned_clock: bool = False,
    cpg: bool = True,
    forced_swing: float = 0.25,
    time_features: str = 'fourier',
):
    split_s, _, phase_s = str(start).partition('@')
    split, phase = float(split_s), float(phase_s or 0.0)
    band = _floats(clock_band)
    band = (band[0], band[1]) if len(band) >= 2 else (0.3, 3.0)

    env = DuckGaitVecEnv(num_envs = 1, horizon = horizon, seed = seed, learned_clock = learned_clock, clock_band = band,
                         cpg = cpg, forced_swing = forced_swing, time_features = time_features)
    env._apply_state([0], dict(split = split, phase = phase, tilt = tilt, scale = scale, target = 0.0))

    agent, obs_dim = load_agent(checkpoint, num_latents = num_latents)

    if obs_dim != env.obs_dim:
        raise SystemExit(f'checkpoint expects obs_dim {obs_dim}, env provides {env.obs_dim}')

    m = env.env.model
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height = VIDEO_SIZE[0], width = VIDEO_SIZE[1])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAMERA_LOOKAT
    cam.distance = CAMERA_DISTANCE
    cam.azimuth = CAMERA_AZIMUTH
    cam.elevation = CAMERA_ELEVATION

    frames = []
    trace = []

    for step_i in range(horizon):
        state = torch.from_numpy(np.ascontiguousarray(env._obs[[0]])).float()

        with torch.no_grad():
            action = agent.get_actor_actions(state, latent_id = latent_id, sample = sample, use_unwrapped_model = True)

        if isinstance(action, tuple):
            action = action[0]

        env._active = np.array([0], np.int64)

        if env.learned_clock:
            joint_targets = env._synthesize(action.numpy().astype(np.float32), env._active)
        else:
            joint_targets = action.numpy().astype(np.float32)

        # step the inner env directly so a fall is the rendered final frame
        # (through the SGS wrapper the slot would be reseeded first)

        obs, _, term, trunc, _ = env.env.step_batch(joint_targets, worker_ids = env._active)
        env._time[0] += 1
        env._obs[0] = env._augment(obs[0], 0)

        d.qpos[:] = np.asarray(env.env.eng.qpos)[0]
        d.qvel[:] = np.asarray(env.env.eng.qvel)[0]
        mujoco.mj_forward(m, d)

        trace.append(dict(
            step = step_i,
            x = round(float(d.qpos[0]), 4),
            z = round(float(d.qpos[2]), 4),
            uprightness = round(float(np.asarray(env.env.eng.xmat)[0].reshape(m.nbody, 3, 3)[TRUNK_BODY, -1, -1]), 4),
        ))

        cam.lookat[0] = d.qpos[0] + 0.3
        renderer.update_scene(d, camera = cam)
        frames.append(renderer.render())

        if bool(term[0]):
            break

    out = Path(out_path)
    out.parent.mkdir(parents = True, exist_ok = True)
    imageio.mimsave(out, frames, fps = CONTROL_HZ)

    trace_path = out.with_suffix('.trace.json')
    trace_path.write_text(json.dumps(dict(
        checkpoint = str(checkpoint), start = str(start), tilt = tilt, scale = scale,
        latent_id = latent_id, frames = len(frames),
        x_end = trace[-1]['x'] if trace else None,
        min_uprightness = round(min(t['uprightness'] for t in trace), 3) if trace else None,
        trace = trace,
    ), indent = 1))

    print(f'saved {out} ({len(frames)} frames, final x {trace[-1]["x"] if trace else float("nan"):+.2f})')

    renderer.close()
    env.close()

# trainer

def main(
    save_path: str = 'experiments/gait_sgs_wrapped/ckpt.pt', log_path: str = 'experiments/gait_sgs_wrapped/metrics.jsonl',
    total_env_steps: int = 2_000_000, num_envs: int = 1024, num_latents: int = 32, episodes_per_latent: int = 32,
    # task
    horizon: int = 400, command_speed: float = 0.15, line_step: float = 0.05,
    line_bonus: float = 0.1, target_bonus: float = 1.0, fall_penalty: float = 0.0,
    min_steps_success: int = 3, min_sustain_steps: int = 50,
    stance_soft: bool = True, forward_pull: float = 0.0, shaping: float = 1.0,
    # exploration annealing + the SGS bank
    temp_start: float = 0.05, temp_end: float = 1.0, temp_cycles: int = 50, temp_gate_steps: int = 250,
    targets: str = '0.05,0.15,0.3,0.5,0.8', bank_kind: str = 'tiers',
    primed_splits: str = '0.0,0.15,0.3', bank_tiers: int = 6,
    walk_vxs: str = '0.0,0.05,0.10', walk_phases: str = '0,1.5708,3.1416,4.7124', walk_split_amp: float = 0.15,
    stand_scales: str = '0.0,0.5,1.0', stand_commands: str = '',
    time_features: str = 'off', cpg: bool = False, forced_swing: float = 0.25, phase_reset: float = 0.0,
    learned_clock: bool = True, clock_band = (0.3, 3.0),
    target: float = 0.66, kappa: float = 5.0,
    sgs_eps: float = 1e-4, sgs_window: int = 100, temperature: float = 2.0, sgs_uniform_prob: float = 0.0,
    # optimization
    eval_every: int = 3, eval_episodes: int = 128, render: bool = True,
    epochs: int = 4, batch_size: int = 1024, learning_rate: float = 5e-4, entropy_weight: float = 0.002,
    actor_mlp_depth: int = 2, actor_dim: int = 128, critic_mlp_depth: int = 2, critic_dim: int = 128,
    use_hidden_spr: bool = True, hidden_spr_weight: float = 1.0,
    max_grad_norm: float = 0.5, resume_from: str = '',
    # online DAgger: a no-CPG student distills from the CPG teacher
    distill: bool = False, student_envs: int = 256, student_rollout_steps: int = 200, student_rollout_every: int = 1,
    distill_epochs: int = 2, distill_batch_size: int = 1024, distill_samples: int = 65536,
    distill_lr: float = 5e-4, distill_buffer: int = 200000,
    student_save_path: str = '', student_resume_from: str = '',
    rep_align: bool = False, rep_align_weight: float = 1.0, seed: int = 42,
):
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))

    if distill and (not cpg or learned_clock):
        # the teacher labels student states by splicing in its own CPG phase:
        # teacher obs must be exactly student obs + 2 cpg dims

        print('[SGS-gait] distill: forcing cpg True / learned_clock False (teacher obs = student obs + CPG phase)')
        cpg, learned_clock = True, False

    save_path, log_path = Path(save_path).resolve(), Path(log_path).resolve()
    save_path.parent.mkdir(parents = True, exist_ok = True); log_path.parent.mkdir(parents = True, exist_ok = True)
    log_path.write_text('')

    bank = make_gait_bank(bank_kind, targets = targets, stand_scales = stand_scales, stand_commands = stand_commands,
                          primed_splits = primed_splits, bank_tiers = bank_tiers, walk_vxs = walk_vxs,
                          walk_phases = walk_phases, walk_split_amp = walk_split_amp)

    sgs_kwargs = dict(target = target, kappa = kappa, eps = sgs_eps, window = sgs_window, temperature = temperature,
                      uniform_prob = sgs_uniform_prob, log_every = 128, log_path = str(log_path.with_name('sgs_progression.csv')), seed = seed)

    env = DuckGaitVecEnv(
        num_envs = num_envs, horizon = horizon, seed = seed, command = command_speed, learned_clock = learned_clock,
        clock_band = clock_band, line_step = line_step, line_bonus = line_bonus, target_bonus = target_bonus,
        fall_penalty = fall_penalty, min_steps_success = min_steps_success, min_sustain_steps = min_sustain_steps,
        stance_soft = stance_soft, forward_pull = forward_pull, time_features = time_features, cpg = cpg,
        forced_swing = forced_swing, phase_reset = phase_reset, shaping = shaping, bank = bank, sgs = sgs_kwargs,
    )

    print(f'[SGS-gait] {num_envs} ducks, {len(bank)} cells = initial states x targets {targets}, SGS target {target} '
          f'kappa {kappa} eps {sgs_eps}; success = cross the cell target on foot (>= {min_steps_success} steps, '
          f'sustained >= {min_sustain_steps}) before the episode ends')

    agent = env.to_epo_agent(
        num_latents = num_latents, dim_latent = 32, actor_dim = actor_dim, actor_mlp_depth = actor_mlp_depth,
        critic_dim = critic_dim, critic_mlp_depth = critic_mlp_depth, use_critic_ema = False, clip_value = False,
        use_hidden_spr = use_hidden_spr, hidden_spr_weight = hidden_spr_weight, use_diversity_discr = False,
        actor_lr = learning_rate, critic_lr = learning_rate, batch_size = batch_size, max_grad_norm = max_grad_norm,
        actor_loss_kwargs = dict(entropy_weight = entropy_weight, eps_clip = 0.2, norm_advantages = True), quiet = True,
    )

    print(f'[SGS-gait] teacher {init_gait_agent(agent, resume_from)}')

    epo = EPO(agent, episodes_per_latent = episodes_per_latent, max_episode_length = horizon, action_sample_temperature = temp_start)

    # online DAgger: the student lives in the same task without the CPG, sees the
    # same commanded-goal bank, and learns from the teacher's labels on its own
    # visited states - optionally with representation alignment (x-ppo 0.0.12)

    student = student_env = student_optim = distiller = None

    if distill:
        student_env = env.spawn(num_envs = student_envs, seed = seed + 7919, cpg = False, learned_clock = False, phase_reset = 0.0,
                                sgs = dict(sgs_kwargs, log_every = None, log_path = None, seed = seed + 7919))

        student = student_env.to_epo_agent(
            num_latents = num_latents, dim_latent = 32, actor_dim = actor_dim, actor_mlp_depth = actor_mlp_depth,
            use_critic_ema = False, clip_value = False, use_hidden_spr = False, hidden_spr_weight = 0.0,
            use_diversity_discr = False, quiet = True,
        )

        print(f'[SGS-gait] student {init_gait_agent(student, student_resume_from)}')

        student_optim = torch.optim.Adam(student.unwrap_model(student.actor).parameters(), lr = distill_lr)

        # cosine-match the student's hidden embedding right before its action
        # head - the actor.mlp output, the embedding hidden_spr rides on

        rep_align_kwargs = dict(
            student_layer = student.unwrap_model(student.actor).mlp,
            teacher_layer = agent.unwrap_model(agent.actor).mlp,
            rep_align_weight = rep_align_weight,
        ) if rep_align else dict()

        if rep_align:
            print(f'[SGS-gait] representation alignment on: student/teacher actor.mlp hooked (cosine), weight {rep_align_weight}')

        from x_ppo import DaggerDistiller

        def student_dist(state):
            return student.get_actor_distribution(state, latent_id = 0, use_unwrapped_model = True)

        def teacher_dist(teacher_state):
            return agent.get_actor_distribution(teacher_state, latent_id = 0, use_unwrapped_model = True)

        def teacher_view(state, info, step):
            # the teacher sees the CPG phase spliced in after the command dims
            t_sec = torch.as_tensor(student_env._time, dtype = torch.float64, device = state.device) / CONTROL_HZ
            phi = (2.0 * torch.pi * CPG_FREQ * t_sec) % (2.0 * torch.pi)
            cpg = torch.stack([phi.sin(), phi.cos()], -1).to(state.dtype)
            return torch.cat([state[:, :50], cpg, state[:, 50:]], dim = -1)

        def to_student_space(state, action, info, step):
            # the teacher's effective control (residual + wave / ACTION_SCALE)
            # mapped back into the student's action space
            t_sec = student_env._time.astype(np.float64) / CONTROL_HZ
            cmd = state[:, 48:50].detach().cpu().numpy().astype(np.float64)
            wave = torch.as_tensor(env.env.cpg_waveform_at(t_sec, cmd), device = state.device, dtype = torch.float64)
            return (torch.clamp(torch.clamp(1.5 * action.double(), -1.0, 1.0) + wave / ACTION_SCALE, -1.0, 1.0) / 1.5).float()

        def student_env_step(actions):
            student_env._active = np.arange(student_env.num_envs, dtype = np.int64)
            return student_env.sgs.step(np.asarray(actions, dtype = np.float32))

        distiller = DaggerDistiller(
            student = student_dist, teacher = teacher_dist, env = student_env,
            teacher_state_fn = teacher_view, label_transform = to_student_space,
            optimizer = student_optim, env_step_fn = student_env_step, **rep_align_kwargs,
            buffer_size = distill_buffer, max_samples = distill_samples, batch_size = distill_batch_size,
            epochs = distill_epochs, max_grad_norm = 1.0,
        )

        print(f'[SGS-gait] online DAgger (x-ppo 0.0.12): no-CPG student ({student_env.obs_dim} obs / {student_env.action_dim} act)')

    pbar = tqdm(total = total_env_steps, desc = 'SGS gait (single duck)', unit = 'step', dynamic_ncols = True)
    total_steps = cycle = 0
    ep_steps_recent, best_walk, t_start = None, -float('inf'), time.time()

    try:
        while total_steps < total_env_steps:
            cycle += 1

            # exploration annealing: near-deterministic actions first (episodes
            # reach the horizon and milestones accumulate), opening up as the
            # gait forms - held back while the policy cannot survive yet

            if temp_cycles > 0:
                want = temp_start + (temp_end - temp_start) * min(1.0, cycle / float(temp_cycles))

                if ep_steps_recent is not None and ep_steps_recent < temp_gate_steps:
                    want = min(want, float(epo.action_sample_temperature))

                epo.action_sample_temperature = want

            t0 = time.perf_counter()
            m = epo.gather_experience_from(env)
            t_gather = time.perf_counter() - t0

            total_steps += len(m.memories)
            t0 = time.perf_counter()
            _, fitness = agent.learn_from(m, epochs = epochs)
            t_learn = time.perf_counter() - t0

            mean_reward = float(m.cumulative_rewards.mean().item())
            ep_steps_recent = float(env.ep_stats['ep_steps'])
            best_latent = int(fitness.argmax().item()) if (fitness is not None and agent.has_latent_genes) else 0

            # DAgger step: the student rolls out on its own no-CPG states, the
            # teacher labels them, the student fits the growing buffer

            s_loss, s_breakdown, t_distill = float('nan'), None, 0.0

            if distill and (cycle % student_rollout_every == 0 or cycle % eval_every == 0):
                t0 = time.perf_counter()
                s_stats = distiller.step(collect_steps = student_rollout_steps, return_loss_breakdown = True)
                s_loss, s_breakdown = s_stats['loss'], s_stats.get('loss_breakdown')
                t_distill = time.perf_counter() - t0

            probs, rates, counts = env.probs.cpu().numpy(), env.success_rates.cpu().numpy(), env.counts.cpu().numpy()
            frontier = int(np.nan_to_num(probs, nan = -1).argmax())
            spec = env.bank.draw_state(frontier) if probs[frontier] > 1e-9 else dict(split = 0., phase = 0., tilt = 0., scale = 1., target = 0.)
            scored = (counts > 0) & (rates >= 0)

            record = dict(
                cycle = cycle, total_steps = total_steps, mean_reward = mean_reward, best_latent = best_latent,
                wall_time = round(time.time() - t_start, 1), temperature = float(epo.action_sample_temperature),
                gather_time = round(t_gather, 2), learn_time = round(t_learn, 2), distill_time = round(t_distill, 2),
                student_loss = s_loss, student_samples = int(min(distiller.count, distill_buffer)) if distill else 0,
                student_policy = float(s_breakdown.policy) if s_breakdown is not None else float('nan'),
                student_align = float(s_breakdown.alignment) if (s_breakdown is not None and s_breakdown.alignment is not None) else float('nan'),
                sgs_frontier = frontier, sgs_frontier_spec = spec, sgs_visited = int((counts > 0).sum()),
                sgs_success_cells = int((scored & (rates > 0)).sum()),
                sgs_best_rate = float(rates[scored].max()) if scored.any() else 0.0,
                sgs_probs = np.round(probs, 4).tolist(), sgs_rates = np.round(np.nan_to_num(rates, nan = -1.0), 3).tolist(),
                sgs_counts = counts.tolist(), episodes_done = int(env.sgs.sampler.step_count), **env.ep_stats,
            )

            if cycle % eval_every == 0 or total_steps >= total_env_steps:
                ev = evaluate(env, agent, num_episodes = eval_episodes, horizon = horizon, latent_id = best_latent)
                record.update(ev)
                st_note = ''

                if distill:
                    # the student is evaluated with the teacher, same protocol
                    st_ev = evaluate(student_env, student, num_episodes = eval_episodes, horizon = horizon, latent_id = 0)
                    record.update({f'st_{k[3:]}': v for k, v in st_ev.items()})
                    student.save(Path(student_save_path) if student_save_path else save_path.with_name('student.pt'), overwrite = True)

                    align = s_breakdown.alignment if s_breakdown is not None else None
                    st_note = (f' | student: x={record["st_x"]:+.3f} lines={record["st_lines"]:.1f} reach={record["st_reach"]:.2f} '
                               f'falls={record["st_falls"]:.2f} loss={s_loss:.2f}' + (f' align={float(align):.3f}' if align is not None else ''))

                with log_path.open('a') as f:
                    f.write(json.dumps(record) + '\n')

                if ev['ev_x'] > best_walk:
                    best_walk = ev['ev_x']
                    agent.save(save_path, overwrite = True)

                agent.save(save_path.with_name('ckpt_last.pt'), overwrite = True)
                env.sgs.reset()  # the eval scrambled the slots - reseed fresh SGS episodes

                tqdm.write(
                    f'[c{cycle:04d} | {total_steps/1e6:.2f}M] ret={mean_reward:+.2f} ep_steps={record["ep_steps"]:.0f} succ={record["ep_success"]:.2f} '
                    f'| eval: x={ev["ev_x"]:+.3f} lines={ev["ev_lines"]:.1f} surv={ev["ev_survival"]:.2f} falls={ev["ev_falls"]:.2f} '
                    f'| sgs: {record["sgs_visited"]}/{len(probs)} cells, best rate {record["sgs_best_rate"]:.2f}, '
                    f'frontier #{frontier} (sc {spec["scale"]:.2f} tgt {spec["target"]:.2f}) p={probs[frontier]:.3f}{st_note}'
                )
            else:
                with log_path.open('a') as f:
                    f.write(json.dumps(record) + '\n')

            pbar.update(min(len(m.memories), max(0, total_env_steps - pbar.n)))
            pbar.set_postfix(cycle = cycle, ret = f'{mean_reward:+.2f}', sgs = f'{record["sgs_visited"]}/{len(probs)}')

    finally:
        agent.save(save_path.with_name('ckpt_last.pt'), overwrite = True)
        pbar.close()
        print(f'\n[SGS-gait] finished at {total_steps:,} steps in {time.time() - t_start:.0f}s')

    # on a mac (mps), show the gait - one full episode each, from the clean stand

    if render and torch.backends.mps.is_available():
        videos = save_path.with_name('videos')
        videos.mkdir(exist_ok = True)
        student_ckpt = Path(student_save_path) if student_save_path else save_path.with_name('student.pt')

        for name, ckpt, cpg_mode in (('teacher', save_path, cpg), ('student', student_ckpt, False)):
            if name == 'student' and not distill:
                continue

            render_gait(
                checkpoint = str(ckpt), out_path = str(videos / f'{name}.mp4'), cpg = cpg_mode, horizon = horizon,
                learned_clock = learned_clock if name == 'teacher' else False, time_features = time_features,
                latent_id = best_latent if name == 'teacher' else 0,
            )
            print(f'[SGS-gait] gait video: {videos / f"{name}.mp4"}')

if __name__ == '__main__':
    fire.Fire(main)
