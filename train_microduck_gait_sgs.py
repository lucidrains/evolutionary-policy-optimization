#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["fire", "mujoco", "mlx", "numpy", "torch", "tqdm", "einx", "imageio",
#     "evolutionary-policy-optimization", "mujoco_warp_lucidrains_fork[mlx,nim]==0.0.6", "env-ssl-wrapper>=0.5.13", "x-ppo==0.0.27", "nimporter-plus", "setuptools"]
# [tool.uv.sources]
# evolutionary-policy-optimization = { path = ".", editable = true }
# mujoco_warp_lucidrains_fork = { path = "../mujoco_warp", editable = true }
# ///
"""Single microduck gait, sparse gait rewards, SGS-chosen initial states.

Success = crossing the cell's target distance on foot while upright; the
sampler concentrates training on the frontier cells. By default the actor is
conditioned on a fixed latent sampled from N(0, I) per episode and DADS rewards
the discriminator's ability to recover it; pass --latent_mode=gene_pool to use
the original evolved EPO pool. With --distill, a no-CPG student learns from the
CPG teacher's labels on its own visited states (online DAgger via x-ppo 0.0.27),
optionally with representation alignment. Pass --use_contact_aux to train an
anticipatory contact head on the actor / critic hidden states.

The CPG clock and sensory-feedback primitives live in cpg.py at the project
root; the duck's leg-pattern table lives here. `--cpg_adaptive` learns the
cadence online from foot contacts. On a Mac the run ends by rendering the gait
into <run>/videos/ - fixed latents render the top eval latents as teacher_z*.mp4.

  uv run train_microduck_gait_sgs.py --total_env_steps 2000000

  # original evolved EPO pool
  uv run train_microduck_gait_sgs.py --latent_mode=gene_pool \
    --num_latents=32 --episodes_per_latent=32

CPU-only / force the compiled Nim fast paths:

  uv run train_microduck_gait_sgs.py --nim_cpu True --total_env_steps 2000000
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
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from env_ssl_wrapper import SuccessGuidedSamplingWrapper
from mujoco_warp.mps.robot import BatchedEngine
from tqdm import tqdm

from cpg import CPG, AdaptiveFrequencyOscillator
from evolutionary_policy_optimization import (
    EPO,
    create_agent,
    rescale_from_to,
)

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
CPG_KNEE_SCALE = 1.6
CPG_ANKLE_SCALE = 0.55
CPG_ROLL_AMP = 0.035
CPG_RAMP_CYCLES = 1.0
CPG_TURN_GAIN = 0.1
CPG_DRIVE_LPF_HZ = 4.0
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

def cpg_waveform(cpg, phi, steps, command, joints, dim, swing = None, freq = None):
    """The duck leg pattern on the CPG's sine basis (axis-mirrored hips and
    ankles, half-wave knees, quarter-phase rolls, turn offset on hip pitch)."""

    swing = FORCED_SWING if swing is None else swing
    k = len(phi)
    command = np.asarray(command, np.float64)
    freq = np.asarray(CPG_FREQ if freq is None else freq, np.float64)

    t = np.asarray(steps, np.float64) * (PHYSICS_DT * N_SUBSTEPS)
    ramp = np.minimum(1., t * freq / CPG_RAMP_CYCLES)
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

class MicroduckMlxEngine(BatchedEngine):
  """MLX engine plus the row/state helpers the env uses."""

  backend = 'mlx'

  def set_ctrl(self, ctrl):
    self.ctrl = mx.array(np.asarray(ctrl, np.float32))

  def sync(self):
    mx.eval(self.qpos, self.qvel, self.xmat, self.xpos, self.cvel, self.xipos, self.cinert, self.geom_xpos, self.subtree_com)

  def set_rows(self, rows, qpos, qvel):
    for i, q, v in zip(np.asarray(rows, np.int64).reshape(-1), np.asarray(qpos, np.float32), np.asarray(qvel, np.float32)):
      self.qpos = self.qpos.at[i].subtract(self.qpos[i]).at[i].add(mx.array(q))
      self.qvel = self.qvel.at[i].subtract(self.qvel[i]).at[i].add(mx.array(v))
      self.warm = self.warm.at[i].subtract(self.warm[i])

  def snapshot_rows(self, rows):
    mx.eval(self.qpos, self.qvel)
    return np.asarray(self.qpos)[rows].copy(), np.asarray(self.qvel)[rows].copy()

  def restore_rows(self, rows, qpos, qvel):
    q, v = np.array(self.qpos), np.array(self.qvel)
    q[rows], v[rows] = qpos, qvel
    self.qpos, self.qvel = mx.array(q), mx.array(v)

  def foot_contacts(self, geoms):
    if not hasattr(self, 'pair_conct'):
      return np.zeros((self.nworld, len(geoms)), bool)

    pc, g1, g2 = np.asarray(self.pair_conct), self.rig.pair_g1, self.rig.pair_g2
    return np.stack([(pc[:, (g1 == g) | (g2 == g)] > 0).any(axis = 1) for g in geoms], axis = 1)


class MicroduckCpuEngine:
  """`mujoco_warp` CPU engine with the Nim fast paths, same row/state surface."""

  backend = 'cpu'

  def __init__(self, model, nworld):
    import mujoco_warp
    from mujoco_warp._src import nim_cpu

    # large models use the sparse constraint layout + Nim sparse/tree solver
    sparse = model.nv > 32
    self.mjw, self.m = mujoco_warp, mujoco_warp.put_model(model, nim_sparse = sparse, nim_tree = sparse)
    self.d = mujoco_warp.make_data(model, nworld = nworld)
    self.nworld, self.nim_available = nworld, nim_cpu.get_ops() is not None

  qpos = property(lambda self: self.d.qpos.numpy())
  qvel = property(lambda self: self.d.qvel.numpy())
  xmat = property(lambda self: self.d.xmat.numpy())
  xpos = property(lambda self: self.d.xpos.numpy())
  cvel = property(lambda self: self.d.cvel.numpy())
  xipos = property(lambda self: self.d.xipos.numpy())
  cinert = property(lambda self: self.d.cinert.numpy())
  subtree_com = property(lambda self: self.d.subtree_com.numpy())
  geom_xpos = property(lambda self: self.d.geom_xpos.numpy())

  def set_ctrl(self, ctrl):
    self.d.ctrl.numpy()[:] = np.asarray(ctrl, np.float32)

  def step(self, kin = False):
    self.mjw.step(self.m, self.d)

  def sync(self):
    pass

  def set_rows(self, rows, qpos, qvel):
    self.d.qpos.numpy()[rows] = np.asarray(qpos, np.float32)
    self.d.qvel.numpy()[rows] = np.asarray(qvel, np.float32)
    self.d.qacc_warmstart.numpy()[rows] = 0.0

  def snapshot_rows(self, rows):
    return self.qpos[rows].copy(), self.qvel[rows].copy()

  def restore_rows(self, rows, qpos, qvel):
    self.d.qpos.numpy()[rows] = np.asarray(qpos, np.float32)
    self.d.qvel.numpy()[rows] = np.asarray(qvel, np.float32)

  def foot_contacts(self, geoms):
    nacon = min(int(self.d.nacon.numpy()[0]), self.d.contact.geom.shape[0])
    geom, world = self.d.contact.geom.numpy()[:nacon], self.d.contact.worldid.numpy()[:nacon]
    out = np.zeros((self.nworld, len(geoms)), bool)

    for j, g in enumerate(geoms):
      out[world[(geom == g).any(axis = 1)], j] = True

    return out

  def set_world_tables(self, *args, **kwargs):
    raise NotImplementedError('CPU backend is flat-floor only')


def make_engine(model, nworld, backend = 'mlx', coop = True):
  if backend == 'mlx':
    # packed-tree solve is faster once the cooperative kernel is off (large models)
    return MicroduckMlxEngine(model, nworld = nworld, coop = coop, tree = model.nv >= 64)

  if backend == 'cpu':
    return MicroduckCpuEngine(model, nworld)

  raise ValueError(f"backend must be 'mlx' or 'cpu', got {backend!r}")


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
    cpg_adaptive: bool = False,
    cpg_adapt_rate: float = 10.0,
    cpg_freq_range = (0.5, 4.0),
    forced_swing: float | None = FORCED_SWING,
    fixed_command: tuple | None = None,
    rescale_to=ACTION_INTERFACE_RANGE,
    coop: bool = True,
    backend: str = 'mlx',
  ):
    import mujoco

    self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    self.model.opt.timestep = PHYSICS_DT
    self.model.actuator_gainprm[:, 0] *= kp_mult
    self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]
    self.model.actuator_biasprm[:, 2] = -kv

    self.eng = make_engine(self.model, num_envs, backend = backend, coop = coop)
    self.backend = backend
    self.num_envs = num_envs
    self.nv = self.model.nv
    self.action_dim = self.model.nu
    self.horizon = horizon
    self.max_episode_length = horizon
    self.cpg = cpg
    self.cpg_adaptive = bool(cpg_adaptive)
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

    # robot geoms follow the single floor geom, grouped by body id 1..nbody-1

    geom_bodyid = self.model.geom_bodyid[1:]
    assert (geom_bodyid > 0).all()
    self._body_geom_starts = np.array(
      [int(np.flatnonzero(geom_bodyid == b)[0]) for b in range(1, self.model.nbody)], dtype=np.int64
    )
    self._non_foot_body_idx0 = self._non_foot_bodies - 1
    self._body_contact_z_thresh = 0.035
    self.num_contact_bodies = self.model.nbody - 1

    joint_id = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) - 1  # noqa: E731
    self.l_hip_pitch = joint_id("left_hip_pitch")
    self.r_hip_pitch = joint_id("right_hip_pitch")
    self.l_knee = joint_id("left_knee")
    self.r_knee = joint_id("right_knee")
    self.l_ankle = joint_id("left_ankle")
    self.r_ankle = joint_id("right_ankle")
    self.l_hip_roll = joint_id("left_hip_roll")
    self.r_hip_roll = joint_id("right_hip_roll")

    # phase clock; the duck's leg-pattern table rides its sine basis

    self.cpg_gen = CPG(freq = CPG_FREQ, dt = PHYSICS_DT * N_SUBSTEPS)

    # online cadence adaptation from the contact difference (zero-mean per stride)

    self.cpg_afo = AdaptiveFrequencyOscillator(
      num = num_envs, freq = CPG_FREQ, dt = PHYSICS_DT * N_SUBSTEPS,
      adapt_rate = cpg_adapt_rate, freq_range = cpg_freq_range, freq_tau = 2.0,
    ) if self.cpg_adaptive else None
    self._cpg_drive = np.zeros(num_envs, np.float64)
    self._cpg_drive_alpha = 1.0 - math.exp(-2.0 * math.pi * CPG_DRIVE_LPF_HZ * PHYSICS_DT * N_SUBSTEPS)
    self._cpg_adapt_mask = np.zeros(num_envs, bool)

    self.cpg_joints = dict(
      l_hip_pitch = self.l_hip_pitch, r_hip_pitch = self.r_hip_pitch, l_knee = self.l_knee, r_knee = self.r_knee,
      l_ankle = self.l_ankle, r_ankle = self.r_ankle, l_hip_roll = self.l_hip_roll, r_hip_roll = self.r_hip_roll,
    )

    self._last_action = np.zeros((num_envs, self.action_dim), np.float64)
    self._ctrl_np = np.zeros((num_envs, self.action_dim), np.float32)
    self._command = np.zeros((num_envs, COMMAND_DIM), np.float64)
    self._steps = np.zeros(num_envs, np.int64)
    self._episodes = np.zeros(num_envs, np.int64)
    self._foot_air = np.zeros((num_envs, NUM_FEET), np.float64)
    self._phi = np.zeros(num_envs, np.float64)
    self._rngs = [np.random.default_rng(seed + i) for i in range(num_envs)]

    self._reset_scale = np.ones(num_envs)
    self._reset_split = np.zeros(num_envs)
    self._reset_phase = np.zeros(num_envs)
    self._reset_tilt = np.zeros((num_envs, 2))
    self._reset_vel = np.zeros(num_envs)
    self._reset_cmd = np.full((num_envs, COMMAND_DIM), np.nan)
    self._phase_reset_gain = 0.0
    self._phase_prev_contact = np.zeros((num_envs, NUM_FEET), bool)

    for i in range(num_envs):
      self.reset_one(i, seed=seed + i)

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

  def _cpg_waveform(self, phi, steps, command, freq = None):
    return cpg_waveform(
      self.cpg_gen, phi, steps, command, self.cpg_joints, self.action_dim, self.forced_swing, freq = freq
    )

  def cpg_waveform_at(self, t_seconds, command, phi0 = 0.0):
    """CPG waveform at episode time `t_seconds` for a reset-started episode."""
    t = np.asarray(t_seconds, np.float64)
    return self._cpg_waveform(self.cpg_gen.phase_at(t, phi0).numpy(), t / (PHYSICS_DT * N_SUBSTEPS), command)

  def _phase_reset(self, phi, active, contact):
    """Contact-driven phase reset; gain 0 disables the entrainment."""
    gain = self._phase_reset_gain

    if gain <= 0.0:
      return phi

    prev = self._phase_prev_contact[active]
    phi, contact = self.cpg_gen.phase_reset(phi, contact, prev, gain = gain)
    self._phase_prev_contact[active] = contact
    return phi.numpy()

  def _cpg_frequency(self, active, contact):
    """Learned cadence for the active rows, or the fixed clock."""
    if not self.cpg_adaptive:
      return CPG_FREQ

    # low-pass the contact bounce; the AFO state persists across episodes
    drive = contact[:, 0].astype(np.float64) - contact[:, 1].astype(np.float64)
    self._cpg_drive[active] = (
      self._cpg_drive_alpha * drive + (1.0 - self._cpg_drive_alpha) * self._cpg_drive[active]
    )

    freq = self.cpg_afo.step(
      torch.as_tensor(self._cpg_drive[active]), dt = self.cpg_gen.dt, rows = active,
      mask = torch.as_tensor(self._cpg_adapt_mask[active]),
    )
    return freq.cpu().numpy()

  @property
  def foot_contacts(self):
    """(num_envs, NUM_FEET) bool: which feet are touching the ground right now."""
    return self.eng.foot_contacts(self._foot_geoms)

  # ------------------------------------------------------------------ helpers
  def _write_state(self, i: int, qpos: np.ndarray, qvel: np.ndarray):
    self.eng.set_rows([i], qpos[None], qvel[None])

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

    qpos = self.home_qpos.copy()

    # per-worker reset perturbation scale: 0 = exact home pose at rest (the
    # easiest initial state), 1 = the stock reset noise. SGS can ride a bank of
    # these scales (initial-state distribution curriculum).

    s = self._reset_scale[i]
    qpos[: SPATIAL_DIM] = WORLD_UP * (self.home_qpos[SPATIAL_DIM - 1] + s * rng.uniform(*RESET_Z_RANGE))
    spawn_quat = np.array([1.0, 0.0, 0.0, 0.0])
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

    # mid-stride momentum prime: forward trunk velocity on top of the spawned
    # state (world-x is the free joint's first velocity dof)

    qvel[0] += self._reset_vel[i]

    self._write_state(i, qpos, qvel)
    self._last_action[i] = 0.0
    self._command[i] = command
    self._steps[i] = 0
    self._foot_air[i] = 0.0
    self._episodes[i] += 1
    self._phi[i] = self._reset_phase[i]
    self._phase_prev_contact[i] = False
    self._cpg_adapt_mask[i] = False
    self._cpg_drive[i] = 0.0

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
    keep = self.eng.snapshot_rows(idle) if len(idle) else None

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
      contact = self.eng.foot_contacts(self._foot_geoms)[active]
      freq = self._cpg_frequency(active, contact)
      phi = self._phase_reset(phi, active, contact)
      cpg_target = self._cpg_waveform(phi, steps, command, freq = freq)
      ctrl = ctrl + cpg_target
      phi = self.cpg_gen.advance(phi, freq = freq).numpy()

    self._ctrl_np[active] = ctrl.astype(np.float32)
    self.eng.set_ctrl(self._ctrl_np)

    for s in range(N_SUBSTEPS):
      self.eng.step(kin = (s == N_SUBSTEPS - 1))

    self.eng.sync()

    if keep is not None:
      self.eng.restore_rows(idle, *keep)

    qpos = np.asarray(self.eng.qpos)[active].astype(np.float64)
    qvel = np.asarray(self.eng.qvel)[active].astype(np.float64)
    self._xmat = np.asarray(self.eng.xmat)[active].astype(np.float64).reshape(k, self.model.nbody, 3, 3)
    xpos = np.asarray(self.eng.xpos)[active].astype(np.float64)
    cvel = np.asarray(self.eng.cvel)[active].astype(np.float64)
    xipos = np.asarray(self.eng.xipos)[active].astype(np.float64)
    cinert = np.asarray(self.eng.cinert)[active].astype(np.float64)
    subtree_com = np.asarray(self.eng.subtree_com)[active].astype(np.float64)
    geom_xpos = np.asarray(self.eng.geom_xpos)[active].astype(np.float64)

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

    up = np.broadcast_to(WORLD_UP, (k, 3))
    trunk_up = self._xmat_trunk[:, :, SPATIAL_DIM - 1]
    uprightness = (trunk_up * up).sum(-1)
    trunk_h = (xpos[:, TRUNK_BODY] * up).sum(-1)

    # the next step may adapt the cadence only from a healthy, moving gait
    self._cpg_adapt_mask[active] = (
      (uprightness >= 0.8) & (trunk_h >= 0.09) & (np.abs(command[:, 0]) > COMMAND_EPS) & (steps >= 10)
    )

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

    foot_contact = self.eng.foot_contacts(self._foot_geoms)[active]
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

    # 15-body ground contact / proximity (2 feet + 13 non-foot 3D body points)
    geom_z = np.where(bad[:, None], 1.0, geom_xpos[:, 1:, 2])
    body_min_z = np.minimum(
      np.minimum.reduceat(geom_z, self._body_geom_starts, axis = 1),
      np.where(bad[:, None], 1.0, xpos[:, 1:, 2]),
    )
    non_foot_contact = (body_min_z[:, self._non_foot_body_idx0] <= self._body_contact_z_thresh).astype(np.float32)
    body_contacts = np.concatenate([foot_contact.astype(np.float32), non_foot_contact], axis = 1)

    if bad.any():
      obs = np.where(bad[:, None], self._safe_obs(command, self._last_action[active], phi), obs)
      reward = np.where(bad, NAN_PENALTY, reward)
      terminated = terminated | bad

    return obs, reward.astype(np.float32), terminated, np.where(bad, False, truncated), dict(body_contacts = body_contacts)

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
TARGETS = (0.0, 0.05, 0.15, 0.3, 0.5, 0.8)

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

@torch.no_grad()
def evaluate(env: DuckGaitVecEnv, agent, num_episodes = 128, horizon = 400, latent_id = 0, latent = None, reach = 0.35):
    """Greedy eval from the clean stand: distance, lines, foot steps, falls.

    `ev_lines` is standing-gated progress (same credit the env pays), and
    `ev_reach` the fraction of eval episodes that walk `reach` metres upright.
    `latent` is either one fixed vector or one row per episode for fixed latents.
    """

    n = min(num_episodes, env.num_envs)
    rows = np.arange(n, dtype = np.int64)

    if latent is not None:
        latent = torch.as_tensor(latent).float()
        latent = latent.expand(n, -1) if latent.ndim == 1 else latent[:n]
        assert latent.shape[0] == n

    env._active = rows
    env._apply_state(rows, dict(split = 0.0, phase = 0.0, tilt = 0.0, scale = 0.0, target = 0.0))

    alive = np.ones(n, bool)
    x_max, x_end, lines = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    speed_sum, n_steps, surv_steps, fell = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, bool)
    prev_contact = np.zeros((n, NUM_FEET), bool)
    strike_errs = []
    prev_x = np.asarray(env.env.eng.qpos)[rows, 0].astype(np.float64)

    agent.eval()
    t_eval = time.perf_counter()

    for _ in range(horizon):
        idx = np.flatnonzero(alive)

        if len(idx) == 0:
            break

        state = torch.from_numpy(np.ascontiguousarray(env._obs[idx])).float()
        latent_batch = latent[idx] if latent is not None else None
        action = agent.get_actor_actions(state, latent_id = latent_id, latent = latent_batch, sample = False, use_unwrapped_model = True)
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

@torch.no_grad()
def evaluate_latents(env, agent, zs, num_episodes = 128, horizon = 400):
    """Spread the eval budget over independently sampled fixed latents."""

    per = max(1, num_episodes // len(zs))
    per_latent_evals = [evaluate(env, agent, num_episodes = per, horizon = horizon, latent = z) for z in zs]
    per_latent_x = [e['ev_x'] for e in per_latent_evals]
    best_latent = int(np.argmax(per_latent_x))

    eval_metrics = {k: float(np.mean([e[k] for e in per_latent_evals])) for k in per_latent_evals[0]}
    eval_metrics.update(
        ev_best_latent_x = per_latent_x[best_latent],
        ev_best_latent_lines = per_latent_evals[best_latent]['ev_lines'],
        ev_latent_x = [round(x, 4) for x in per_latent_x],
        ev_latent_x_std = float(np.std(per_latent_x)),
        ev_latent_x_range = float(np.ptp(per_latent_x)),
        ev_latent_lines_std = float(np.std([e['ev_lines'] for e in per_latent_evals])),
    )

    return eval_metrics, zs[best_latent].clone()

@torch.no_grad()
def dads_metrics(agent, memories, max_samples = 4096):
    """Held-out DADS skill identifiability: q(z | s, a, s') MSE against the true
    episode z, versus the same prediction against shuffled z. `dads_r2` near 1
    means behavior is strongly latent-specific; near 0 means it is not."""

    if not (agent.use_diversity_discr and agent.has_fixed_latents):
        return {}

    real = [m for m in memories if int(m.episode_id) != -1]

    if not real:
        return {}

    if len(real) > max_samples:
        indices = np.random.choice(len(real), max_samples, replace = False)
        real = [real[i] for i in indices]

    states = torch.stack([m.state for m in real]).to(agent.device)
    next_states = torch.stack([m.next_state for m in real]).to(agent.device)
    actions = torch.stack([m.action for m in real]).to(agent.device)
    z = torch.stack([m.latent_gene_id for m in real]).to(agent.device)

    discr = agent.unwrap_model(agent.diversity_discr)
    discr.eval()

    pred = discr(states, next_states, action = actions)
    mse = F.mse_loss(pred, z)
    shuffled = z[torch.randperm(len(z), device = z.device)]
    mse_shuffled = F.mse_loss(pred, shuffled)

    per_sample = ((pred - z) ** 2).mean(dim = -1)

    return dict(
        dads_mse = float(mse),
        dads_mse_shuffled = float(mse_shuffled),
        dads_mse_std = float(per_sample.std()),
        dads_mse_min = float(per_sample.min()),
        dads_mse_max = float(per_sample.max()),
        dads_r2 = float(1. - mse / mse_shuffled.clamp(min = 1e-8)),
        dads_reward = float(-mse),
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
        cpg_adaptive: bool = False,
        cpg_adapt_rate: float = 10.0,
        cpg_freq_range = (0.5, 4.0),
        forced_swing: float | None = 0.25,
        phase_reset: float = 0.0,
        shaping: float = 1.0,
        bank = None,
        sgs: dict | None = None,
        backend: str = 'mlx',
    ):
        self.num_envs = int(num_envs)

        self._ctor_kwargs = dict(
            num_envs = num_envs, horizon = horizon, mjcf_path = mjcf_path, seed = seed, command = command,
            learned_clock = learned_clock, clock_band = clock_band, line_step = line_step, line_bonus = line_bonus,
            target_bonus = target_bonus, fall_penalty = fall_penalty, min_steps_success = min_steps_success,
            min_sustain_steps = min_sustain_steps, stance_soft = stance_soft, forward_pull = forward_pull,
            time_features = time_features, cpg = cpg, cpg_adaptive = cpg_adaptive, cpg_adapt_rate = cpg_adapt_rate,
            cpg_freq_range = cpg_freq_range,
            forced_swing = forced_swing, phase_reset = phase_reset,
            shaping = shaping, bank = bank, sgs = sgs, backend = backend,
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
        self.cpg_adaptive = bool(cpg_adaptive)

        self.env = MicroduckMlxBatchedEnv(
            Path(mjcf_path) if mjcf_path else MJCF,
            num_envs = self.num_envs,
            horizon = self.horizon,
            seed = seed,
            cpg = cpg,
            cpg_adaptive = cpg_adaptive,
            cpg_adapt_rate = cpg_adapt_rate,
            cpg_freq_range = cpg_freq_range,
            forced_swing = forced_swing,
            fixed_command = (command, 0.0),
            backend = backend,
        )
        self.backend = backend
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
        self._prev_contact = np.zeros((self.num_envs, NUM_FEET), bool)
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
            reset_configs = bank if bank is not None else make_stand_bank(),
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
        obs, env_rew, terminated, truncated, env_info = self.env.step_batch(joint_targets, worker_ids = rows)
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

        body_contacts_out = np.zeros((self.num_envs, self.env.num_contact_bodies), np.float32)
        if 'body_contacts' in env_info:
            body_contacts_out[rows] = env_info['body_contacts']

        info = dict(
            episode_done = done,
            episode_success = success,
            episode_progress = prog_out,
            episode_target = self._target.copy(),
            episode_steps = steps_out,
            body_contacts = body_contacts_out,
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

# anticipatory contact aux - predict future contact bins from actor / critic hidden

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def future_contact_bins(contacts, num_bins = 4, bin_width = 5, mode = 'any'):
    # (t c) -> targets (t b c), mask (t b); bin b is the b-th future window
    assert mode in ('any', 'mean')

    contacts = torch.as_tensor(contacts, dtype = torch.float32)
    steps = contacts.shape[0]
    horizon = num_bins * bin_width

    windows = F.pad(contacts[1:], (0, 0, 0, horizon)).unfold(0, horizon, 1)
    windows = rearrange(windows, 't c (b w) -> t b w c', b = num_bins)
    targets = reduce(windows, 't b w c -> t b c', 'max' if mode == 'any' else 'mean')

    t = torch.arange(steps)[:, None]
    b = torch.arange(num_bins)[None, :]
    mask = (t + (b + 1) * bin_width) < steps

    return targets, mask

class AnticipatoryContactHead(nn.Module):
    def __init__(
        self,
        dim,
        num_actions,
        num_contacts,
        num_bins,
        num_feet = 2,
        dim_action = 32,
        body_loss_share = 0.5,
        dim_critic = None
    ):
        super().__init__()
        self.num_bins = num_bins
        self.num_contacts = num_contacts
        self.num_feet = min(num_feet, num_contacts)
        self.body_loss_share = body_loss_share

        dim_out = num_bins * num_contacts
        dim_critic = default(dim_critic, dim)

        self.to_bins = Rearrange('... (b c) -> ... b c', b = num_bins, c = num_contacts)

        self.action_proj = nn.Sequential(
            nn.Linear(num_actions, dim_action),
            nn.SiLU(),
            nn.Linear(dim_action, dim_action)
        )

        self.film = nn.Linear(dim_action, dim * 2, bias = False)
        nn.init.zeros_(self.film.weight)

        self.state_head = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, dim_out)
        )

        self.net = nn.Sequential(
            nn.RMSNorm(dim + dim_action),
            nn.Linear(dim + dim_action, dim),
            nn.SiLU(),
            nn.Linear(dim, dim_out)
        )

        self.critic_net = nn.Sequential(
            nn.RMSNorm(dim_critic),
            nn.Linear(dim_critic, dim_critic),
            nn.SiLU(),
            nn.Linear(dim_critic, dim_out)
        )

    def predict(self, hidden, action):
        action_emb = self.action_proj(action)
        gamma, beta = self.film(action_emb).chunk(2, dim = -1)
        hidden_cond = hidden * (gamma + 1.) + beta

        logits = self.state_head(hidden) + self.net(torch.cat((hidden_cond, action_emb), dim = -1))
        return self.to_bins(logits)

    def predict_critic(self, hidden):
        return self.to_bins(self.critic_net(hidden))

    def bce(self, logits, targets, mask):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')
        mask_f = mask[..., None].float()
        denom = mask_f.sum().clamp(min = 1.)

        per_channel = (bce * mask_f).sum(dim = (0, 1)) / denom
        foot = per_channel[:self.num_feet].mean()
        body = per_channel[self.num_feet:].mean()

        loss = torch.lerp(foot, body, self.body_loss_share) if self.num_contacts > self.num_feet else foot
        return loss, foot, body

    @torch.no_grad()
    def stats(self, logits, targets, mask, loss, foot, body):
        correct = ((logits >= 0.) == (targets >= 0.5)).float()
        mask_f = mask[..., None].float()
        denom = mask_f.sum().clamp(min = 1.)

        per_channel = (correct * mask_f).sum(dim = (0, 1)) / denom
        per_bin = (correct * mask_f).sum(dim = (0, 2)) / (mask_f.sum(dim = (0, 2)) * self.num_contacts).clamp(min = 1.)

        return dict(
            loss = float(loss),
            foot_acc = float(per_channel[:self.num_feet].mean()),
            body_acc = float(per_channel[self.num_feet:].mean()) if self.num_contacts > self.num_feet else float('nan'),
            bin_acc = per_bin.tolist()
        )

    def loss(self, hidden, action, targets, mask, critic = False):
        logits = self.predict_critic(hidden) if critic else self.predict(hidden, action)
        loss, foot, body = self.bce(logits, targets, mask)
        return loss, self.stats(logits, targets, mask, loss, foot, body)

class AnticipatoryContactAux(nn.Module):
    def __init__(
        self,
        actor,
        critic,
        env,
        actor_optim,
        critic_optim,
        num_bins = 4,
        bin_width = 5,
        num_contacts = 15,
        num_feet = 2,
        body_loss_share = 0.5,
        bin_mode = 'any',
        weight = 0.5,
        max_grad_norm = 0.5,
        max_episode_length = 400,
        lr = 3e-4
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.weight = weight
        self.max_grad_norm = max_grad_norm
        self.max_episode_length = max_episode_length
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.bin_mode = bin_mode
        self.num_contacts = num_contacts

        self.head = AnticipatoryContactHead(
            dim = actor.dim,
            num_actions = actor.num_actions,
            num_contacts = num_contacts,
            num_bins = num_bins,
            num_feet = num_feet,
            body_loss_share = body_loss_share,
            dim_critic = critic.init_layer[0].out_features
        )
        self.optim = torch.optim.AdamW(self.head.parameters(), lr = lr, weight_decay = 0.)
        self.opts = [opt for opt in (actor_optim, critic_optim, self.optim) if exists(opt)]

        self._last_state = None
        self._last_latent = None
        self._steps = [[] for _ in range(env.num_envs)]
        self._buffer = []

        actor.register_forward_pre_hook(self._capture_state, with_kwargs = True)

        self.attach_env(env)

    def _capture_state(self, _, args, __):
        if torch.is_grad_enabled() or not torch.is_tensor(args[0]):
            return

        self._last_state = args[0].detach().cpu()
        latent = args[1] if len(args) > 1 else None
        self._last_latent = latent.detach().cpu() if torch.is_tensor(latent) else None

    def attach_env(self, env):
        reset_one, step_batch = env.reset_one, env.step_batch

        def reset(worker, seed = None):
            self._steps[int(worker)] = []
            return reset_one(worker, seed = seed)

        def step(actions, worker_ids = None):
            out = step_batch(actions, worker_ids = worker_ids)
            info = out[4] if len(out) > 4 and isinstance(out[4], dict) else {}

            if 'body_contacts' not in info or not exists(self._last_state):
                return out

            active = np.arange(env.num_envs) if not exists(worker_ids) else np.asarray(list(worker_ids), np.int64)
            contacts = np.asarray(info['body_contacts'], np.float32)[..., :self.num_contacts]
            terminated, truncated = np.asarray(out[2], bool), np.asarray(out[3], bool)
            actions = torch.as_tensor(actions, dtype = torch.float32).cpu()

            for k, w in enumerate(active):
                w = int(w)
                latent = self._last_latent[k] if exists(self._last_latent) else None
                self._steps[w].append((self._last_state[k], latent, actions[k], contacts[k]))

                if not (terminated[k] or truncated[k] or len(self._steps[w]) >= self.max_episode_length):
                    continue

                traj, self._steps[w] = self._steps[w], []

                targets, mask = future_contact_bins(
                    np.stack([s[3] for s in traj]),
                    num_bins = self.num_bins,
                    bin_width = self.bin_width,
                    mode = self.bin_mode
                )

                if not mask.any():
                    continue

                has_latent = exists(traj[0][1])
                self._buffer.append((
                    torch.stack([s[0] for s in traj]),
                    torch.stack([s[1] for s in traj]) if has_latent else None,
                    torch.stack([s[2] for s in traj]),
                    targets,
                    mask
                ))

            return out

        env.reset_one, env.step_batch = reset, step
        return env

    def step(self, epochs = 4, batch_size = 1024):
        if not self._buffer:
            return {}

        device = next(self.head.parameters()).device
        states, latents, actions, targets, masks = zip(*self._buffer)
        self._buffer.clear()

        states = torch.cat(states).to(device)
        actions = torch.cat(actions).to(device)
        targets = torch.cat(targets).to(device)
        masks = torch.cat(masks).to(device)
        has_latents = exists(latents[0])
        latents = torch.cat(latents).to(device) if has_latents else None

        self.head.train()
        self.actor.train()
        self.critic.train()

        stats = []

        for _ in range(epochs):
            for idx in torch.randperm(len(states), device = device).split(batch_size):
                lat = latents[idx] if has_latents else None

                actor_hidden = self.actor.latent(states[idx], lat)
                loss, actor_stats = self.head.loss(actor_hidden, actions[idx], targets[idx], masks[idx])
                stats.append(actor_stats)

                critic_hidden = self.critic.latent(states[idx], lat)
                critic_loss, _ = self.head.loss(critic_hidden, None, targets[idx], masks[idx], critic = True)
                loss = loss + critic_loss

                for opt in self.opts:
                    opt.zero_grad()

                (self.weight * loss).backward()

                if exists(self.max_grad_norm):
                    for module in (self.actor, self.critic, self.head):
                        nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)

                for opt in self.opts:
                    opt.step()
                    opt.zero_grad()

        def mean(vals):
            vals = np.asarray(vals, dtype = np.float64)
            return float(np.nanmean(vals)) if not np.isnan(vals).all() else float('nan')

        return dict(
            contact_aux_loss = mean([s['loss'] for s in stats]),
            contact_foot_acc = mean([s['foot_acc'] for s in stats]),
            contact_body_acc = mean([s['body_acc'] for s in stats]),
            contact_bin_acc = [mean([s['bin_acc'][b] for s in stats]) for b in range(self.num_bins)]
        )

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

def infer_dim_latent(pkg):
    actor = pkg['actor']

    for key in ('mlp.latent_proj.weight', 'mlp.encode_latent.1.weight'):
        if key in actor:
            return int(actor[key].shape[1])

    return 32

def load_agent(checkpoint: str, num_latents: int = 0):

    pkg = torch.load(str(checkpoint), map_location = 'cpu', weights_only = True)
    obs_dim = int(pkg['actor']['init_layer.0.weight'].shape[1])
    num_actions = int(pkg['actor']['to_out.1.weight'].shape[0]) // 2

    latent_mode = pkg.get('latent_mode', 'gene_pool')
    dim_latent = int(pkg['dim_latent']) if 'dim_latent' in pkg else infer_dim_latent(pkg)

    if latent_mode == 'fixed':
        num_latents = 1
    elif num_latents <= 0 and pkg.get('latents') is not None:
        num_latents = int(pkg['latents']['latents'].shape[0])

    num_latents = num_latents if num_latents > 0 else 32

    use_diversity_discr = pkg.get('diversity_discr') is not None

    agent = build_epo_agent(
        obs_dim,
        num_actions,
        dict(
            num_latents = num_latents,
            dim_latent = dim_latent,
            latent_mode = latent_mode,
            use_critic_ema = False,
            use_hidden_spr = False,
            use_diversity_discr = use_diversity_discr,
            quiet = True
        ),
    )

    agent.unwrap_model(agent.actor).load_state_dict(pkg['actor'])
    agent.unwrap_model(agent.critic).load_state_dict(pkg['critic'])

    if pkg.get('latents') is not None and agent.has_latent_genes:
        agent.unwrap_model(agent.latent_gene_pool).load_state_dict(pkg['latents'])

    if use_diversity_discr:
        agent.unwrap_model(agent.diversity_discr).load_state_dict(pkg['diversity_discr'])

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
    latent = None,
    latent_seed: int = 0,
    num_latents: int = 0,
    sample: bool = False,
    learned_clock: bool = False,
    cpg: bool = True,
    cpg_adaptive: bool = False,
    forced_swing: float = 0.25,
    time_features: str = 'fourier',
    backend: str = 'mlx',
):
    split_s, _, phase_s = str(start).partition('@')
    split, phase = float(split_s), float(phase_s or 0.0)
    band = _floats(clock_band)
    band = (band[0], band[1]) if len(band) >= 2 else (0.3, 3.0)

    env = DuckGaitVecEnv(num_envs = 1, horizon = horizon, seed = seed, learned_clock = learned_clock, clock_band = band,
                         cpg = cpg, cpg_adaptive = cpg_adaptive,
                         forced_swing = forced_swing, time_features = time_features, backend = backend)
    env._apply_state([0], dict(split = split, phase = phase, tilt = tilt, scale = scale, target = 0.0))

    agent, obs_dim = load_agent(checkpoint, num_latents = num_latents)

    if agent.has_fixed_latents and latent is None:
        latent = torch.randn(agent.dim_latent, generator = torch.Generator().manual_seed(latent_seed))

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
            action = agent.get_actor_actions(state, latent_id = latent_id, latent = latent, sample = sample, use_unwrapped_model = True)

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
        latent_id = latent_id, latent_seed = latent_seed, frames = len(frames),
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
    total_env_steps: int = 2_000_000, num_envs: int = 1024, num_latents: int = 1024, episodes_per_latent: int = 1,
    latent_mode: str = 'fixed', dim_latent: int = 8,
    use_diversity_discr: bool | None = None, diversity_reward_weight: float = 1.0,
    diversity_time_gap: int = 1, diversity_discr_dim: int = 64, diversity_discr_depth: int = 2,
    diversity_discr_lr: float = 3e-4, eval_latents: int = 16, render_latents: int = 6,
    # task
    horizon: int = 400, command_speed: float = 0.15, line_step: float = 0.05,
    line_bonus: float = 0.1, target_bonus: float = 1.0, fall_penalty: float = 0.0,
    min_steps_success: int = 0, min_sustain_steps: int = 50,
    stance_soft: bool = False, forward_pull: float = 0.0, shaping: float = 1.0,
    # exploration annealing + the SGS bank
    temp_start: float = 1.0, temp_end: float = 1.0, temp_cycles: int = 0, temp_gate_steps: int = 250,
    targets: str = '0.0,0.05,0.15,0.3,0.5,0.8',
    stand_scales: str = '0.0,0.5,1.0', stand_commands: str = '',
    time_features: str = 'fourier', cpg: bool = True, cpg_adaptive: bool = False,
    cpg_adapt_rate: float = 10.0, cpg_freq_min: float = 0.5, cpg_freq_max: float = 4.0,
    forced_swing: float = 0.25, phase_reset: float = 0.0,
    learned_clock: bool = False, clock_band = (0.3, 3.0),
    target: float = 0.66, kappa: float = 5.0,
    sgs_eps: float = 1e-6, sgs_window: int = 100, temperature: float = 2.0, sgs_uniform_prob: float = 0.25,
    # optimization
    eval_every: int = 3, eval_episodes: int = 128, render: bool = True, backend: str = 'mlx', nim_cpu: bool = False,
    epochs: int = 4, batch_size: int = 1024, learning_rate: float = 5e-4, entropy_weight: float = 0.002,
    actor_mlp_depth: int = 2, actor_dim: int = 128, critic_mlp_depth: int = 2, critic_dim: int = 128,
    use_hidden_spr: bool = True, hidden_spr_weight: float = 1.0,
    use_contact_aux: bool = False, contact_aux_weight: float = 0.5,
    contact_num_bins: int = 4, contact_bin_width: int = 5, contact_bodies: str = 'full',
    contact_bin_mode: str = 'any', contact_body_loss_share: float = 0.5,
    use_hpo: bool = True, per_action_dim: bool = True, max_grad_norm: float = 0.5, resume_from: str = '',
    # online DAgger: a no-CPG student distills from the CPG teacher
    distill: bool = False, student_envs: int = 256, student_rollout_steps: int = 200, student_rollout_every: int = 1,
    distill_epochs: int = 2, distill_batch_size: int = 1024, distill_samples: int = 65536,
    distill_lr: float = 5e-4, distill_buffer: int = 200000,
    student_save_path: str = '', student_resume_from: str = '',
    rep_align: bool = False, rep_align_weight: float = 1.0, seed: int = 42,
):
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))
    torch.manual_seed(seed)
    np.random.seed(seed)

    assert latent_mode in ('gene_pool', 'fixed'), f'latent_mode must be gene_pool / fixed, got {latent_mode}'
    assert not (distill and latent_mode == 'fixed'), 'online DAgger currently needs the gene-pool latent mode'

    if use_diversity_discr is None:
        use_diversity_discr = latent_mode == 'fixed'

    # --nim_cpu forces the CPU backend with the compiled Nim paths, failing
    # loudly below if they are unavailable

    if nim_cpu:
        backend = 'cpu'
        os.environ.pop('MUJOCO_WARP_DISABLE_NIM', None)
        print('[SGS-gait] --nim_cpu: using mujoco_warp CPU with Nim fast paths')

    assert backend in ('mlx', 'cpu'), f"backend must be mlx / cpu, got {backend}"

    if distill and (not cpg or learned_clock):
        # the teacher labels student states by splicing in its own CPG phase:
        # teacher obs must be exactly student obs + 2 cpg dims

        print('[SGS-gait] distill: forcing cpg True / learned_clock False (teacher obs = student obs + CPG phase)')
        cpg, learned_clock = True, False

    if distill and cpg_adaptive:
        # the teacher waveform is reconstructed from episode time, so keep the fixed clock
        print('[SGS-gait] distill: cpg_adaptive off (teacher waveform is reconstructed from time)')
        cpg_adaptive = False

    save_path, log_path = Path(save_path).resolve(), Path(log_path).resolve()
    save_path.parent.mkdir(parents = True, exist_ok = True); log_path.parent.mkdir(parents = True, exist_ok = True)
    log_path.write_text('')

    bank = make_stand_bank(scales = _floats(stand_scales), targets = _floats(targets),
                           commands = _pairs(stand_commands) or None)

    sgs_kwargs = dict(target = target, kappa = kappa, eps = sgs_eps, window = sgs_window, temperature = temperature,
                      uniform_prob = sgs_uniform_prob, log_every = 128, log_path = str(log_path.with_name('sgs_progression.csv')), seed = seed)

    env = DuckGaitVecEnv(
        num_envs = num_envs, horizon = horizon, seed = seed, command = command_speed, learned_clock = learned_clock,
        clock_band = clock_band, line_step = line_step, line_bonus = line_bonus, target_bonus = target_bonus,
        fall_penalty = fall_penalty, min_steps_success = min_steps_success, min_sustain_steps = min_sustain_steps,
        stance_soft = stance_soft, forward_pull = forward_pull, time_features = time_features, cpg = cpg,
        cpg_adaptive = cpg_adaptive, cpg_adapt_rate = cpg_adapt_rate,
        cpg_freq_range = (cpg_freq_min, cpg_freq_max),
        forced_swing = forced_swing, phase_reset = phase_reset, shaping = shaping, bank = bank, sgs = sgs_kwargs,
        backend = backend,
    )

    if backend == 'cpu':
        nim = 'on' if getattr(env.env.eng, 'nim_available', False) else 'OFF'
        print(f'[SGS-gait] backend: mujoco_warp CPU, Nim fast paths {nim}')

        if nim_cpu and nim != 'on':
            raise RuntimeError('--nim_cpu requested but the Nim fast paths are unavailable')
    else:
        import mujoco_warp
        print(f'[SGS-gait] backend: mujoco_warp MLX ({mujoco_warp.__file__})')

    print(f'[SGS-gait] {num_envs} ducks, {len(bank)} cells = initial states x targets {targets}, SGS target {target} '
          f'kappa {kappa} eps {sgs_eps}; success = cross the cell target on foot (>= {min_steps_success} steps, '
          f'sustained >= {min_sustain_steps}) before the episode ends')

    contact_num_channels = 0

    if use_contact_aux:
        assert contact_bodies in ('full', 'feet'), f"contact_bodies must be 'full' or 'feet', got {contact_bodies!r}"
        assert contact_bin_mode in ('any', 'mean'), f"contact_bin_mode must be 'any' or 'mean', got {contact_bin_mode!r}"
        assert contact_num_bins * contact_bin_width < horizon, 'contact lookahead must be shorter than the episode'
        contact_num_channels = env.env.num_contact_bodies if contact_bodies == 'full' else NUM_FEET

    agent = env.to_epo_agent(
        num_latents = num_latents, dim_latent = dim_latent, actor_dim = actor_dim, actor_mlp_depth = actor_mlp_depth,
        critic_dim = critic_dim, critic_mlp_depth = critic_mlp_depth, use_critic_ema = False, clip_value = False,
        use_hidden_spr = use_hidden_spr, hidden_spr_weight = hidden_spr_weight,
        latent_mode = latent_mode, use_diversity_discr = use_diversity_discr, diversity_time_gap = diversity_time_gap,
        diversity_discr_kwargs = dict(dim = diversity_discr_dim, depth = diversity_discr_depth),
        diversity_discr_lr = diversity_discr_lr,
        actor_lr = learning_rate, critic_lr = learning_rate, batch_size = batch_size, max_grad_norm = max_grad_norm,
        actor_loss_kwargs = dict(
            entropy_weight = entropy_weight, eps_clip = 0.2, norm_advantages = True, use_hpo = use_hpo, per_action_dim = per_action_dim,
        ),
        quiet = True,
    )

    contact_aux = None

    if use_contact_aux:
        contact_aux = AnticipatoryContactAux(
            actor = agent.unwrap_model(agent.actor),
            critic = agent.unwrap_model(agent.critic),
            env = env,
            actor_optim = agent.actor_optim,
            critic_optim = agent.critic_optim,
            num_bins = contact_num_bins,
            bin_width = contact_bin_width,
            num_contacts = contact_num_channels,
            num_feet = NUM_FEET,
            body_loss_share = contact_body_loss_share,
            bin_mode = contact_bin_mode,
            weight = contact_aux_weight,
            max_grad_norm = max_grad_norm,
            max_episode_length = horizon,
        ).to(agent.device)

        horizon_seconds = contact_num_bins * contact_bin_width * PHYSICS_DT * N_SUBSTEPS
        print(f'[SGS-gait] anticipatory contact aux: {contact_num_channels} bodies ({contact_bodies}), '
              f'{contact_num_bins} bins x {contact_bin_width} steps ({horizon_seconds:.2f}s), weight {contact_aux_weight}')

    loss_name = 'HPO (x-ppo 0.0.27)' if use_hpo else 'PPO clip'
    print(f'[SGS-gait] actor loss: {loss_name}, per_action_dim={per_action_dim}')

    diversity_note = f'DADS weight {diversity_reward_weight}' if use_diversity_discr else 'no DADS'
    fixed_note = f' ({dim_latent}d, {num_latents} sampled per cycle, {diversity_note})' if latent_mode == 'fixed' else ''
    print(f'[SGS-gait] latent mode: {latent_mode}{fixed_note}')

    print(f'[SGS-gait] teacher {init_gait_agent(agent, resume_from)}')

    epo = EPO(
        agent,
        episodes_per_latent = episodes_per_latent,
        max_episode_length = horizon,
        action_sample_temperature = temp_start,
        diversity_reward_weight = diversity_reward_weight,
    )

    # online DAgger: a no-CPG student learns from the teacher's labels on its own states

    student = student_env = student_optim = distiller = None

    if distill:
        student_env = env.spawn(num_envs = student_envs, seed = seed + 7919, cpg = False, learned_clock = False, phase_reset = 0.0,
                                cpg_adaptive = False,
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

        print(f'[SGS-gait] online DAgger (x-ppo 0.0.27): no-CPG student ({student_env.obs_dim} obs / {student_env.action_dim} act)')

    pbar = tqdm(total = total_env_steps, desc = 'SGS gait (single duck)', unit = 'step', unit_scale = True, dynamic_ncols = True)
    total_steps = cycle = 0
    ep_steps_recent, best_forward_x, t_start = None, -float('inf'), time.time()

    last_eval_metrics = dict(ev_x = float('nan'), ev_lines = float('nan'), ev_survival = float('nan'), ev_latent_x_range = float('nan'))
    eval_zs = None
    last_latent_eval = None
    if latent_mode == 'fixed':
        eval_latents = max(1, min(eval_latents, eval_episodes))
        eval_zs = torch.randn(eval_latents, dim_latent, generator = torch.Generator().manual_seed(seed + 12345))
        best_eval_z = eval_zs[0].clone()

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
            rollout = epo.gather_experience_from(env)
            t_gather = time.perf_counter() - t0

            steps_this_cycle = len(rollout.memories)
            total_steps += steps_this_cycle

            # held-out DADS check before the discriminator is updated on this batch

            dads_stats = dads_metrics(agent, rollout.memories)

            t0 = time.perf_counter()
            _, fitness = agent.learn_from(rollout, epochs = epochs)
            contact_stats = contact_aux.step(epochs = epochs, batch_size = batch_size) if exists(contact_aux) else {}
            t_learn = time.perf_counter() - t0

            mean_reward = float(rollout.cumulative_rewards.mean().item())
            ep_steps_recent = float(env.ep_stats['ep_steps'])

            if fitness is not None and agent.has_latent_genes:
                best_latent = int(fitness.argmax().item())
            else:
                best_latent = -1 if agent.has_fixed_latents else 0

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
                dads_discr_loss = float(agent.last_diversity_discr_loss),
                **contact_stats,
                **dads_stats,
            )

            if env.env.cpg_afo is not None:
                record['cpg_freq'] = float(env.env.cpg_afo.freq.mean())

            if cycle % eval_every == 0 or total_steps >= total_env_steps:
                if agent.has_fixed_latents:
                    eval_metrics, best_eval_z = evaluate_latents(env, agent, eval_zs, eval_episodes, horizon)
                    record['best_latent'] = -1
                    last_latent_eval = (eval_zs.clone(), eval_metrics['ev_latent_x'])
                else:
                    eval_metrics = evaluate(env, agent, num_episodes = eval_episodes, horizon = horizon, latent_id = best_latent)

                record.update(eval_metrics)
                last_eval_metrics = eval_metrics
                st_note = ''

                if distill:
                    # the student is evaluated with the teacher, same protocol
                    student_eval_metrics = evaluate(student_env, student, num_episodes = eval_episodes, horizon = horizon, latent_id = 0)
                    record.update({f'st_{k[3:]}': v for k, v in student_eval_metrics.items()})
                    student.save(Path(student_save_path) if student_save_path else save_path.with_name('student.pt'), overwrite = True)

                    align = s_breakdown.alignment if s_breakdown is not None else None
                    st_note = (f' | student: x={record["st_x"]:+.3f} lines={record["st_lines"]:.1f} reach={record["st_reach"]:.2f} '
                               f'falls={record["st_falls"]:.2f} loss={s_loss:.2f}' + (f' align={float(align):.3f}' if align is not None else ''))

                with log_path.open('a') as f:
                    f.write(json.dumps(record) + '\n')

                if eval_metrics['ev_x'] > best_forward_x:
                    best_forward_x = eval_metrics['ev_x']
                    agent.save(save_path, overwrite = True)

                agent.save(save_path.with_name('ckpt_last.pt'), overwrite = True)
                env.sgs.reset()  # the eval scrambled the slots - reseed fresh SGS episodes

                cpg_note = f' | cpg {record["cpg_freq"]:.2f}Hz' if 'cpg_freq' in record else ''

                tqdm.write(
                    f'[c{cycle:04d} | {total_steps/1e6:.2f}M] ret={mean_reward:+.2f} ep_steps={record["ep_steps"]:.0f} succ={record["ep_success"]:.2f} '
                    f'| eval: x={eval_metrics["ev_x"]:+.3f} lines={eval_metrics["ev_lines"]:.1f} surv={eval_metrics["ev_survival"]:.2f} falls={eval_metrics["ev_falls"]:.2f} '
                    f'| sgs: {record["sgs_visited"]}/{len(probs)} cells, best rate {record["sgs_best_rate"]:.2f}, '
                    f'frontier #{frontier} (sc {spec["scale"]:.2f} tgt {spec["target"]:.2f}) p={probs[frontier]:.3f}{st_note}{cpg_note}'
                )
            else:
                with log_path.open('a') as f:
                    f.write(json.dumps(record) + '\n')

            pbar.update(min(steps_this_cycle, max(0, total_env_steps - pbar.n)))
            pbar.set_postfix(
                cycle = cycle,
                reward = f'{mean_reward:+.1f}',
                forward_x = f'{last_eval_metrics.get("ev_x", float("nan")):+.3f}',
                x_spread = f'{last_eval_metrics.get("ev_latent_x_range", float("nan")):.3f}',
                eval_lines = f'{last_eval_metrics.get("ev_lines", float("nan")):.1f}',
                eval_survival = f'{last_eval_metrics.get("ev_survival", float("nan")):.2f}',
                dads_r2 = f'{dads_stats.get("dads_r2", float("nan")):.2f}',
                sgs_cells = f'{record["sgs_visited"]}/{len(probs)}',
            )

    finally:
        agent.save(save_path.with_name('ckpt_last.pt'), overwrite = True)
        pbar.close()
        print(f'\n[SGS-gait] finished at {total_steps:,} steps in {time.time() - t_start:.0f}s')

    # on a mac (mps), show the gait - one full episode each, from the clean stand

    if render:
        videos = save_path.with_name('videos')
        videos.mkdir(exist_ok = True)
        student_ckpt = Path(student_save_path) if student_save_path else save_path.with_name('student.pt')

        for name, ckpt, cpg_mode in (('teacher', save_path, cpg), ('student', student_ckpt, False)):
            if name == 'student' and not distill:
                continue

            common = dict(
                checkpoint = str(ckpt), cpg = cpg_mode, horizon = horizon,
                cpg_adaptive = cpg_adaptive if name == 'teacher' else False,
                learned_clock = learned_clock if name == 'teacher' else False,
                time_features = time_features, backend = backend,
            )

            if name == 'teacher' and agent.has_fixed_latents and last_latent_eval is not None and render_latents > 0:
                zs, scores = last_latent_eval
                order = np.argsort(scores)[::-1][:render_latents]

                for rank, zi in enumerate(order):
                    out = ('teacher.mp4' if rank == 0 else f'teacher_z{rank}.mp4')
                    render_gait(out_path = str(videos / out), latent = zs[zi], latent_seed = int(zi), **common)
                    print(f'[SGS-gait] gait video: {videos / out} (latent {int(zi)}, eval x {scores[zi]:+.3f})')
            else:
                latent_kwargs = dict(latent = best_eval_z) if (name == 'teacher' and agent.has_fixed_latents) else dict(latent_id = best_latent if name == 'teacher' else 0)

                render_gait(out_path = str(videos / f'{name}.mp4'), **latent_kwargs, **common)
                print(f'[SGS-gait] gait video: {videos / f"{name}.mp4"}')

if __name__ == '__main__':
    fire.Fire(main)
