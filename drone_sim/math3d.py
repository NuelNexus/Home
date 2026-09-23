"""Vectorised 3-D rotation maths.

Conventions used throughout the project
---------------------------------------
* World frame: ENU-style, ``z`` points **up**, gravity is ``(0, 0, -g)``.
* Body frame: FLU -- ``x`` forward, ``y`` left, ``z`` up (thrust direction).
* Quaternions are Hamilton, scalar-first ``[w, x, y, z]`` and rotate vectors
  from the body frame into the world frame: ``v_w = R(q) @ v_b``.
* Every function accepts a leading batch dimension ``(N, ...)``.
"""
from __future__ import annotations

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` for arrays of shape (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) mapping body vectors into the world frame."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - wz)
    R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy)
    R[..., 2, 1] = 2 * (yz + wx)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R


def quat_from_euler(roll, pitch, yaw) -> np.ndarray:
    """Z-Y-X (yaw, pitch, roll) Tait-Bryan angles -> quaternion."""
    roll, pitch, yaw = np.asarray(roll, float), np.asarray(pitch, float), np.asarray(yaw, float)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion -> (roll, pitch, yaw) in radians, shape (..., 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def quat_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation vector (..., 3) -> quaternion (exponential map)."""
    theta = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = 0.5 * theta
    # sin(x/2)/x with a Taylor expansion near zero for numerical safety.
    k = np.where(theta > 1e-8, np.sin(half) / np.maximum(theta, 1e-12), 0.5 - theta**2 / 48.0)
    return np.concatenate((np.cos(half), axis_angle * k), axis=-1)


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix [v]x, shape (..., 3, 3)."""
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def matvec(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Batched ``M @ v`` for M (..., 3, 3) and v (..., 3)."""
    return np.einsum("...ij,...j->...i", M, v)


def matTvec(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Batched ``M.T @ v``."""
    return np.einsum("...ji,...j->...i", M, v)


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched cross product for (..., 3) arrays (much faster than np.cross for small N)."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)
