"""Motion estimation (Methods, Motion estimation): Lucas-Kanade speed and global direction.

The window loop and the summation order are kept as in the code that produced the paper's
results, so numbers agree with them to floating-point rounding. Do not vectorise without
checking the results against this implementation.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import params as P


def prepare_movie(movie: np.ndarray, sigma: float) -> np.ndarray:
    """float32 movie in [0, 1] (uint8 movies are divided by 255), blurred frame by frame
    (Gaussian, standard deviation ``sigma`` px; 0 = no blur). Blur is applied before the scotomas."""
    m = np.asarray(movie).astype(np.float32)
    if m.max() > 1.0:
        m = np.clip(m, 0.0, 255.0) / 255.0
    if sigma > 0:
        m = np.stack([cv2.GaussianBlur(f, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma)) for f in m])
    return m


def derivatives(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central differences I_x, I_y, I_t on the interior grid, each (T-2, H-2, W-2)."""
    F = frames.astype(np.float64)
    Ix = 0.5 * (F[:, :, 2:] - F[:, :, :-2])[1:-1, 1:-1, :]
    Iy = 0.5 * (F[:, 2:, :] - F[:, :-2, :])[1:-1, :, 1:-1]
    It = 0.5 * (F[2:, :, :] - F[:-2, :, :])[:, 1:-1, 1:-1]
    return Ix, Iy, It


def lucas_kanade(Ix, Iy, It, radius: int, stride: int) -> tuple[float, float, int]:
    """Mean local flow (u_x, u_y) over the valid windows, and the number of valid windows.

    Windows of (2 radius + 1)^2 px, centres every ``stride`` px, pooled over all frames. Per window
    u = M^-1 b with M = [[S_xx, S_xy], [S_xy, S_yy]] and b = (-S_xt, -S_yt). A window is discarded
    if lambda_min(M) < 0 or |det M| < 1e-12 (and, first, if it has no gradient at all).
    """
    T, H, W = Ix.shape
    r, s = int(radius), max(1, int(stride))
    ux_sum = uy_sum = 0.0
    n = 0
    for y in range(r, H - r, s):
        for x in range(r, W - r, s):
            ix = Ix[:, y - r:y + r + 1, x - r:x + r + 1].reshape(-1)
            iy = Iy[:, y - r:y + r + 1, x - r:x + r + 1].reshape(-1)
            it = It[:, y - r:y + r + 1, x - r:x + r + 1].reshape(-1)
            sxx = float(np.sum(ix * ix))
            syy = float(np.sum(iy * iy))
            sxy = float(np.sum(ix * iy))
            sxt = float(np.sum(ix * (-it)))
            syt = float(np.sum(iy * (-it)))
            if sxx + syy <= 1e-12:
                continue
            M = np.array([[sxx, sxy], [sxy, syy]], dtype=np.float64)
            if float(np.linalg.eigvalsh(M)[0]) < 0.0:
                continue
            if abs(float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])) < 1e-12:
                continue
            u = np.linalg.solve(M, np.array([sxt, syt], dtype=np.float64))
            ux_sum += float(u[0])
            uy_sum += float(u[1])
            n += 1
    if n == 0:
        return 0.0, 0.0, 0
    return ux_sum / n, uy_sum / n, n


def direction_map(Ix, Iy, It, sigma: float = P.POSTERIOR_SIGMA, step_deg: float = P.ANGLE_STEP_DEG) -> float:
    """MAP direction (deg) of the global posterior over direction, speed marginalised.

    For each candidate theta, A = I_x cos(theta) + I_y sin(theta); S_tA = sum(-I_t A), S_AA = sum(A^2);
    RSS = S_tt if S_tA > 0 (speed forced <= 0), else S_tt - S_tA^2 / S_AA; log L = -RSS / (2 sigma^2).
    """
    It_neg = -It
    S_tt = np.sum(It_neg * It_neg)
    S_xx, S_yy, S_xy = np.sum(Ix * Ix), np.sum(Iy * Iy), np.sum(Ix * Iy)
    S_xt, S_yt = np.sum(Ix * It_neg), np.sum(Iy * It_neg)
    thetas = np.arange(0.0, 360.0, step_deg, dtype=np.float64)
    c, s = np.cos(np.deg2rad(thetas)), np.sin(np.deg2rad(thetas))
    S_tA = c * S_xt + s * S_yt
    S_AA = c * c * S_xx + 2.0 * c * s * S_xy + s * s * S_yy + 1e-15
    rss = np.where(S_tA > 0, S_tt, S_tt - S_tA * S_tA / S_AA)
    log_l = -rss / (2.0 * sigma * sigma)
    posterior = np.exp(log_l - np.max(log_l))
    return float(thetas[int(np.argmax(posterior))] % 360.0)


def angular_error(est_deg: float, gt_deg: float) -> float:
    """|estimate - ground truth| in degrees, wrapped to [0, 180]."""
    return abs(((est_deg - gt_deg + 180.0) % 360.0) - 180.0)


def analyze(movie: np.ndarray, mask: np.ndarray, radius: int, stride: int, gt_direction_deg: float) -> dict:
    """One stimulus x one scotoma mask: masked pixels set to 0 on every frame, then speed and direction."""
    Ix, Iy, It = derivatives(movie * mask[None, :, :])
    ux, uy, n_valid = lucas_kanade(Ix, Iy, It, radius, stride)
    est = direction_map(Ix, Iy, It)
    return {"ux": ux, "uy": uy, "n_valid": n_valid, "direction_deg": est,
            "direction_error_deg": angular_error(est, gt_direction_deg)}
