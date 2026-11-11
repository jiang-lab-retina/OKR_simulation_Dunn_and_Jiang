"""Model stimuli: moving dots, drifting sinusoidal gratings, scotoma masks and the optotype E.

Every generator is deterministic (fixed seeds), so the stimuli are regenerated rather than shipped.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import params as P


# ---------------------------------------------------------------------------- moving dots

def dots_movie(seed: int = P.DOTS["seed"], size: int = P.IMAGE_SIZE) -> np.ndarray:
    """Movie of moving dots, uint8 (frames, size, size), values {background, foreground}.

    300 discs of diameter 39 px move upward at 1200 px/s (13.33 px/frame at 90 frames/s). Each dot
    is visible for 150-200 ms, invisible for 150-200 ms, then reappears at a random place; dots
    wrap vertically. Dots move before each frame is drawn. The state is float32 and the
    arithmetic follows NumPy >= 2 type promotion; the random draws are made in a fixed order
    (4 x 300 values per frame), so the movie is identical on every machine.
    """
    d = P.DOTS
    rng = np.random.default_rng(seed)
    n, frames, dt_ms = d["num_particles"], d["num_frames"], 1000.0 / d["fps"]
    dia = d["particle_diameter_px"]
    h = w = size
    movie = np.full((frames, h, w), d["background"], dtype=np.uint8)

    x = rng.uniform(0, max(1, w - dia), size=n).astype(np.float32)
    y = rng.uniform(0, max(1, h - dia), size=n).astype(np.float32)
    visible = np.ones(n, dtype=bool)
    timer = rng.uniform(*d["initial_timer_ms_range"], size=n).astype(np.float32)
    r = dia / 2.0

    for t in range(frames):
        new_vis = rng.integers(d["visible_ms_range"][0], d["visible_ms_range"][1] + 1, size=n).astype(np.float32)
        new_invis = rng.integers(d["invisible_ms_range"][0], d["invisible_ms_range"][1] + 1, size=n).astype(np.float32)
        new_x = rng.uniform(0, max(1, w - dia), size=n).astype(np.float32)
        new_y = rng.uniform(0, max(1, h - dia), size=n).astype(np.float32)

        timer -= dt_ms
        expired = timer <= 0
        hide = expired & visible
        show = expired & ~visible
        k_hide = np.cumsum(hide) - 1          # the k-th dot changing state takes the k-th draw
        k_show = np.cumsum(show) - 1
        timer[hide] = new_invis[k_hide[hide]]
        visible[hide] = False
        x[show] = new_x[k_show[show]]
        y[show] = new_y[k_show[show]]
        timer[show] = new_vis[k_show[show]]
        visible[show] = True

        y += -d["velocity_px_per_sec"] / 1000.0 * dt_ms      # upward: y decreases
        y[y < -dia] += h + dia
        y[y > h + dia] -= h + dia

        for i in np.flatnonzero(visible):                     # filled discs, top-left corner at (x, y)
            cx, cy = x[i] + r, y[i] + r
            y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
            x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            rows = np.arange(y0, y1)[:, None]
            cols = np.arange(x0, x1)[None, :]
            disc = (cols - cx) ** 2 + (rows - cy) ** 2 <= r * r
            movie[t, y0:y1, x0:x1][disc] = d["foreground"]
    return movie


# ---------------------------------------------------------------------------- gratings

def grating_movie(n_cycles: int, size: int = P.IMAGE_SIZE, num_frames: int = P.DOTS["num_frames"],
                  shift_px_per_frame: float = P.GT_SPEED_PX_PER_FRAME) -> np.ndarray:
    """Sinusoidal grating, float32 (frames, size, size) in [0, 1], ``n_cycles`` per image width,
    drifting rightward at 13.33 px/frame (the speed of the dots)."""
    x = np.arange(size, dtype=np.float32)
    movie = np.empty((num_frames, size, size), dtype=np.float32)
    for t in range(num_frames):
        phase = 2.0 * np.pi * (n_cycles / size) * (x - shift_px_per_frame * t)
        movie[t] = (0.5 + 0.5 * np.sin(phase))[None, :]
    return movie


# ---------------------------------------------------------------------------- scotoma masks

def _render_holes(size: int, cx: np.ndarray, cy: np.ndarray, radius: int) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    Y, X = np.ogrid[:size, :size]
    r2 = float(radius) * float(radius)
    for x0, y0 in zip(cx, cy):
        mask[(X - x0) ** 2 + (Y - y0) ** 2 <= r2] = False
    return mask


def scotoma_mask(rng: np.random.Generator, level_pct: float, size: int = P.IMAGE_SIZE,
                 radius: int = P.HOLE_RADIUS_PX, tol: float = P.MASK_TOLERANCE_PCT,
                 max_holes_factor: float = P.MASK_MAX_HOLES_FACTOR) -> np.ndarray:
    """One mask (True = visible pixel) whose fraction of lost pixels is ``level_pct`` +/- ``tol``.

    Disc-shaped scotomas (radius 40 px) are placed at random. The number of discs is found by a
    binary search over a fixed list of random centres; if the search cannot reach the tolerance
    (overlapping discs at high coverage), discs are added or removed one at a time.
    """
    if level_pct <= 0:
        return np.ones((size, size), dtype=bool)
    if level_pct >= 99.5:
        return np.zeros((size, size), dtype=bool)
    target = float(level_pct)
    estimate = max(1, int(round(target / 100.0 * size * size / (np.pi * radius * radius))))
    hi = min(max(1, int(estimate * max_holes_factor)), 5000)
    cx = rng.uniform(0, size, size=hi).astype(np.float32)
    cy = rng.uniform(0, size, size=hi).astype(np.float32)

    best, best_err, best_n, lo = np.ones((size, size), dtype=bool), float("inf"), 0, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        m = _render_holes(size, cx[:mid], cy[:mid], radius)
        err = 100.0 * (1.0 - m.mean()) - target
        if abs(err) < best_err:
            best, best_err, best_n = m, abs(err), mid
        if abs(err) <= tol:
            return m
        if err < 0:
            lo = mid + 1
        else:
            hi = mid - 1
    if best_n <= 0:
        return best

    cx, cy, m = cx[:best_n].copy(), cy[:best_n].copy(), best
    for _ in range(5000):
        err = 100.0 * (1.0 - m.mean()) - target
        if abs(err) <= tol:
            break
        if err < 0:
            cx = np.append(cx, float(rng.uniform(0, size)))
            cy = np.append(cy, float(rng.uniform(0, size)))
        elif len(cx) > 1:
            i = int(rng.integers(0, len(cx)))
            cx, cy = np.delete(cx, i), np.delete(cy, i)
        else:
            break
        m = _render_holes(size, cx, cy, radius)
    return m


def scotoma_masks(seed: int, levels=P.SCOTOMA_LEVELS_PCT, size: int = P.IMAGE_SIZE) -> np.ndarray:
    """The masks of one realisation, bool (len(levels), size, size). One random generator per seed
    is used for the levels in increasing order, so a realisation always starts at 0 %."""
    rng = np.random.default_rng(int(seed))
    return np.stack([scotoma_mask(rng, float(lv), size) for lv in levels])


# ---------------------------------------------------------------------------- optotype E

def optotype_e(size: int = P.IMAGE_SIZE, font_size: int = P.OPTOTYPE["font_size"],
               background: float = P.OPTOTYPE["background"], foreground: float = P.OPTOTYPE["foreground"]) -> np.ndarray:
    """Snellen-style E of five strokes (4 x 5 units of font_size // 5 px), opening to the right (0 deg)."""
    img = np.full((size, size), background, dtype=np.float32)
    unit = font_size // 5
    ew, eh = 4 * unit, 5 * unit
    x0, y0 = (size - ew) // 2, (size - eh) // 2
    img[y0:y0 + eh, x0:x0 + unit] = foreground
    for row in (0, 2, 4):
        img[y0 + row * unit:y0 + (row + 1) * unit, x0:x0 + ew] = foreground
    return img


def rotate(image: np.ndarray, angle_deg: float, background: float = P.OPTOTYPE["background"]) -> np.ndarray:
    """Counter-clockwise rotation about the image centre, bilinear, background outside."""
    h, w = image.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=background).astype(np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of one image (float32); sigma <= 0 returns the image unchanged."""
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image.astype(np.float32), ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
