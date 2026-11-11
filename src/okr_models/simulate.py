"""Run the models for every condition used in Fig 4 and Fig S5.

Results are written per condition and mask realisation (``<results>/<condition>/seed_<s>.npz``),
so an interrupted run resumes where it stopped. ``collect`` stacks them into arrays of shape
(n_realisations, n_levels).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import motion, optotype, stimuli
from . import params as P


@dataclass(frozen=True)
class Condition:
    name: str
    model: str              # "motion" | "optotype"
    stimulus: str           # "dots" | "grating" | "E"
    sigma: float            # Gaussian blur, px
    window: str | None = None
    n_cycles: int | None = None


def _motion(window, stimulus, sigma, n=None):
    tag = "dots" if stimulus == "dots" else f"grating{n}"
    return Condition(f"{window}_{tag}_s{sigma:g}", "motion", stimulus, float(sigma), window, n)


CONDITIONS: dict[str, Condition] = {c.name: c for c in [
    _motion("r3", "dots", 0), _motion("r3", "dots", 10),
    *[_motion("r3", "grating", 0, n) for n in P.GRATING_CYCLES],
    _motion("rw120", "dots", 10),
    *[_motion("rw120", "grating", 10, n) for n in P.GRATING_CYCLES],
    *[_motion("rw105", "dots", s) for s in (0, 3, 10, 30)],
    *[Condition(f"E_s{s:g}", "optotype", "E", float(s)) for s in P.OPTOTYPE_SIGMAS],
]}

# Which conditions each figure needs
PANELS = {
    "4a": ["r3_dots_s0"] + [f"r3_grating{n}_s0" for n in P.GRATING_CYCLES],
    "4b-d": ["r3_dots_s0", "r3_dots_s10"],
    "4e": ["rw120_dots_s10"] + [f"rw120_grating{n}_s10" for n in P.GRATING_CYCLES],
    "4f-h": [f"rw105_dots_s{s}" for s in (0, 3, 10, 30)],
    "4i-l": [f"E_s{s:g}" for s in P.OPTOTYPE_SIGMAS],
    "S5": ["r3_dots_s0", "r3_dots_s10"],
}


# ---------------------------------------------------------------------------- masks (cached)

def mask_file(results: Path, seed: int) -> Path:
    return results / "masks" / f"seed_{seed}.npz"


def _make_mask_file(args) -> int:
    results, seed = args
    path = mask_file(Path(results), seed)
    if not path.exists():
        masks = stimuli.scotoma_masks(seed)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, masks=masks, levels_pct=np.array(P.SCOTOMA_LEVELS_PCT), seed=np.int64(seed))
        tmp.replace(path)
    return seed


def ensure_masks(results: Path, seeds, workers: int, log=print) -> None:
    todo = [s for s in seeds if not mask_file(results, s).exists()]
    if not todo:
        return
    mask_file(results, todo[0]).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s in ex.map(_make_mask_file, [(str(results), s) for s in todo]):
            log(f"  masks seed {s}")
    log(f"  {len(todo)} mask realisations in {time.time() - t0:.0f} s")


def load_masks(results: Path, seed: int) -> np.ndarray:
    with np.load(mask_file(results, seed)) as z:
        return z["masks"]


# ---------------------------------------------------------------------------- workers

_MOVIE: np.ndarray | None = None


def _init(movie: np.ndarray | None) -> None:
    global _MOVIE
    _MOVIE = movie
    cv2.setNumThreads(1)


def _motion_unit(args):
    li, mask, radius, stride, gt = args
    return li, motion.analyze(_MOVIE, mask, radius, stride, gt)


def _optotype_unit(args):
    results, seed, sigma = args
    templates, tests = optotype.templates_and_tests(sigma)
    masks = load_masks(Path(results), seed)
    orients = P.OPTOTYPE["orientations"]
    conf = np.zeros((len(orients), len(masks)), dtype=np.float32)
    correct = np.zeros((len(orients), len(masks)), dtype=bool)
    for oi, gt in enumerate(orients):
        for li, m in enumerate(masks):
            r = optotype.trial(tests[gt], gt, m, templates)
            conf[oi, li], correct[oi, li] = r["confidence"], r["correct"]
    return seed, conf, correct


def stimulus_movie(c: Condition) -> np.ndarray:
    raw = stimuli.dots_movie() if c.stimulus == "dots" else stimuli.grating_movie(c.n_cycles)
    return motion.prepare_movie(raw, c.sigma)


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def run_condition(c: Condition, results: Path, seeds, workers: int, log=print) -> None:
    out = results / c.name
    todo = [s for s in seeds if not (out / f"seed_{s}.npz").exists()]
    if not todo:
        log(f"{c.name}: complete")
        return
    ensure_masks(results, todo, workers, log)
    t0 = time.time()
    levels = np.array(P.SCOTOMA_LEVELS_PCT)
    if c.model == "optotype":
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(None,)) as ex:
            for seed, conf, correct in ex.map(_optotype_unit, [(str(results), s, c.sigma) for s in todo]):
                _save(out / f"seed_{seed}.npz", confidence=conf, correct=correct, levels_pct=levels, seed=np.int64(seed))
                log(f"  {c.name} seed {seed}")
    else:
        radius, stride = P.WINDOWS[c.window]
        gt = P.GT_DIRECTION_DEG[c.stimulus]
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(stimulus_movie(c),)) as ex:
            for s in todo:
                ts = time.time()
                masks = load_masks(results, s)
                res = dict(ex.map(_motion_unit, [(li, masks[li], radius, stride, gt) for li in range(len(masks))]))
                keys = ("ux", "uy", "n_valid", "direction_deg", "direction_error_deg")
                arrays = {k: np.array([res[li][k] for li in range(len(masks))]) for k in keys}
                _save(out / f"seed_{s}.npz", **arrays, levels_pct=levels, seed=np.int64(s))
                log(f"  {c.name} seed {s}: {time.time() - ts:.1f} s")
    log(f"{c.name}: {len(todo)} realisations in {time.time() - t0:.0f} s")


def collect(results: Path, name: str, seeds=P.MASK_SEEDS) -> dict[str, np.ndarray]:
    """Stack the per-realisation results of one condition: arrays (n_realisations, n_levels), or
    (n_realisations x 4 orientations, n_levels) for the optotype. Missing realisations are skipped."""
    files = [results / name / f"seed_{s}.npz" for s in seeds]
    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"no results for {name} in {results}; run `okr-models run` first")
    parts = [dict(np.load(f)) for f in files]
    out = {"levels_pct": parts[0]["levels_pct"], "seeds": np.array([int(p["seed"]) for p in parts])}
    if "confidence" in parts[0]:
        out["confidence"] = np.concatenate([p["confidence"] for p in parts], axis=0)   # orientation-major per seed
        out["correct"] = np.concatenate([p["correct"] for p in parts], axis=0)
    else:
        for k in ("ux", "uy", "n_valid", "direction_error_deg"):
            out[k] = np.stack([p[k] for p in parts])
        out["speed"] = np.hypot(out["ux"], out["uy"])
    return out
