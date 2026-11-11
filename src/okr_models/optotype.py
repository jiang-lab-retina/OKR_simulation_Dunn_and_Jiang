"""Orientation discrimination by template matching (Methods, Orientation discrimination).

The test image (the E at one orientation, blurred, with scotomas) is compared with four
templates (the unblurred E at 0, 90, 180 and 270 deg, with the same scotomas) by normalised
cross-correlation. The reported confidence is the softmax (temperature 0.05) of the four
correlations, taken at the true orientation.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import params as P
from . import stimuli


def templates_and_tests(sigma: float) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Templates (rotated E) and test images (rotated E, blurred with ``sigma`` px)."""
    base = stimuli.optotype_e()
    rotated = {a: stimuli.rotate(base, a) for a in P.OPTOTYPE["orientations"]}
    tests = {a: stimuli.gaussian_blur(img, sigma) for a, img in rotated.items()}
    return rotated, tests


def apply_scotomas(image: np.ndarray, mask: np.ndarray, value: float = P.OPTOTYPE["background"]) -> np.ndarray:
    """Pixels under a scotoma (mask False) take the background grey."""
    out = image.copy()
    out[~mask] = value
    return out


def _whole_image_ncc(image: np.ndarray, template: np.ndarray) -> float:
    img, tpl = image.astype(float).ravel(), template.astype(float).ravel()
    si, st = np.std(img), np.std(tpl)
    if si < 1e-10 or st < 1e-10:
        return 0.0
    return float(np.sum((img - img.mean()) * (tpl - tpl.mean())) / (len(img) * si * st))


def correlation(image: np.ndarray, template: np.ndarray, mask: np.ndarray) -> float:
    """Maximum normalised cross-correlation (OpenCV TM_CCOEFF_NORMED) of the template over the image.

    Both are converted to 8 bit (truncation, grey 0.5 -> 127) and get the scotomas (grey). The
    template is cropped to the bounding box of its non-grey pixels plus a margin of
    max(10 px, 10 %) before matching.
    """
    img = (image * 255).astype(np.uint8)
    tpl = (template * 255).astype(np.uint8)
    grey = int(0.5 * 255)
    img[~mask] = grey
    tpl[~mask] = grey
    non_bg = np.abs(tpl.astype(float) - 0.5 * 255) > 20
    if not np.any(non_bg):
        return 0.0
    rows, cols = np.where(np.any(non_bg, axis=1))[0], np.where(np.any(non_bg, axis=0))[0]
    rmin, rmax, cmin, cmax = rows[0], rows[-1], cols[0], cols[-1]
    pad = max(10, int(0.1 * min(rmax - rmin, cmax - cmin)))
    rmin, rmax = max(0, rmin - pad), min(tpl.shape[0], rmax + pad)
    cmin, cmax = max(0, cmin - pad), min(tpl.shape[1], cmax + pad)
    crop = tpl[rmin:rmax, cmin:cmax]
    if crop.shape[0] >= img.shape[0] or crop.shape[1] >= img.shape[1]:
        return _whole_image_ncc(img, tpl)
    return float(np.max(cv2.matchTemplate(img, crop, cv2.TM_CCOEFF_NORMED)))


def softmax(scores: dict[int, float], temperature: float) -> dict[int, float]:
    keys = sorted(scores)
    z = np.array([scores[k] for k in keys]) / temperature
    e = np.exp(z - np.max(z))
    return {k: float(v) for k, v in zip(keys, e / np.sum(e))}


def trial(test_image: np.ndarray, gt_orientation: int, mask: np.ndarray, templates: dict[int, np.ndarray]) -> dict:
    """One trial: confidence at the true orientation and whether the best match is correct."""
    masked = apply_scotomas(test_image, mask)
    corr = {o: correlation(masked, t, mask) for o, t in templates.items()}
    prob = softmax(corr, P.OPTOTYPE["softmax_temperature"])
    predicted = max(corr, key=lambda k: corr[k])
    return {"confidence": prob[gt_orientation], "correct": predicted == gt_orientation}
