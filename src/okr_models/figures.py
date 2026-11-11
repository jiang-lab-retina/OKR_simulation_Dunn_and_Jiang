"""Fig 4 (a-l) and Fig S5 (a-c) from the simulation results, with every plotted value in a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from . import motion, optotype, stimuli  # noqa: E402
from . import params as P  # noqa: E402
from .simulate import collect, load_masks, mask_file  # noqa: E402

GT = P.GT_SPEED_PX_PER_FRAME
NO_BLUR, BLUR = "steelblue", "coral"
SIGMA_COLORS = {0: "#1f77b4", 3: "#ff7f0e", 10: "#d62728", 30: "#9467bd"}
E_COLORS = {0.0: "steelblue", 10.0: "darkorange", 100.0: "darkred"}
FIG4E_COLORS = {2: "#C77CFF", 4: "#00BFC4", 8: "#7CAE00", 16: "#F8766D", 32: "#CD9600"}
INSET_SEED, INSET_LEVEL = 100, 50.0


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def _letter(ax, s: str) -> None:
    ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="left")


def _mean_sd(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(traces, axis=0), np.nanstd(traces, axis=0)


# ---------------------------------------------------------------------------- contamination model

def contamination(p, a0, k, alpha):
    """y(p) = a0 (1 - p) exp(-k p) + alpha p, with p the scotoma fraction."""
    return a0 * (1.0 - p) * np.exp(-k * p) + alpha * p


def fit_contamination(p: np.ndarray, y: np.ndarray, a0: float) -> tuple[float, float, float]:
    """Fit the two free parameters k and alpha with a0 fixed to the measured baseline.
    Returns (k, alpha, R^2); several starting points, least squares, 0 <= k <= 100, 0 <= alpha <= 1."""
    def f(pp, k, alpha):
        return contamination(pp, a0, k, alpha)
    best = (np.inf, None)
    for k0 in (0.01, 0.1, 0.5, 2, 5, 15, 30):
        for al0 in (0.01, 0.1, 0.3, 0.5):
            try:
                popt, _ = curve_fit(f, p, y, p0=[k0, al0], bounds=([0, 0], [100, 1]), maxfev=50000)
            except RuntimeError:
                continue
            cost = float(np.sum((y - f(p, *popt)) ** 2))
            if cost < best[0]:
                best = (cost, popt)
    k, alpha = (float(v) for v in best[1])
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - best[0] / ss_tot if ss_tot > 1e-15 else 0.0
    return k, alpha, r2


# ---------------------------------------------------------------------------- insets

def _inset(ax, img, label, x0, cmap_max=1.0, w=0.28, y0=0.06):
    ia = ax.inset_axes([x0, y0, w, w])
    ia.imshow(img, cmap="gray", vmin=0, vmax=cmap_max, interpolation="antialiased")
    ia.set_xticks([])
    ia.set_yticks([])
    for s in ia.spines.values():
        s.set_visible(False)
    ia.set_title(label, fontsize=8, pad=2)


def _inset_mask(results: Path) -> np.ndarray:
    li = int(np.argmin(np.abs(np.array(P.SCOTOMA_LEVELS_PCT) - INSET_LEVEL)))
    if mask_file(results, INSET_SEED).exists():
        return load_masks(results, INSET_SEED)[li]
    return stimuli.scotoma_masks(INSET_SEED, P.SCOTOMA_LEVELS_PCT[:li + 1])[li]


def _dots_insets(ax, frame: np.ndarray, mask: np.ndarray) -> None:
    _inset(ax, frame, "0%", 0.03)
    _inset(ax, frame * mask, "50%", 0.45)


# ---------------------------------------------------------------------------- panels

def panel_4a(ax, results, rec):
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(P.GRATING_CYCLES):
        d = collect(results, f"r3_grating{n}_s0")
        m, s = _mean_sd(d["speed"])
        c = cmap(i / (len(P.GRATING_CYCLES) - 1))
        ax.fill_between(d["levels_pct"], m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.plot(d["levels_pct"], m, color=c, marker="o", ms=2.5, lw=1.5, label=str(n))
        rec[f"4a_grating{n}"] = {"x": d["levels_pct"], "mean": m, "sd": s, "n": len(d["speed"])}
    d = collect(results, "r3_dots_s0")
    m, s = _mean_sd(d["speed"])
    ax.fill_between(d["levels_pct"], m - s, m + s, color="#333333", alpha=0.15, lw=0)
    dots_line, = ax.plot(d["levels_pct"], m, color="#333333", marker="x", ms=3, lw=1.5)
    rec["4a_dots"] = {"x": d["levels_pct"], "mean": m, "sd": s, "n": len(d["speed"])}
    h, lab = ax.get_legend_handles_labels()
    leg = ax.legend(h[::-1], lab[::-1], title="Gratings (cycles per image width)", loc="upper right",
                    fontsize=7, title_fontsize=7, frameon=False)
    ax.add_artist(leg)
    ax.legend([dots_line], ["Moving Dots"], loc="upper center", fontsize=7, frameon=False)
    ax.set_title("Without Gaussian Blur: gratings vs dots")
    ax.set_ylabel(r"Mean LK speed $\|u\|$ (pixel/frame)")


def _normalised_traces(speed: np.ndarray) -> np.ndarray:
    base = speed[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(base > 0, speed / base, np.nan)


def _trace_panel(ax, x, traces, color, alpha=0.15):
    for t in traces:
        ax.plot(x, t, color=color, alpha=alpha, lw=0.8, zorder=1)
    m, s = _mean_sd(traces)
    ax.fill_between(x, m - s, m + s, color="black", alpha=0.2, lw=0, zorder=2)
    ax.plot(x, m, color="black", lw=1.8, marker="o", ms=2.5, zorder=3)
    return m, s


def panels_4bcd(axes, results, rec, frames, mask):
    nb, wb = collect(results, "r3_dots_s0"), collect(results, "r3_dots_s10")
    x = nb["levels_pct"]
    for ax, d, color, key, title, frame in (
            (axes[0], nb, NO_BLUR, "4b", "Dots without Gaussian Blur", frames[0]),
            (axes[1], wb, BLUR, "4c", "Dots with Gaussian Blur (σ = 10)", frames[1])):
        m, s = _trace_panel(ax, x, _normalised_traces(d["speed"]), color)
        rec[key] = {"x": x, "mean": m, "sd": s, "n": len(d["speed"]), "normalised_per_trace_to": "0 %"}
        ax.set_ylim(0, 1.5)
        ax.set_title(title)
        _dots_insets(ax, frame, mask)
    axes[0].set_ylabel("Normalized LK speed")
    ax = axes[2]
    for d, color, label, key in ((nb, NO_BLUR, "Without Blur", "4d_no_blur"), (wb, BLUR, "With Blur (σ = 10)", "4d_blur")):
        m, s = _mean_sd(d["speed"])
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)
        ax.plot(x, m, color=color, lw=1.8, marker="o" if color == NO_BLUR else "s", ms=3, label=label)
        rec[key] = {"x": x, "mean": m, "sd": s, "n": len(d["speed"])}
    ax.axhline(GT, color="red", ls="--", lw=1.5, label=f"Ground Truth ({GT:.2f})")
    ax.set_title("Dots Comparison: without and with blur")
    ax.set_ylabel(r"Mean LK speed $\|u\|$ (pixels/frame)")
    ax.legend(loc="upper left", frameon=False)


def panel_4e(ax, results, rec):
    pf = np.linspace(0, 1, 400)
    for n in P.GRATING_CYCLES:
        d = collect(results, f"rw120_grating{n}_s10")
        y = np.nanmedian(d["speed"], axis=0) / GT
        p = d["levels_pct"] / 100.0
        k, alpha, r2 = fit_contamination(p, y, a0=float(y[0]))
        ax.scatter(d["levels_pct"], y, color=FIG4E_COLORS[n], s=10, zorder=4)
        ax.plot(pf * 100, contamination(pf, y[0], k, alpha), color=FIG4E_COLORS[n], lw=1.5, label=str(n))
        rec[f"4e_grating{n}"] = {"x": d["levels_pct"], "median_norm": y, "a0": y[0], "k": k, "alpha": alpha, "r2": r2}
    d = collect(results, "rw120_dots_s10")
    y = np.nanmedian(d["speed"], axis=0) / GT
    k, alpha, r2 = fit_contamination(d["levels_pct"] / 100.0, y, a0=float(y[0]))
    ax.scatter(d["levels_pct"], y, color="#444444", marker="s", s=14, zorder=4)
    ax.plot(pf * 100, contamination(pf, y[0], k, alpha), color="#444444", lw=1.8, label="Moving Dots")
    rec["4e_dots"] = {"x": d["levels_pct"], "median_norm": y, "a0": y[0], "k": k, "alpha": alpha, "r2": r2}
    ax.set_title(r"Integration window ($R_w$ = 120 px), Blur (σ = 10)")
    ax.set_ylabel(r"LK speed / $u_{\mathrm{norm}}$")
    ax.set_ylim(bottom=0)
    ax.legend(title="Gratings (cycles per image width)", loc="upper right", fontsize=7, title_fontsize=7, frameon=False)


def _w105_curves(results):
    out = {}
    for s in (0, 3, 10, 30):
        d = collect(results, f"rw105_dots_s{s}")
        med = np.nanmedian(d["speed"], axis=0)
        sd = np.nanstd(d["speed"], axis=0, ddof=1)
        base = med[0] if med[0] > 1e-12 else 1.0
        out[s] = {"x": d["levels_pct"], "median": med, "sd": sd, "norm": med / base, "sd_norm": sd / base,
                  "n": len(d["speed"])}
    return out


def panels_4fgh(axes, results, rec, okr):
    curves = _w105_curves(results)
    pf = np.linspace(0, 1, 500)
    for s, c in curves.items():
        k, alpha, r2 = fit_contamination(c["x"] / 100.0, c["norm"], a0=1.0)
        c.update(k=k, alpha=alpha, r2=r2)
        col = SIGMA_COLORS[s]
        axes[0].errorbar(c["x"], c["median"], yerr=c["sd"], fmt="o", color=col, ms=3, alpha=0.6, capsize=2, elinewidth=0.8)
        axes[0].plot(pf * 100, contamination(pf, 1.0, k, alpha) * c["median"][0], color=col, lw=1.6, label=f"σ = {s}")
        axes[1].errorbar(c["x"], c["norm"], yerr=c["sd_norm"], fmt="o", color=col, ms=3, alpha=0.6, capsize=2, elinewidth=0.8)
        axes[1].plot(pf * 100, contamination(pf, 1.0, k, alpha), color=col, lw=1.6,
                     label=f"σ = {s}, k = {k:.2f}, α = {alpha:.2f}, $R^2$ = {r2:.3f}")
        rec[f"4fg_sigma{s}"] = c
    axes[0].axhline(GT, color="gray", ls="--", lw=1.2, label="Ground truth")
    axes[0].set_title(r"Absolute LK Speed ($R_w$ = 105 px)")
    axes[0].set_ylabel(r"LK speed $u$ (pixel/frame)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(title="Blur σ", loc="upper right", frameon=False)
    axes[1].set_title(r"Normalized LK Speed ($R_w$ = 105 px)")
    axes[1].set_ylabel("Normalized LK speed")
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend(title="Blur σ", loc="upper right", fontsize=6, title_fontsize=6.5, frameon=False)

    ax = axes[2]
    c = curves[10]
    ax.errorbar(c["x"], c["norm"], yerr=c["sd_norm"], fmt="o-", color=SIGMA_COLORS[10], ms=3, lw=1.5, capsize=2,
                elinewidth=0.8, label="Simulation (σ = 10)")
    if okr is not None:
        ax.errorbar(okr["loss_pct"], okr["mean_norm"], yerr=okr["sd_norm"], fmt="s-", color="#2ca02c", ms=3, lw=1.5,
                    capsize=2, elinewidth=0.8, label="Observed OKR")
        rec["4h_okr"] = okr
    else:
        ax.text(0.97, 0.55, "Observed OKR (Fig 1b):\nhuman data, available\nfrom the authors",
                transform=ax.transAxes, ha="right", va="center", fontsize=7, color="#2ca02c")
    rec["4h_simulation"] = {"x": c["x"], "norm": c["norm"], "sd_norm": c["sd_norm"]}
    ax.set_title("Simulation vs. Observed OKR")
    ax.set_ylabel("Normalized LK speed")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper right", frameon=False)


def panels_4ijkl(axes, results, rec, mask):
    stats = {}
    for ax, s, title in zip(axes[:3], P.OPTOTYPE_SIGMAS,
                            ("E without Gaussian Blur", "E with Gaussian Blur (σ = 10)", "E with Gaussian Blur (σ = 100)")):
        d = collect(results, f"E_s{s:g}")
        x, tr = d["levels_pct"], d["confidence"].astype(float) * 100
        col = E_COLORS[s]
        for t in tr:
            ax.plot(x, t, color=col, alpha=0.08, lw=0.5, zorder=1)
        m, sd = _mean_sd(tr)
        ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.3, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=2.0, marker="o", ms=3, zorder=3)
        lo, hi = np.min(m - sd), np.max(m + sd)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_title(title)
        test = optotype.templates_and_tests(s)[1][0]
        _inset(ax, test, "0%", 0.03)
        _inset(ax, optotype.apply_scotomas(test, mask), "50%", 0.45)
        stats[s] = (x, m, sd)
        rec[f"4ijk_sigma{s:g}"] = {"x": x, "mean": m, "sd": sd, "n": len(tr)}
    axes[0].set_ylabel("Confidence (%)")
    ax = axes[3]
    for s, (x, m, sd), label in zip(P.OPTOTYPE_SIGMAS, stats.values(),
                                    ("Without Blur", "With Blur (σ = 10)", "With Blur (σ = 100)")):
        ax.fill_between(x, m - sd, m + sd, color=E_COLORS[s], alpha=0.2, lw=0)
        ax.plot(x, m, color=E_COLORS[s], lw=2.0, marker="o", ms=3, label=label)
    ax.set_title("E Comparison: without and with blur")
    ax.legend(loc="lower left", frameon=False)


def panels_s5(axes, results, rec, frames, mask):
    nb, wb = collect(results, "r3_dots_s0"), collect(results, "r3_dots_s10")
    x = nb["levels_pct"]
    for ax, d, color, key, title, frame in (
            (axes[0], nb, NO_BLUR, "S5a", "Dots without Gaussian Blur", frames[0]),
            (axes[1], wb, BLUR, "S5b", "Dots with Gaussian Blur (σ = 10)", frames[1])):
        m, s = _trace_panel(ax, x, d["direction_error_deg"], color)
        rec[key] = {"x": x, "mean": m, "sd": s, "n": len(d["direction_error_deg"])}
        ax.set_title(title)
        _inset(ax, frame, "0%", 0.08, y0=0.62)
        _inset(ax, frame * mask, "50%", 0.5, y0=0.62)
    ax = axes[2]
    for d, color, label in ((nb, NO_BLUR, "Without Blur"), (wb, BLUR, "With Blur (σ = 10)")):
        m, s = _mean_sd(d["direction_error_deg"])
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.2, lw=0)
        ax.plot(x, m, color=color, lw=1.8, marker="o" if color == NO_BLUR else "s", ms=3, label=label)
    ax.set_title("Dots Comparison: without and with blur")
    ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("|Estimated − Ground Truth| (deg)")
    ymax = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(0, ymax)


# ---------------------------------------------------------------------------- figures

def load_okr_csv(path: Path) -> dict:
    """Optional human data for Fig 4h. CSV: first row ``scotoma_fraction,<f1>,<f2>,...``; then one row per
    subject ``<id>,<gain at f1>,...``. Returns loss (%), mean and SD (ddof 1) across subjects, divided by
    the mean at the smallest loss."""
    rows = [r.split(",") for r in Path(path).read_text().strip().splitlines()]
    x = np.array([float(v) for v in rows[0][1:]]) * 100.0
    vals = np.array([[float(v) if v.strip() else np.nan for v in r[1:]] for r in rows[1:]])
    mean, sd = np.nanmean(vals, axis=0), np.nanstd(vals, axis=0, ddof=1)
    o = np.argsort(x)
    base = mean[o][0] if mean[o][0] > 1e-12 else 1.0
    return {"loss_pct": x[o], "mean_norm": mean[o] / base, "sd_norm": sd[o] / base, "n_subjects": len(rows) - 1}


def _dots_frames() -> tuple[np.ndarray, np.ndarray]:
    raw = stimuli.dots_movie()
    return motion.prepare_movie(raw[:1], 0)[0], motion.prepare_movie(raw[:1], 10)[0]


def _finish(ax):
    ax.set_xlabel("% Scotomas")
    ax.set_xlim(-3, 98)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _save(fig, out: Path, stem: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")]
    for p in paths:
        fig.savefig(p, dpi=300 if p.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return paths


def make_figures(results: Path, out: Path, okr_csv: Path | None = None, log=print) -> list[Path]:
    _style()
    results, out = Path(results), Path(out)
    okr = load_okr_csv(okr_csv) if okr_csv else None
    frames, mask = _dots_frames(), _inset_mask(results)
    rec: dict = {}

    fig, ax = plt.subplots(3, 4, figsize=(17, 12))
    fig.subplots_adjust(wspace=0.32, hspace=0.42)
    panel_4a(ax[0, 0], results, rec)
    panels_4bcd(ax[0, 1:], results, rec, frames, mask)
    panel_4e(ax[1, 0], results, rec)
    panels_4fgh(ax[1, 1:], results, rec, okr)
    panels_4ijkl(ax[2, :], results, rec, mask)
    for a, letter in zip(ax.ravel(), "abcdefghijkl"):
        _finish(a)
        _letter(a, letter)
    paths = _save(fig, out, "Fig4")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    fig.subplots_adjust(wspace=0.25)
    panels_s5(ax, results, rec, frames, mask)
    for a, letter in zip(ax, "abc"):
        _finish(a)
        _letter(a, letter)
    paths += _save(fig, out, "FigS5")

    values = out / "plotted_values.json"
    values.write_text(json.dumps(_jsonable(rec), indent=1) + "\n")
    paths.append(values)
    ns = {k: v["n"] for k, v in rec.items() if isinstance(v, dict) and "n" in v}
    if any(n < len(P.MASK_SEEDS) * (4 if k.startswith("4ijk") else 1) for k, n in ns.items()):
        log(f"note: some panels use fewer than the paper's 51 mask realisations: {ns}")
    for p in paths:
        log(f"wrote {p}")
    return paths
