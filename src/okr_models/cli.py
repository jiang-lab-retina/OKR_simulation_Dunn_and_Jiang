"""Command line: ``okr-models run | figures | demo`` (see README)."""

from __future__ import annotations

import os

# One thread per process for the linear-algebra libraries: the runs parallelise over processes,
# and single-threaded sums are reproducible. Must be set before NumPy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import params as P  # noqa: E402
from . import simulate  # noqa: E402


def _seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    bad = [s for s in out if s not in P.MASK_SEEDS]
    if bad:
        raise argparse.ArgumentTypeError(f"seeds must be within 100-150 (the paper's 51 realisations): {bad}")
    return out


def cmd_run(a) -> int:
    names = []
    for f in a.figure:
        names += list(simulate.CONDITIONS) if f == "all" else simulate.PANELS[f]
    names = list(dict.fromkeys(names))
    print(f"{len(names)} conditions x {len(a.seeds)} mask realisations, {a.workers} workers -> {a.results}")
    t0 = time.time()
    for n in names:
        simulate.run_condition(simulate.CONDITIONS[n], a.results, a.seeds, a.workers)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


def cmd_figures(a) -> int:
    from .figures import make_figures
    make_figures(a.results, a.out, a.okr_csv)
    return 0


# Values of the full run for mask realisation 100 (seed 100) at 0 %, 50 % and 95 % scotomas.
DEMO = ["r3_dots_s10", "rw105_dots_s10", "rw120_grating8_s10", "E_s10"]
DEMO_EXPECTED: dict[str, dict[str, list[float]]] = {
    "r3_dots_s10": {"speed_px_per_frame": [11.540521048577281, 8.68436831202484, 3.5008652408763],
                    "direction_error_deg": [0.5, 0.25, 5.0]},
    "rw105_dots_s10": {"speed_px_per_frame": [10.465666495309346, 0.9049498870687925, 0.1724311642683262],
                       "direction_error_deg": [0.5, 0.25, 5.0]},
    "rw120_grating8_s10": {"speed_px_per_frame": [0.0, 0.7391954785186438, 0.17122422520423794],
                           "direction_error_deg": [83.75, 0.25, 4.75]},
    "E_s10": {"mean_confidence": [0.9965686202049255, 0.9487642198801041, 0.8426978141069412]},
}


def cmd_demo(a) -> int:
    results = a.results
    t0 = time.time()
    for n in DEMO:
        simulate.run_condition(simulate.CONDITIONS[n], results, [100], a.workers)
    print(f"\ncomputed in {time.time() - t0:.0f} s; values at 0 %, 50 % and 95 % scotomas (mask realisation 100):")
    got: dict[str, dict[str, list[float]]] = {}
    idx = [P.SCOTOMA_LEVELS_PCT.index(v) for v in (0.0, 50.0, 95.0)]
    for n in DEMO:
        d = simulate.collect(results, n, seeds=[100])
        if "speed" in d:
            got[n] = {"speed_px_per_frame": [float(d["speed"][0, i]) for i in idx],
                      "direction_error_deg": [float(d["direction_error_deg"][0, i]) for i in idx]}
        else:
            got[n] = {"mean_confidence": [float(d["confidence"][:, i].mean()) for i in idx]}
        print(f"  {n:22s} {json.dumps(got[n])}")
    ok = True
    for n, metrics in DEMO_EXPECTED.items():
        for k, want in metrics.items():
            if not np.allclose(got[n][k], want, rtol=1e-6, atol=1e-9):
                ok = False
                print(f"MISMATCH {n} {k}: got {got[n][k]}, expected {want}")
    if DEMO_EXPECTED:
        print("demo matches the expected values" if ok else "demo differs from the expected values")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="okr-models", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    workers = max(1, (os.cpu_count() or 2) - 1)

    r = sub.add_parser("run", help="run the models (resumable)")
    r.add_argument("--figure", nargs="+", default=["all"], choices=["all", *simulate.PANELS],
                   help="which panels to compute (default: all)")
    r.add_argument("--seeds", type=_seeds, default=list(P.MASK_SEEDS), help="mask realisations, e.g. 100-150 or 100,101")
    r.add_argument("--workers", type=int, default=workers)
    r.add_argument("--results", type=Path, default=Path("results"))
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("figures", help="draw Fig 4 and Fig S5 from the results")
    f.add_argument("--results", type=Path, default=Path("results"))
    f.add_argument("--out", type=Path, default=Path("figures"))
    f.add_argument("--okr-csv", type=Path, default=None, help="optional human OKR data for Fig 4h (format in README)")
    f.set_defaults(func=cmd_figures)

    d = sub.add_parser("demo", help="a few minutes: one mask realisation, four conditions, checked against expected values")
    d.add_argument("--workers", type=int, default=workers)
    d.add_argument("--results", type=Path, default=Path("results_demo"))
    d.set_defaults(func=cmd_demo)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
