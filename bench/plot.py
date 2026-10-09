#!/usr/bin/env python3
"""Runtime and speedup plots for one benchmark round (rust_perf vs RAJAPerf).

Usage: bench/plot.py <round>   e.g. bench/plot.py 2026-10

Reads bench/{nvidia,amd}/<round>/{RAJAPerf-timing-Average[-ltimes].csv,rust-{sync,async}.txt}
and writes bench/<round>/{runtime,speedup}.png plus bench/<round>/summary.md.
The rust .txt tables are used instead of the .csv because older CSVs lacked the launch column.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUND = sys.argv[1] if len(sys.argv) > 1 else "2026-10"
BENCH = Path(__file__).resolve().parent
KERNELS = ["DEL_DOT_VEC_2D", "ENERGY", "FIR", "LTIMES", "MATVEC_3D_STENCIL", "PRESSURE", "VOL3D"]
GPUS = [("nvidia", "H100", "CUDA"), ("amd", "MI250X (1 GCD)", "HIP")]

# categorical slots 1-4 of the reference palette (validated, light surface)
COL = {"base": "#2a78d6", "raja": "#eb6834", "sync": "#1baf7a", "async": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def read_raja(d, api):
    """kernel -> {'base': s, 'raja': s}; the -ltimes file carries the RAJA tunings."""
    out = {}
    for name in ("RAJAPerf-timing-Average.csv", "RAJAPerf-timing-Average-ltimes.csv"):
        f = d / name
        if not f.exists():
            continue
        rows = [[c.strip() for c in l.split(",")] for l in f.read_text().splitlines()]
        variants, tunings = rows[1][1:], rows[2][1:]
        for r in rows[3:]:
            k = r[0].removeprefix("Apps_")
            vals = {}
            for v, t, x in zip(variants, tunings, r[1:]):
                if x == "Not run":
                    continue
                key = "base" if v == f"Base_{api}" else "raja" if v == f"RAJA_{api}" else None
                # several RAJA tunings may exist; keep the fastest
                if key and (key not in vals or float(x) < vals[key]):
                    vals[key] = float(x)
            out.setdefault(k, {}).update(vals)
    return out


def read_rust(f, checksum=False):
    """kernel -> total seconds (launch + sync), or the checksum, from the printed table."""
    out = {}
    for l in f.read_text().splitlines():
        m = re.match(r"^(\w+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", l)
        if m:
            out[m.group(1)] = float(m.group(5) if checksum else m.group(4))
    return out


def read_raja_checksums(f, api):
    """kernel -> Base_<api> checksum from RAJAPerf-checksum.txt."""
    out, kernel = {}, None
    for l in f.read_text().splitlines():
        if l.startswith("Apps_"):
            kernel = l.split()[0].removeprefix("Apps_")
        elif kernel and l.startswith(f"Base_{api}-"):
            out[kernel] = float(l.split()[3])
    return out


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


data = {}
for gpu, label, api in GPUS:
    d = BENCH / gpu / ROUND
    data[gpu] = {"raja": read_raja(d, api), "sync": read_rust(d / "rust-sync.txt"), "async": read_rust(d / "rust-async.txt"),
                 "chk_raja": read_raja_checksums(d / "RAJAPerf-checksum.txt", api),
                 "chk_sync": read_rust(d / "rust-sync.txt", checksum=True),
                 "chk_async": read_rust(d / "rust-async.txt", checksum=True)}

series = [("base", "RAJAPerf Base"), ("raja", "RAJAPerf RAJA"), ("sync", "Rust sync"), ("async", "Rust async")]
x = list(range(len(KERNELS)))
w = 0.19

# --- runtime
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
for ax, (gpu, label, api) in zip(axes, GPUS):
    dd = data[gpu]
    for i, (key, name) in enumerate(series):
        vals = [dd["raja"].get(k, {}).get(key) if key in ("base", "raja") else dd[key].get(k) for k in KERNELS]
        ax.bar([xi + (i - 1.5) * w for xi in x], [v * 1e3 if v else 0 for v in vals], width=w - 0.02,
               color=COL[key], label=name)
    style(ax)
    ax.set_xticks(x, [k.replace("_", "\n") for k in KERNELS], fontsize=7, color=INK2)
    ax.set_title(label, fontsize=10, color=INK, loc="left")
    ax.set_ylabel("total time over all reps (ms)", fontsize=8, color=INK2)
axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle(f"rust_perf vs RAJAPerf, {ROUND} (lower is better)", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout()
out = BENCH / ROUND
out.mkdir(exist_ok=True)
fig.savefig(out / "runtime.png")

# --- speedup relative to RAJAPerf Base (>1 = Rust slower)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
for ax, (gpu, label, api) in zip(axes, GPUS):
    dd = data[gpu]
    for i, key in enumerate(("sync", "async")):
        ratios = [dd[key][k] / dd["raja"][k]["base"] for k in KERNELS]
        bars = ax.bar([xi + (i - 0.5) * (w * 1.6) for xi in x], ratios, width=w * 1.6 - 0.03, color=COL[key],
                      label="Rust sync" if key == "sync" else "Rust async")
        if key == "async":
            # keep the labels clear of the parity line when the bar ends near it
            for b, r in zip(bars, ratios):
                y = b.get_height() if abs(b.get_height() - 1.0) > 0.12 else 1.0
                ax.text(b.get_x() + b.get_width() / 2, y + 0.03, f"{r:.2f}", ha="center",
                        va="bottom", fontsize=7, color=INK2)
    ax.axhline(1.0, color=INK2, linewidth=0.8)
    style(ax)
    ax.set_xticks(x, [k.replace("_", "\n") for k in KERNELS], fontsize=7, color=INK2)
    ax.set_title(label, fontsize=10, color=INK, loc="left")
    ax.set_ylabel("Rust time / RAJAPerf Base time", fontsize=8, color=INK2)
axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
fig.suptitle(f"Rust relative to RAJAPerf Base_{{CUDA,HIP}}, {ROUND} (1.0 = parity, lower is better)",
             fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out / "speedup.png")

# --- summary table
lines = [f"# rust_perf vs RAJAPerf, {ROUND}", "",
         "Total seconds over all reps; ratio = Rust / RAJAPerf Base (1.0 = parity, lower is better).", ""]
for gpu, label, api in GPUS:
    dd = data[gpu]
    lines += [f"## {label}", "", f"| kernel | Base_{api} | RAJA_{api} | Rust sync | Rust async | async/Base | sync/Base |",
              "|---|---|---|---|---|---|---|"]
    for k in KERNELS:
        b, r = dd["raja"][k]["base"], dd["raja"][k].get("raja", float("nan"))
        s, a = dd["sync"][k], dd["async"][k]
        lines.append(f"| {k} | {b:.5f} | {r:.5f} | {s:.5f} | {a:.5f} | {a / b:.2f} | {s / b:.2f} |")
    lines.append("")
lines += ["## Verification: checksums vs RAJAPerf Base (same init sequence and checksum formula)", "",
          "| kernel | " + " | ".join(f"{label} rel. diff" for _, label, _ in GPUS) + " |", "|---|" + "---|" * len(GPUS)]
for k in KERNELS:
    cells = []
    for gpu, label, api in GPUS:
        dd = data[gpu]
        r = dd["chk_raja"].get(k)
        a, b = dd["chk_sync"].get(k), dd["chk_async"].get(k)
        if r is None or a is None:
            cells.append("n/a")
        else:
            cells.append(f"{abs(a - r) / r:.1e}" + ("" if a == b else " (sync != async!)"))
    lines.append(f"| {k} | " + " | ".join(cells) + " |")
lines += ["", "Rust prints 6 decimals, so ~1e-11 is the comparison floor. LTIMES accumulates into `phi`, so the",
          "default 50 warm-up runs double its checksum; run with `RUST_PERF_WARMUP=0` to compare it.", ""]
(out / "summary.md").write_text("\n".join(lines))
print("\n".join(lines))
