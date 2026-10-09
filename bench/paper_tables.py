#!/usr/bin/env python3
"""Emit the pgfplots tables of the paper (\\rustdata*, \\rajadata*) from one benchmark round.

Usage: bench/paper_tables.py <round>  -> prints LaTeX and writes bench/<round>/paper-tables.tex
RAJAPerf: minimum over 10 passes (p10/); Rust: printed tables, async (barrier-bit run on AMD when
present) and sync.
"""
import re
import sys
from pathlib import Path

ROUND = sys.argv[1] if len(sys.argv) > 1 else "2026-10"
BENCH = Path(__file__).resolve().parent
KERNELS = ["DEL_DOT_VEC_2D", "ENERGY", "FIR", "LTIMES", "MATVEC_3D_STENCIL", "PRESSURE", "VOL3D"]
GPUS = [("amd", "tioga", "HIP"), ("nvidia", "matrix", "CUDA")]


def read_raja(d, api):
    f = d / "p10" / "RAJAPerf-timing-Minimum.csv"
    rows = [[c.strip() for c in l.split(",")] for l in f.read_text().splitlines()]
    variants = rows[1][1:]
    out = {}
    for r in rows[3:]:
        k = r[0].removeprefix("Apps_")
        vals = {}
        for v, x in zip(variants, r[1:]):
            if x == "Not run":
                continue
            key = "base" if v == f"Base_{api}" else "raja" if v == f"RAJA_{api}" else None
            if key and (key not in vals or float(x) < vals[key]):
                vals[key] = float(x)
        out[k] = vals
    return out


def read_rust(f):
    out = {}
    for l in f.read_text().splitlines():
        m = re.match(r"^(\w+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", l)
        if m:
            out[m.group(1)] = float(m.group(4))
    return out


def avg(vals):
    return sum(vals) / len(vals)


lines = []
for gpu, machine, api in GPUS:
    d = BENCH / gpu / ROUND
    raja = read_raja(d, api)
    async_f = d / "rust-async-barrier.txt"
    if not async_f.exists():
        async_f = d / "rust-async.txt"
    ra, rs = read_rust(async_f), read_rust(d / "rust-sync.txt")
    lines.append(f"% {machine}: RAJAPerf min of 10 passes; Rust async from {async_f.name}, sync from rust-sync.txt")
    lines.append(r"\pgfplotstableread[row sep=newline, col sep=space]{")
    lines.append("App RustOffloadSync RustOffload")
    for k in KERNELS:
        lines.append(f"{k:24s} {rs[k]:.6f}  {ra[k]:.6f}")
    lines.append(f"{'avg':24s} {avg([rs[k] for k in KERNELS]):.6f}  {avg([ra[k] for k in KERNELS]):.6f}")
    lines.append(r"}\rustdata" + machine)
    lines.append(f"% rust (async) slowdown over Base{api.capitalize()}")
    for k in KERNELS:
        lines.append(f"% {k:24s} {ra[k] / raja[k]['base']:.3f}")
    lines.append("")
    lines.append(r"\pgfplotstableread[row sep=newline, col sep=space]{")
    lines.append(f"Base{api.capitalize()} Raja{api.capitalize()}")
    for k in KERNELS:
        lines.append(f"{raja[k]['base']:.6f}    {raja[k]['raja']:.6f}")
    lines.append(f"{avg([raja[k]['base'] for k in KERNELS]):.6f}    {avg([raja[k]['raja'] for k in KERNELS]):.6f}")
    lines.append(r"}\rajadata" + machine)
    lines.append("%last row is avg; LTIMES is the fastest RAJA tuning (launch_256); RAJAPerf run with --ltimes_num_d 64")
    lines.append("")
text = "\n".join(lines) + "\n"
(BENCH / ROUND / "paper-tables.tex").write_text(text)
print(text)
