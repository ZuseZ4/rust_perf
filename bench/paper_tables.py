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


def read_seq(d):
    """kernel -> Base_Seq seconds (3-pass average) if the seq run exists."""
    f = d / "RAJAPerf-timing-Average-seq.csv"
    if not f.exists():
        return {}
    rows = [[c.strip() for c in l.split(",")] for l in f.read_text().splitlines()]
    variants, out = rows[1][1:], {}
    for r in rows[3:]:
        for v, x in zip(variants, r[1:]):
            if v == "Base_Seq" and x != "Not run":
                out[r[0].removeprefix("Apps_")] = float(x)
    return out


KERNEL_ALIASES = {"matvec_3d": "matvec3dstencil"}  # RAJAPerf kernel name -> rust_perf kernel name


def rust_kernel_name(mangled):
    """Last length-prefixed identifier of a Rust v0 symbol: _RNvNtNtC9rust_perf4apps3fir3fir -> fir."""
    i, last = mangled.find("C") + 1, None
    while i < len(mangled):
        m = re.match(r"\d+", mangled[i:])
        if not m:
            break
        n = int(m.group()); i += len(m.group())
        last = mangled[i:i + n]; i += n
    return last


def canon(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def read_kernel_stats(f):
    """kernel -> average ns, from rocprofv3 *_kernel_stats.csv or nsys cuda_gpu_kern_sum csv."""
    import csv
    if not f.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        name = r.get("Name") or r.get("name") or ""
        avg_ns = r.get("AverageNs") or r.get("Avg (ns)") or r.get("Avg(ns)")
        if not avg_ns or "daxpy" in name or "rocclr" in name:
            continue
        if name.startswith("_R"):
            short = rust_kernel_name(name)
        else:
            m = re.search(r"apps::(\w+)", name)
            short = m.group(1) if m else name
            short = KERNEL_ALIASES.get(short, short)
        out[canon(short)] = float(avg_ns)
    return out


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
    seq = read_seq(d)
    lines.append(r"\pgfplotstableread[row sep=newline, col sep=space]{")
    cols = (["BaseSeq"] if seq else []) + [f"Base{api.capitalize()}", f"Raja{api.capitalize()}"]
    lines.append(" ".join(cols))
    for k in KERNELS:
        row = ([f"{seq[k]:.6f}"] if seq else []) + [f"{raja[k]['base']:.6f}", f"{raja[k]['raja']:.6f}"]
        lines.append("    ".join(row))
    row = ([f"{avg([seq[k] for k in KERNELS]):.6f}"] if seq else []) + [
        f"{avg([raja[k]['base'] for k in KERNELS]):.6f}", f"{avg([raja[k]['raja'] for k in KERNELS]):.6f}"]
    lines.append("    ".join(row))
    lines.append(r"}\rajadata" + machine)
    lines.append("%last row is avg; LTIMES is the fastest RAJA tuning (launch_256); RAJAPerf run with --ltimes_num_d 64"
                 + ("; BaseSeq is the 3-pass average" if seq else "; no Base_Seq run yet"))
    lines.append("")
    # per-kernel GPU times (paper Fig. 2)
    kr = read_kernel_stats(d / "analysis" / "kernels" / "rust.csv")
    kj = read_kernel_stats(d / "analysis" / "kernels" / "raja.csv")
    if kr and kj:
        common = [k for k in sorted(kj, key=lambda k: -kj[k]) if k in kr]
        tag = machine.capitalize()
        lines.append(r"\pgfplotstableread[row sep=newline, col sep=comma]{")
        lines.append("Benchmark,RajaNs,RustNs,Speedup")
        for k in common:
            lines.append(f"{k},{kj[k]:.1f},{kr[k]:.1f},{kj[k] / kr[k]:.4f}")
        lines.append(f"avg,{avg([kj[k] for k in common]):.1f},{avg([kr[k] for k in common]):.1f},{avg([kj[k] / kr[k] for k in common]):.4f}")
        lines.append(r"}\profilingKernelTime" + tag)
        lines.append("% average GPU time per kernel launch (ns), rocprofv3/nsys kernel stats; Speedup = Raja/Rust (>1: Rust faster)")
        lines.append("")
text = "\n".join(lines) + "\n"
(BENCH / ROUND / "paper-tables.tex").write_text(text)
print(text)
