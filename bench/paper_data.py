#!/usr/bin/env python3
"""Regenerate the paper's pgfplots data file in its original structure from one benchmark round.

Usage: bench/paper_data.py <round>   -> bench/<round>/paper-data.tex
Macro names, column headers, separators and row order follow the paper's file. The
`RustOffloadNaive` column (interface A, transfer per launch) and the A2000 table cannot be
reproduced with this toolchain and are carried over verbatim.
"""
import csv
import re
import sys
from pathlib import Path

ROUND = sys.argv[1] if len(sys.argv) > 1 else "2026-10"
BENCH = Path(__file__).resolve().parent
D_AMD, D_NV = BENCH / "amd" / ROUND, BENCH / "nvidia" / ROUND
APPS = ["DEL_DOT_VEC_2D", "ENERGY", "FIR", "LTIMES", "MATVEC_3D_STENCIL", "PRESSURE", "VOL3D"]
# kernel order of the paper's derived tables
KERNEL_ORDER = ["pressurecalc2", "matvec3d", "pressurecalc1", "ltimes", "energycalc5", "energycalc3",
                "deldotvec2d", "energycalc1", "vol3d", "fir", "energycalc4", "energycalc2", "energycalc6"]
NAIVE_TIOGA = {"DEL_DOT_VEC_2D": 0.333227, "ENERGY": 0.217053, "FIR": 2.456091, "LTIMES": 0.053783,
               "MATVEC_3D_STENCIL": 0.969724, "PRESSURE": 2.261837, "VOL3D": 0.267186}
UOFT = r"""\pgfplotstableread[row sep=newline, col sep=comma]{
Benchmark,RustNs,RustAlgebraicNs,BaseCUDANs,RAJACUDANs,Speedup
pressurecalc2,124869.0,124772.1,125322.8,124372.0,
matvec3d,740781.9,671459.0,643838.0,566294.3,
pressurecalc1,64543.6,64283.5,63612.3,64082.2,
ltimes,786712.3,570984.6,397597.8,406900.8,
energycalc5,261171.1,248742.2,251317.5,251285.5,
energycalc3,218042.0,217393.4,217404.1,217338.9,
deldotvec2d,1252695.8,1028520.5,1110918.8,1023101.3,
energycalc1,181135.8,180655.3,180708.5,180684.4,
vol3d,959662.5,701820.0,737184.2,758056.1,
fir,387479.9,173078.7,196399.9,197350.6,
energycalc4,93215.3,93450.6,93465.3,93470.9,
energycalc2,62116.0,62170.9,62249.6,62424.1,
energycalc6,34537.7,34789.3,34509.5,34568.4,
avg,397458.6846,320932.3154,316502.1769,306148.4231,
}\profilingKernelTimeUofT"""


def avg(v):
    v = list(v)
    return sum(v) / len(v)


def read_rust(f):
    out = {}
    for l in f.read_text().splitlines():
        m = re.match(r"^(\w+)\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", l)
        if m:
            out[m.group(1)] = float(m.group(4))
    return out


def newest_async(d):
    return next(f for f in (d / "rust-async-fir-rt.txt", d / "rust-async-barrier.txt", d / "rust-async.txt") if f.exists())


def read_raja_table(f, api):
    rows = [[c.strip() for c in l.split(",")] for l in f.read_text().splitlines()]
    variants, out = rows[1][1:], {}
    for r in rows[3:]:
        k, vals = r[0].removeprefix("Apps_"), {}
        for v, x in zip(variants, r[1:]):
            if x == "Not run":
                continue
            key = {f"Base_{api}": "base", f"RAJA_{api}": "raja", "Base_Seq": "seq"}.get(v)
            if key and (key not in vals or float(x) < vals[key]):
                vals[key] = float(x)
        out[k] = vals
    return out


def raja(d, api):
    out = read_raja_table(d / "p10" / "RAJAPerf-timing-Minimum.csv", api)
    seq = read_raja_table(d / "RAJAPerf-timing-Average-seq.csv", api)
    for k in out:
        out[k]["seq"] = seq[k]["seq"]
    return out


def rust_kernel_name(mangled):
    i, last = mangled.find("C") + 1, None
    while i < len(mangled):
        m = re.match(r"\d+", mangled[i:])
        if not m:
            break
        n = int(m.group()); i += len(m.group()); last = mangled[i:i + n]; i += n
    return last


def short_name(name):
    """Paper's kernel names: matvec3d, deldotvec2d, pressurecalc1, ..."""
    s = rust_kernel_name(name) if name.startswith("_R") else re.search(r"apps::(\w+)", name).group(1)
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return {"matvec3dstencil": "matvec3d", "matvec3d": "matvec3d", "pressurecalc1": "pressurecalc1",
            "pressurecalc2": "pressurecalc2", "deldotvec2d": "deldotvec2d"}.get(s, s)


def nsys_rows(f):
    rows = []
    for r in csv.DictReader(open(f)):
        if "daxpy" in r["Name"] or "rocclr" in r["Name"]:
            continue
        rows.append(dict(total=float(r["Total Time (ns)"]), n=int(r["Instances"]), avg=float(r["Avg (ns)"]),
                         med=float(r["Med (ns)"]), mn=float(r["Min (ns)"]), mx=float(r["Max (ns)"]),
                         sd=float(r["StdDev (ns)"]), name=short_name(r["Name"])))
    return sorted(rows, key=lambda r: -r["total"])


def rocprof_rows(f):
    rows = []
    for r in csv.DictReader(open(f)):
        if "daxpy" in r["Name"] or "rocclr" in r["Name"]:
            continue
        rows.append(dict(calls=int(r["Calls"]), total=float(r["TotalDurationNs"]), avg=float(r["AverageNs"]),
                         pct=float(r["Percentage"]), name=short_name(r["Name"])))
    return sorted(rows, key=lambda r: -r["total"])


out = []
P = out.append

# ---- runtime tables
rs_t, ra_t = read_rust(D_AMD / "rust-sync.txt"), read_rust(newest_async(D_AMD))
P(f"% MI250X, LLVM 23 bench toolchain {ROUND}: RustOffload = async launches with the libomptarget kernarg-reuse and")
P("% barrier-bit patches; RustOffloadNaive carried over from the")
P("% earlier round (interface A, transfer per launch; not reproducible with the current toolchain).")
P(r"\pgfplotstableread[row sep=newline, col sep=space]{")
P("App RustOffloadNaive RustOffload")
for k in APPS:
    P(f"{k:24s} {NAIVE_TIOGA[k]:.6f}  {ra_t[k]:.6f}")
P(f"{'avg':24s} {avg(NAIVE_TIOGA.values()):.6f}  {avg(ra_t[k] for k in APPS):.6f}")
P(r"}\rustdatatioga")
P("")
rj_t = raja(D_AMD, "HIP")
P("% RAJAPerf 2026-10 build (ROCm 7.2.1), --ltimes_num_d 64; BaseHip/RajaHip = minimum of 10 passes, BaseSeq = 3-pass average")
P(r"\pgfplotstableread[row sep=newline, col sep=space]{")
P("BaseSeq BaseHip RajaHip")
for k in APPS:
    P(f"{rj_t[k]['seq']:.6f}    {rj_t[k]['base']:.6f}   {rj_t[k]['raja']:.6f}")
P(f"{avg(rj_t[k]['seq'] for k in APPS):.6f}    {avg(rj_t[k]['base'] for k in APPS):.6f}   {avg(rj_t[k]['raja'] for k in APPS):.6f}")
P(r"}\rajadatatioga")
P("%last row is avg")
P("")
rs_m, ra_m = read_rust(D_NV / "rust-sync.txt"), read_rust(newest_async(D_NV))
P(f"% H100, LLVM 23 bench toolchain {ROUND}: RustOffload = async launches")
P(r"\pgfplotstableread[row sep=newline, col sep=space]{")
P("App RustOffload")
for k in APPS:
    P(f"{k:24s} {ra_m[k]:.6f}")
P(f"{'avg':24s} {avg(ra_m[k] for k in APPS):.6f}")
P(r"}\rustdatamatrix")
P("")
rj_m = raja(D_NV, "CUDA")
P("% RAJAPerf 2026-10 build (CUDA 12.9), --ltimes_num_d 64; BaseCUDA/RajaCUDA = minimum of 10 passes, BaseSeq = 3-pass average")
P(r"\pgfplotstableread[row sep=newline, col sep=comma]{")
P("BaseSeq,BaseCUDA,RajaCUDA ")
for k in APPS:
    P(f"{rj_m[k]['seq']:.6f}  ,  {rj_m[k]['base']:.6f} ,  {rj_m[k]['raja']:.6f}")
P(f"{avg(rj_m[k]['seq'] for k in APPS):.6f}  ,  {avg(rj_m[k]['base'] for k in APPS):.6f} ,  {avg(rj_m[k]['raja'] for k in APPS):.6f}")
P(r"}\rajadatamatrix")
P("% last row is avg")
P("")
P("")
P("")
# ---- kernel times
P("% Kernel Tiems start")
kdir_nv, kdir_amd = D_NV / "analysis" / "kernels", D_AMD / "analysis" / "kernels"
for label, f in (("Rust", next(kdir_nv.glob("rust*.csv"))), ("Raja", next(kdir_nv.glob("raja*.csv")))):
    P(r"\pgfplotstableread[row sep=newline, col sep=space]{")
    P("Total-Time(ns)  Instances  Avg(ns)  Med(ns)   Min(ns)  Max(ns)  StdDev(ns)       Name")
    for r in nsys_rows(f):
        P(f"{r['total']:,.0f}  {r['n']:6d}  {r['avg']:,.1f}  {r['med']:,.1f}  {r['mn']:,.0f}  {r['mx']:,.0f}  {r['sd']:,.1f}  {r['name']}")
    P(r"}\profiling" + label + "Matrix")
    P("")
for label, f, quote in (("Rust", next(kdir_amd.glob("rust*.csv")), '"'), ("Raja", next(kdir_amd.glob("raja*.csv")), "")):
    rows = rocprof_rows(f)
    P(r"\pgfplotstableread[row sep=newline, col sep=comma]{")
    P('"Name","Calls","TotalDurationNs","AverageNs","Percentage"')
    for r in rows:
        P(f"{quote}{r['name']}{quote},{r['calls']},{r['total']:.0f},{r['avg']:.0f},{r['pct']}")
    P(f"{quote}avg{quote},{avg(r['calls'] for r in rows)},{avg(r['total'] for r in rows)},{avg(r['avg'] for r in rows)},{avg(r['pct'] for r in rows)}")
    P(r"}\profiling" + label + "Tioga")
    P("")
P("")
P("%Total Kernel times start")
for macro, kdir, reader in (("profilingKernelTimeMatrix", kdir_nv, nsys_rows), ("profilingTioga", kdir_amd, rocprof_rows)):
    kr = {r["name"]: r["avg"] for r in reader(next(kdir.glob("rust*.csv")))}
    kj = {r["name"]: r["avg"] for r in reader(next(kdir.glob("raja*.csv")))}
    P(r"\pgfplotstableread[row sep=newline, col sep=comma]{")
    P("Benchmark,RajaNs,RustNs,Speedup")
    for k in KERNEL_ORDER:
        P(f"{k},{kj[k]:.1f},{kr[k]:.1f},{kj[k] / kr[k]}")
    P(f"avg,{avg(kj[k] for k in KERNEL_ORDER):.4f},{avg(kr[k] for k in KERNEL_ORDER):.4f},{avg(kj[k] / kr[k] for k in KERNEL_ORDER):.10f}")
    P("}\\" + macro)
    P("")
P("% RTX A2000 (UofT): not re-measured in this round, carried over")
P(UOFT)
text = "\n".join(out) + "\n"
(BENCH / ROUND / "paper-data.tex").write_text(text)
print(text)
