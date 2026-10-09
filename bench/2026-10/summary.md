# rust_perf vs RAJAPerf, 2026-10

Total seconds over all reps; ratio = Rust / RAJAPerf Base (1.0 = parity, lower is better).
RAJAPerf: minimum over 10 passes (`p10/`) when present, else the 3-pass average. Rust: one run after 50 warm-up iterations.

## H100

| kernel | Base_CUDA | RAJA_CUDA | Rust sync | Rust async | async/Base | sync/Base |
|---|---|---|---|---|---|---|
| DEL_DOT_VEC_2D | 0.00205 | 0.00205 | 0.00517 | 0.00218 | 1.06 | 2.52 |
| ENERGY | 0.00977 | 0.00977 | 0.01372 | 0.00991 | 1.01 | 1.40 |
| FIR | 0.00126 | 0.00125 | 0.00659 | 0.00187 | 1.48 | 5.22 |
| LTIMES | 0.00610 | 0.00611 | 0.00686 | 0.00539 | 0.88 | 1.13 |
| MATVEC_3D_STENCIL | 0.00611 | 0.00539 | 0.00938 | 0.00604 | 0.99 | 1.54 |
| PRESSURE | 0.01264 | 0.01307 | 0.03098 | 0.01195 | 0.95 | 2.45 |
| VOL3D | 0.00178 | 0.00168 | 0.00480 | 0.00190 | 1.07 | 2.70 |

## MI250X (1 GCD)

| kernel | Base_HIP | RAJA_HIP | Rust sync | Rust async | async/Base | sync/Base |
|---|---|---|---|---|---|---|
| DEL_DOT_VEC_2D | 0.00558 | 0.00559 | 0.00627 | 0.00569 | 1.02 | 1.12 |
| ENERGY | 0.02482 | 0.02487 | 0.03257 | 0.02859 | 1.15 | 1.31 |
| FIR | 0.00623 | 0.00622 | 0.00530 | 0.00446 | 0.72 | 0.85 |
| LTIMES | 0.01356 | 0.01359 | 0.00614 | 0.00552 | 0.41 | 0.45 |
| MATVEC_3D_STENCIL | 0.01700 | 0.01764 | 0.01864 | 0.01752 | 1.03 | 1.10 |
| PRESSURE | 0.03014 | 0.03046 | 0.04385 | 0.03687 | 1.22 | 1.45 |
| VOL3D | 0.00621 | 0.00603 | 0.00522 | 0.00479 | 0.77 | 0.84 |

## Verification: checksums vs RAJAPerf Base (same init sequence and checksum formula)

| kernel | H100 rel. diff | MI250X (1 GCD) rel. diff |
|---|---|---|
| DEL_DOT_VEC_2D | 7.6e-12 | 7.6e-12 |
| ENERGY | 4.8e-12 | 4.8e-12 |
| FIR | 2.7e-13 | 2.7e-13 |
| LTIMES | 1.0e+00 | 1.0e+00 |
| MATVEC_3D_STENCIL | 8.2e-13 | 8.2e-13 |
| PRESSURE | 4.8e-12 | 4.8e-12 |
| VOL3D | 1.6e-11 | 1.6e-11 |

Rust prints 6 decimals, so ~1e-11 is the comparison floor. LTIMES accumulates into `phi`, so the
default 50 warm-up runs double its checksum; run with `RUST_PERF_WARMUP=0` to compare it.
