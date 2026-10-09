# rust_perf vs RAJAPerf, 2026-10

Total seconds over all reps; ratio = Rust / RAJAPerf Base (1.0 = parity, lower is better).

## H100

| kernel | Base_CUDA | RAJA_CUDA | Rust sync | Rust async | async/Base | sync/Base |
|---|---|---|---|---|---|---|
| DEL_DOT_VEC_2D | 0.00214 | 0.00206 | 0.00517 | 0.00219 | 1.02 | 2.42 |
| ENERGY | 0.00994 | 0.00980 | 0.01372 | 0.00982 | 0.99 | 1.38 |
| FIR | 0.00128 | 0.00127 | 0.00659 | 0.00185 | 1.44 | 5.14 |
| LTIMES | 0.00626 | 0.00617 | 0.00686 | 0.00540 | 0.86 | 1.10 |
| MATVEC_3D_STENCIL | 0.00621 | 0.00540 | 0.00938 | 0.00590 | 0.95 | 1.51 |
| PRESSURE | 0.01281 | 0.01317 | 0.03098 | 0.01166 | 0.91 | 2.42 |
| VOL3D | 0.00188 | 0.00169 | 0.00480 | 0.00189 | 1.01 | 2.56 |

## MI250X (1 GCD)

| kernel | Base_HIP | RAJA_HIP | Rust sync | Rust async | async/Base | sync/Base |
|---|---|---|---|---|---|---|
| DEL_DOT_VEC_2D | 0.00585 | 0.00561 | 0.00627 | 0.00623 | 1.07 | 1.07 |
| ENERGY | 0.02528 | 0.02489 | 0.03257 | 0.03271 | 1.29 | 1.29 |
| FIR | 0.00624 | 0.00623 | 0.00530 | 0.00540 | 0.87 | 0.85 |
| LTIMES | 0.01376 | 0.01361 | 0.00614 | 0.00624 | 0.45 | 0.45 |
| MATVEC_3D_STENCIL | 0.01717 | 0.01747 | 0.01864 | 0.01805 | 1.05 | 1.09 |
| PRESSURE | 0.03040 | 0.03051 | 0.04385 | 0.04667 | 1.53 | 1.44 |
| VOL3D | 0.00653 | 0.00604 | 0.00522 | 0.00528 | 0.81 | 0.80 |

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
