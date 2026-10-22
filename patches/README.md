# Bench toolchain patches (2026-10)

Apply on `rust-lang/rust` main @ c303e414dd8 (2026-10-22) in a checkout with `llvm.offload = true`,
`clang = true`:

    git am patches/rustc/*.patch
    (cd src/llvm-project && git apply ../../patches/llvm-23.1.1-async-kernel-launch.patch \
                                      ../../patches/llvm-23.1.1-kernarg-reuse.patch)

Rebased 2026-10-22 from a92214c92df (2026-09-23): the four `Region`/`PartitioningStrategy` commits
of rust-lang/rust#158076 merged upstream unchanged on 2026-10-21 and were dropped from the series;
the explicit-data-transfer commits (0002-0004) and the bench-only async launch (0005) have no
upstream PR yet. rust-lang/rust#164455 (2026-10-18) removed `device.bin`: `-Zoffload=Host=` now
takes the device crate's `.bc` from the `--emit=llvm-bc` device pass, and the wrapper reads the
target triple and `target-cpu` attribute from the bitcode (scripts/build.sh step 3). The LLVM
submodule moved to 915c669d232e (release/23.x + 86 backports) without touching `offload/`, so the
LLVM patches are unchanged, but the bump forces a full LLVM rebuild once.

The LLVM patch is llvm/llvm-project#212862 (still open) adapted to the 23.1.1 `interface.cpp`; keep
it uncommitted in the submodule (see the rust2 notes on build stamps) or expect a full LLVM rebuild.
Then `./x build --stage 1 library` and `rustup toolchain link offload <checkout>/build/<host>/stage1`.

`llvm-23.1.1-kernarg-reuse.patch` (amdgpu plugin only): kernel-argument buffers go back to the
plugin's free list as soon as a launch has completed instead of at the next synchronize, and a
stream keeps at most `LIBOMPTARGET_AMDGPU_MAX_INFLIGHT_LAUNCHES` launches in flight (default 128,
0 = unbounded) before a launch waits for the oldest one. Without it every asynchronous launch costs
a fresh `hsa_amd_memory_pool_allocate` + `hsa_amd_agents_allow_access` (~35 us on MI250X, 1405 calls
for the PRESSURE loop), which made async launches no faster than synchronous ones on AMD; with it
the loop needs 133. The cap also keeps the 512-entry AQL queue from filling, so
`LIBOMPTARGET_AMDGPU_HSA_QUEUE_SIZE` is no longer needed; the queue-full spin in `acquirePacket`
hangs sporadically and segfaults inside `librocprofiler-sdk` under `rocprofv3 --kernel-trace`
(pre-existing). Rebuild with `./x build --stage 1 library` (runtime only, ~1.5 min) after applying.

The same patch also sets the AQL **barrier bit** on a dispatch whose only pending dependency is the
preceding kernel of the same HSA queue, instead of pushing a separate barrier-AND packet that waits
on that kernel's completion signal. The barrier packet left the GPU idle ~7-10 us before every
dependent kernel (MI250X PRESSURE: 10 ms of 44; HIP shows 0 us gaps); the barrier bit halves the gap
on gfx1103 (6.5 -> 4.4 us). Streams share the HSA queues round-robin, so the bit also orders after
other streams' preceding packets on that queue - over-synchronization only. Still open: every
dispatch uses system-scope acquire/release fences where HIP uses agent scope between kernels.
