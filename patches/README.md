# Bench toolchain patches (2026-10)

Apply on `rust-lang/rust` main @ a92214c92df (2026-09-23) in a checkout with `llvm.offload = true`,
`clang = true`:

    git am patches/rustc/*.patch
    (cd src/llvm-project && git apply ../../patches/llvm-23.1.1-async-kernel-launch.patch \
                                      ../../patches/llvm-23.1.1-kernarg-reuse.patch)

The LLVM patch is llvm/llvm-project#212862 adapted to the 23.1.1 `interface.cpp`; keep it
uncommitted in the submodule (see the rust2 notes on build stamps) or expect a full LLVM rebuild.
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
