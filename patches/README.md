# Bench toolchain patches (2026-10)

Apply on `rust-lang/rust` main @ a92214c92df (2026-09-23) in a checkout with `llvm.offload = true`,
`clang = true`:

    git am patches/rustc/*.patch
    (cd src/llvm-project && git apply ../../patches/llvm-23.1.1-async-kernel-launch.patch)

The LLVM patch is llvm/llvm-project#212862 adapted to the 23.1.1 `interface.cpp`; keep it
uncommitted in the submodule (see the rust2 notes on build stamps) or expect a full LLVM rebuild.
Then `./x build --stage 1 library` and `rustup toolchain link offload <checkout>/build/<host>/stage1`.
