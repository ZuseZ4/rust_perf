#!/usr/bin/env bash
# Three-pass rustc offload build of rust_perf (HostMetadata -> Device -> Host).
#
#   TC=<rustup toolchain>  (default offload)   GPU=gfx90a | sm_90a | gfx1103 ...
#   scripts/build.sh [extra cargo args, e.g. --no-default-features --features energy,f64]
#
# The toolchain must come from a rustc build with llvm.offload=true (and clang=true); the
# nvptx host link additionally needs `ptxas` from a CUDA toolkit on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

TC=${TC:-offload}
GPU=${GPU:-gfx90a}
CRATE=rust_perf
HOST=x86_64-unknown-linux-gnu
MANIFEST=$PWD/offload.manifest
EXTRA=("$@")

case "$GPU" in
  gfx*) TARGET=amdgcn-amd-amdhsa; DEV_FLAGS="-Ctarget-cpu=$GPU" ;;
  sm_*) TARGET=nvptx64-nvidia-cuda; DEV_FLAGS="-Ctarget-cpu=$GPU -Ctarget-feature=+ptx80" ;;
  *) echo "unknown GPU $GPU" >&2; exit 1 ;;
esac

# 1) manifest of the kernel instantiations the host code needs. `cargo rustc -- ...` hands the
#    offload flags to the final crate only: the manifest pass emits no artifacts, so std/libc and
#    the build scripts would otherwise come out empty.
cargo +$TC rustc --bin $CRATE -r --target $HOST -Zbuild-std=std,panic_abort "${EXTRA[@]}" -- \
  -Zunstable-options -Zoffload=HostMetadata=$MANIFEST

# 2) device code; --emit=llvm-bc keeps value names so handle_offload's dyn_ptr guard works
RUSTFLAGS="-Zunstable-options $DEV_FLAGS --emit=llvm-bc -Zoffload=Device=$MANIFEST" \
cargo +$TC build --lib -r --target $TARGET \
  -Zbuild-std=core,compiler_builtins,panic_abort -Zbuild-std-features=compiler-builtins-mem "${EXTRA[@]}"

DEVICE_BIN=$(ls -t $PWD/target/$TARGET/release/build/$CRATE/*/out/device.bin | head -1)
echo "device image: $DEVICE_BIN"

# 3) host code with the device image; rustc links -lomptarget -lomp itself
cargo +$TC rustc --bin $CRATE -r --target $HOST -Zbuild-std=std,panic_abort "${EXTRA[@]}" -- \
  -Zunstable-options -Zoffload=Host=$DEVICE_BIN

ls -la target/$HOST/release/$CRATE
