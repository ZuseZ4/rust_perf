#!/usr/bin/env bash
# Instruction-mix per kernel from an AMDGPU code object (ELF): loads/stores by width, FMAs, branches,
# waitcnts, instruction count, plus the kernel descriptor's VGPR/SGPR usage.
#   scripts/isa-stats.sh <code object .elf/.co/.so> [kernel-name-filter]
# For raja-perf.exe first extract the gfx90a object: llvm-objdump --offloading bin/raja-perf.exe
set -euo pipefail
OBJ=$1; FILTER=${2:-}
LLVM=${LLVM:-$(dirname "$(dirname "$(rustc +${TC:-offload} --print sysroot)")")/llvm/bin}
[ -x "$LLVM/llvm-objdump" ] || LLVM=/opt/rocm/llvm/bin
"$LLVM/llvm-objdump" -d --no-show-raw-insn "$OBJ" | awk -v filter="$FILTER" '
  /^[0-9a-f]+ <.*>:$/ { if (name != "") flush(); name = $2; gsub(/[<>:]/, "", name); n=0; ld2=0; ld4=0; ld1=0; st=0; fma=0; mul=0; add=0; br=0; wait=0; scr=0; next }
  name != "" && NF > 1 && $1 ~ /^[a-z]/ {
    n++
    if ($1 ~ /^global_load_dwordx4|^global_load_b128/) ld4++
    else if ($1 ~ /^global_load_dwordx2|^global_load_b64/) ld2++
    else if ($1 ~ /^global_load/) ld1++
    if ($1 ~ /^global_store/) st++
    if ($1 ~ /^v_fma_f64|^v_fmac_f64/) fma++
    if ($1 ~ /^v_mul_f64/) mul++
    if ($1 ~ /^v_add_f64/) add++
    if ($1 ~ /^s_cbranch/) br++
    if ($1 ~ /^s_waitcnt|^s_wait_/) wait++
    if ($1 ~ /^scratch_|^buffer_/) scr++
  }
  function flush() { if (filter == "" || index(name, filter)) printf "%-60s insn %4d  ld x4/x2/x1 %3d/%3d/%3d  st %3d  fma %3d mul %3d add %3d  br %2d  wait %3d  scratch %d\n", substr(name,1,60), n, ld4, ld2, ld1, st, fma, mul, add, br, wait, scr }
  END { if (name != "") flush() }'
echo "--- kernel descriptors (vgpr/sgpr/kernarg)"
"$LLVM/llvm-readelf" --notes "$OBJ" 2>/dev/null | grep -E '\.name:|\.vgpr_count|\.sgpr_count|\.kernarg_segment_size|\.spill' | paste - - - - 2>/dev/null | sed 's/ \+/ /g' | head -40
