#![allow(internal_features)]
#![allow(non_snake_case)]
#![feature(abi_gpu_kernel)]
#![feature(link_llvm_intrinsics)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![feature(asm_experimental_arch)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_idx_x as block_idx_x, _syncthreads, _thread_idx_x as thread_idx_x,
    _thread_idx_y as thread_idx_y, _thread_idx_z as thread_idx_z,
};

#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn thread_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.y"]
    fn thread_idx_y() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.z"]
    fn thread_idx_z() -> i32;
    #[link_name = "llvm.amdgcn.s.barrier"]
    fn _syncthreads();
}

const MEA_D1D: usize = 5;
const MEA_Q1D: usize = 10;

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
unsafe fn main() {
    const NE: usize = 2;

    let B: [f64; MEA_Q1D * MEA_D1D] = [1.0; MEA_Q1D * MEA_D1D];

    let D: [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE] = [1.0; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE];

    let mut M: [f64; MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * NE] =
        [0.0; MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * NE];

    mass3dea(B.as_ptr(), D.as_ptr(), M.as_mut_ptr(), NE);
}

#[inline(never)]
fn mass3dea(B: *const f64, D: *const f64, M: *mut f64, NE: usize) {
    core::intrinsics::offload(
        _mass3dea,
        [NE as u32, 1, 1],
        [MEA_D1D as u32, MEA_D1D as u32, MEA_D1D as u32],
        (B, D, M, NE),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _mass3dea(B: *const f64, D: *const f64, M: *mut f64, NE: usize);
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _mass3dea(B: *const f64, D: *const f64, M: *mut f64, NE: usize) {
    let e = block_idx_x() as usize;
    if e >= NE {
        return;
    }

    let x = thread_idx_x() as usize;
    let y = thread_idx_y() as usize;
    let z = thread_idx_z() as usize;

    let mut s_B: [[f64; MEA_D1D]; MEA_Q1D] = [[0.0; MEA_D1D]; MEA_Q1D];

    if z < 1 && x < MEA_D1D && y < MEA_Q1D {
        s_B[y][x] = *B.add(y + MEA_Q1D * x);
    }

    let mut s_D: [[[f64; MEA_Q1D]; MEA_Q1D]; MEA_Q1D] = [[[0.0; MEA_Q1D]; MEA_Q1D]; MEA_Q1D];

    if x < MEA_Q1D && y < MEA_Q1D && z < MEA_Q1D {
        s_D[x][y][z] =
            *D.add(x + MEA_Q1D * y + MEA_Q1D * MEA_Q1D * z + MEA_Q1D * MEA_Q1D * MEA_Q1D * e);
    }

    _syncthreads();

    if x < MEA_D1D && y < MEA_D1D && z < MEA_D1D {
        let i1 = x;
        let i2 = y;
        let i3 = z;

        for j1 in 0..MEA_D1D {
            for j2 in 0..MEA_D1D {
                for j3 in 0..MEA_D1D {
                    let mut val = 0.0f64;

                    for k1 in 0..MEA_Q1D {
                        for k2 in 0..MEA_Q1D {
                            for k3 in 0..MEA_Q1D {
                                val += s_B[k1][i1]
                                    * s_B[k1][j1]
                                    * s_B[k2][i2]
                                    * s_B[k2][j2]
                                    * s_B[k3][i3]
                                    * s_B[k3][j3]
                                    * s_D[k1][k2][k3];
                            }
                        }
                    }

                    let idx = i1
                        + MEA_D1D
                            * (i2
                                + MEA_D1D
                                    * (i3
                                        + MEA_D1D
                                            * (j1
                                                + MEA_D1D * (j2 + MEA_D1D * (j3 + MEA_D1D * e)))));

                    *M.add(idx) = val;
                }
            }
        }
    }
}
