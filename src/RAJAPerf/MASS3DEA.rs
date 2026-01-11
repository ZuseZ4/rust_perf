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

const MEA_D1D: usize = 4;
const MEA_Q1D: usize = 5;
const NE_VAL: usize = 244;

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

#[cfg(target_os = "linux")]
unsafe fn alloc_array<T>(len: usize) -> *mut T {
    let size = len * core::mem::size_of::<T>();
    let ptr = unsafe { libc::malloc(size) } as *mut T;
    if ptr.is_null() {
        panic!();
    }
    ptr
}

#[cfg(target_os = "linux")]
unsafe fn free_array<T>(ptr: *mut T) {
    unsafe { libc::free(ptr as *mut libc::c_void) };
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
}

#[repr(C)]
#[cfg(target_os = "linux")]
#[derive(Copy, Clone)]
struct Timespec {
    tv_sec: libc::time_t,
    tv_nsec: libc::c_long,
}

#[cfg(target_os = "linux")]
const CLOCK_MONOTONIC: i32 = 1;

#[cfg(target_os = "linux")]
unsafe fn get_time_ns() -> u64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
unsafe fn main() {
    unsafe {
        let B = alloc_array::<f64>(MEA_Q1D * MEA_D1D);
        let D = alloc_array::<f64>(MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_VAL);
        let M =
            alloc_array::<f64>(MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * NE_VAL);

        for i in 0..(MEA_Q1D * MEA_D1D) {
            *B.add(i) = 1.0;
        }
        for i in 0..(MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_VAL) {
            *D.add(i) = 1.0;
        }

        let start = get_time_ns();
        core::intrinsics::offload::<_, _, ()>(
            _mass3dea,
            [NE_VAL as u32, 1, 1],
            [MEA_D1D as u32, MEA_D1D as u32, MEA_D1D as u32],
            (
                B as *const [f64; MEA_Q1D * MEA_D1D],
                D as *const [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_VAL],
                M as *mut [f64; MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * NE_VAL],
                NE_VAL,
            ),
        );
        let end = get_time_ns();

        let duration_s = (end - start) as f64 / 1_000_000_000.0;
        libc::printf(c"%f\n".as_ptr(), duration_s);

        free_array(B);
        free_array(D);
        free_array(M);
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _mass3dea(
        B: *const [f64; MEA_Q1D * MEA_D1D],
        D: *const [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_VAL],
        M: *mut [f64; MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * NE_VAL],
        NE: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _mass3dea(
    B: *const [f64; MEA_Q1D * MEA_D1D],
    D: *const [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_VAL],
    M: *mut [f64; MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * NE_VAL],
    NE: usize,
) {
    unsafe {
        let e = block_idx_x() as usize;
        if e >= NE {
            return;
        }

        let x = thread_idx_x() as usize;
        let y = thread_idx_y() as usize;
        let z = thread_idx_z() as usize;

        let mut s_B: [[f64; MEA_D1D]; MEA_Q1D] = [[0.0; MEA_D1D]; MEA_Q1D];
        if z < 1 && x < MEA_D1D && y < MEA_Q1D {
            s_B[y][x] = (*B)[y + MEA_Q1D * x];
        }

        let mut s_D: [[[f64; MEA_Q1D]; MEA_Q1D]; MEA_Q1D] = [[[0.0; MEA_Q1D]; MEA_Q1D]; MEA_Q1D];
        if x < MEA_Q1D && y < MEA_Q1D && z < MEA_Q1D {
            s_D[x][y][z] =
                (*D)[x + MEA_Q1D * y + MEA_Q1D * MEA_Q1D * z + MEA_Q1D * MEA_Q1D * MEA_Q1D * e];
        }

        _syncthreads();

        if x < MEA_D1D && y < MEA_D1D && z < MEA_D1D {
            let i1 = x;
            let i2 = y;
            let i3 = z;

            for j1 in 0..MEA_D1D {
                for j2 in 0..MEA_D1D {
                    for j3 in 0..MEA_D1D {
                        let mut val: f64 = 0.0;
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
                                                    + MEA_D1D
                                                        * (j2 + MEA_D1D * (j3 + MEA_D1D * e)))));
                        (*M)[idx] = val;
                    }
                }
            }
        }
    }
}
