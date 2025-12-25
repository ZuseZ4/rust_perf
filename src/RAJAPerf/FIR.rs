#![allow(internal_features)]
#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(link_llvm_intrinsics)]
#![feature(core_intrinsics)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::_block_dim_x as block_dim_x;
#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::_block_idx_x as block_idx_x;
#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::_thread_idx_x as thread_idx_x;

#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn thread_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workgroup.size.x"]
    fn block_dim_x() -> i32;
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
fn main() {
    let m_in: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut m_out: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    let coeff = [
        3.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, 3.0,
    ];
    let coefflen = 16;
    let iend: usize = 4;

    unsafe {
        fir(
            m_out.as_mut_ptr(),
            m_in.as_ptr(),
            coeff.as_ptr(),
            coefflen,
            iend,
        );
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn fir(m_out: *mut f64, m_in: *const f64, coeff: *const f64, coefflen: usize, iend: usize) {
    core::intrinsics::offload(_fir, (m_in, m_out, coeff, coefflen, iend))
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _fir(m_out: *mut f64, m_in: *const f64, coeff: *const f64, coefflen: usize, iend: usize);
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _fir(
    m_out: *mut f64,
    m_in: *const f64,
    coeff: *const f64,
    coefflen: usize,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

        if i < iend {
            for i in 0..iend {
                let mut sum = 0.0;
                for j in 0..coefflen {
                    sum += *coeff.add(j) * *m_in.add(i + j);
                }
                *m_out.add(i) = sum;
            }
        }
    }
}
