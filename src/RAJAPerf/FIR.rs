#![allow(internal_features)]
#![feature(abi_gpu_kernel)]
#![feature(link_llvm_intrinsics)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

const IEND: usize = 1_000_000;
const COEFFLEN: usize = 16;
const THREADS_PER_BLOCK: u32 = 32;
const BLOCKS: u32 = (IEND as u32).div_ceil(THREADS_PER_BLOCK);

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};

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

use core::ptr;

#[cfg(target_os = "linux")]
unsafe fn alloc_array<T>(len: usize) -> *mut T {
    let size = len * core::mem::size_of::<T>();
    let ptr = unsafe { libc::malloc(size) } as *mut T;
    if ptr.is_null() {
        panic!();
    }
    unsafe { ptr::write_bytes(ptr, 0, len) };
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
        let m_in = alloc_array::<f64>(IEND + COEFFLEN);
        let m_out = alloc_array::<f64>(IEND);
        let coeff = [
            3.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0, -1.0,
            3.0,
        ];

        *m_in.add(0) = 1.0;
        *m_in.add(1) = 2.0;
        *m_in.add(2) = 3.0;
        *m_in.add(3) = 4.0;

        let start = get_time_ns();
        core::intrinsics::offload::<_, _, ()>(
            _fir,
            [BLOCKS, 1, 1],
            [THREADS_PER_BLOCK, 1, 1],
            (
                m_out as *mut [f64; IEND],
                m_in as *const [f64; IEND + COEFFLEN],
                &coeff as *const [f64; COEFFLEN],
                IEND,
            ),
        );
        let end = get_time_ns();

        let duration_s = (end - start) as f64 / 1_000_000_000.0;
        libc::printf(c"%f\n".as_ptr(), duration_s);

        free_array(m_in);
        free_array(m_out);
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _fir(
        m_out: *mut [f64; IEND],
        m_in: *const [f64; IEND + COEFFLEN],
        coeff: *const [f64; COEFFLEN],
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _fir(
    m_out: *mut [f64; IEND],
    m_in: *const [f64; IEND + COEFFLEN],
    coeff: *const [f64; COEFFLEN],
    iend: usize,
) {
    let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

    if i < iend {
        let mut sum: f64 = 0.0;
        let mut j = 0;
        while j < COEFFLEN {
            sum += (*coeff)[j] * (*m_in)[i + j];
            j += 1;
        }
        (*m_out)[i] = sum;
    }
}
