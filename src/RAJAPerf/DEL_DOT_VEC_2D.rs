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
const HALF: f64 = 1.0;
const PTINY: f64 = 0.001;
const THREADS_PER_BLOCK: u32 = 256;
const BLOCKS: u32 = (IEND as u32).div_ceil(THREADS_PER_BLOCK);

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};
use core::ptr;

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
unsafe fn alloc_array<T>(len: usize) -> *mut T {
    let size = len * core::mem::size_of::<T>();
    let ptr = unsafe { libc::malloc(size) } as *mut T;
    if ptr.is_null() {
        panic!("libc malloc failed!");
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
#[derive(Debug, Copy, Clone)]
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
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
unsafe fn main() {
    unsafe {
        let div = alloc_array::<f64>(IEND);

        let mut x = [ptr::null_mut(); 4];
        let mut y = [ptr::null_mut(); 4];
        let mut fx = [ptr::null_mut(); 4];
        let mut fy = [ptr::null_mut(); 4];

        for i in 0..4 {
            x[i] = alloc_array::<f64>(IEND);
            y[i] = alloc_array::<f64>(IEND);
            fx[i] = alloc_array::<f64>(IEND);
            fy[i] = alloc_array::<f64>(IEND);
        }

        let real_zones = alloc_array::<usize>(IEND);

        let start = get_time_ns();
        core::intrinsics::offload::<_, _, ()>(
            _del_dot_vec_2d,
            [BLOCKS, 1, 1],
            [THREADS_PER_BLOCK, 1, 1],
            (
                div, x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], fx[0], fx[1], fx[2], fx[3],
                fy[0], fy[1], fy[2], fy[3], real_zones, HALF, PTINY, IEND,
            ),
        );
        let end = get_time_ns();
        let duration_ns = end - start;
        let duration_s = duration_ns as f64 / 1_000_000_000.0;

        libc::printf(c"%f\n".as_ptr(), duration_s);

        free_array(div);
        for i in 0..4 {
            free_array(x[i]);
            free_array(y[i]);
            free_array(fx[i]);
            free_array(fy[i]);
        }
        free_array(real_zones);
    }
}

unsafe fn del_dot_vec_2d(
    div: *mut [f64; IEND],
    x1: *const [f64; IEND],
    x2: *const [f64; IEND],
    x3: *const [f64; IEND],
    x4: *const [f64; IEND],
    y1: *const [f64; IEND],
    y2: *const [f64; IEND],
    y3: *const [f64; IEND],
    y4: *const [f64; IEND],
    fx1: *const [f64; IEND],
    fx2: *const [f64; IEND],
    fx3: *const [f64; IEND],
    fx4: *const [f64; IEND],
    fy1: *const [f64; IEND],
    fy2: *const [f64; IEND],
    fy3: *const [f64; IEND],
    fy4: *const [f64; IEND],
    real_zones: *const [usize; IEND],
    half: f64,
    ptiny: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _del_dot_vec_2d,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (
            div, x1, x2, x3, x4, y1, y2, y3, y4, fx1, fx2, fx3, fx4, fy1, fy2, fy3, fy4,
            real_zones, half, ptiny, iend,
        ),
    )
}
#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _del_dot_vec_2d(
        div: *mut [f64; IEND],
        x1: *const [f64; IEND],
        x2: *const [f64; IEND],
        x3: *const [f64; IEND],
        x4: *const [f64; IEND],
        y1: *const [f64; IEND],
        y2: *const [f64; IEND],
        y3: *const [f64; IEND],
        y4: *const [f64; IEND],
        fx1: *const [f64; IEND],
        fx2: *const [f64; IEND],
        fx3: *const [f64; IEND],
        fx4: *const [f64; IEND],
        fy1: *const [f64; IEND],
        fy2: *const [f64; IEND],
        fy3: *const [f64; IEND],
        fy4: *const [f64; IEND],
        real_zones: *const [usize; IEND],
        half: f64,
        ptiny: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _del_dot_vec_2d(
    div: *mut [f64; IEND],
    x1: *const [f64; IEND],
    x2: *const [f64; IEND],
    x3: *const [f64; IEND],
    x4: *const [f64; IEND],
    y1: *const [f64; IEND],
    y2: *const [f64; IEND],
    y3: *const [f64; IEND],
    y4: *const [f64; IEND],
    fx1: *const [f64; IEND],
    fx2: *const [f64; IEND],
    fx3: *const [f64; IEND],
    fx4: *const [f64; IEND],
    fy1: *const [f64; IEND],
    fy2: *const [f64; IEND],
    fy3: *const [f64; IEND],
    fy4: *const [f64; IEND],
    real_zones: *const [usize; IEND],
    half: f64,
    ptiny: f64,
    iend: usize,
) {
    unsafe {
        let ii = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if ii < iend {
            let i = (*real_zones)[ii];

            let xi = half * ((*x1)[i] + (*x2)[i] - (*x3)[i] - (*x4)[i]);
            let xj = half * ((*x2)[i] + (*x3)[i] - (*x4)[i] - (*x1)[i]);
            let yi = half * ((*y1)[i] + (*y2)[i] - (*y3)[i] - (*y4)[i]);
            let yj = half * ((*y2)[i] + (*y3)[i] - (*y4)[i] - (*y1)[i]);
            let fxi = half * ((*fx1)[i] + (*fx2)[i] - (*fx3)[i] - (*fx4)[i]);
            let fxj = half * ((*fx2)[i] + (*fx3)[i] - (*fx4)[i] - (*fx1)[i]);
            let fyi = half * ((*fy1)[i] + (*fy2)[i] - (*fy3)[i] - (*fy4)[i]);
            let fyj = half * ((*fy2)[i] + (*fy3)[i] - (*fy4)[i] - (*fy1)[i]);

            let rarea = 1.0 / (xi * yj - xj * yi + ptiny);
            let dfxdx = rarea * (fxi * yj - fxj * yi);
            let dfydy = rarea * (fyj * xi - fyi * xj);
            let affine = ((*fy1)[i] + (*fy2)[i] + (*fy3)[i] + (*fy4)[i])
                / ((*y1)[i] + (*y2)[i] + (*y3)[i] + (*y4)[i]);
            (*div)[i] = dfxdx + dfydy + affine;
        }
    }
}
