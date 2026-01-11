#![allow(internal_features)]
#![allow(non_snake_case)]
#![feature(abi_gpu_kernel)]
#![feature(core_float_math)]
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

const IEND: usize = 1_000_000;
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
fn main() {
    unsafe {
        let e_new = alloc_array::<f64>(IEND);
        let e_old = alloc_array::<f64>(IEND);
        let delvc = alloc_array::<f64>(IEND);
        let p_new = alloc_array::<f64>(IEND);
        let p_old = alloc_array::<f64>(IEND);
        let q_new = alloc_array::<f64>(IEND);
        let q_old = alloc_array::<f64>(IEND);
        let work = alloc_array::<f64>(IEND);
        let compHalfStep = alloc_array::<f64>(IEND);
        let pHalfStep = alloc_array::<f64>(IEND);
        let bvc = alloc_array::<f64>(IEND);
        let pbvc = alloc_array::<f64>(IEND);
        let ql_old = alloc_array::<f64>(IEND);
        let qq_old = alloc_array::<f64>(IEND);
        let vnewc = alloc_array::<f64>(IEND);

        let rho0 = 0.0;
        let e_cut = 0.0;
        let emin = 0.0;
        let q_cut = 0.0;

        let start = get_time_ns();
        energycalc1(
            e_new as *mut [f64; IEND],
            e_old as *const [f64; IEND],
            delvc as *const [f64; IEND],
            p_old as *const [f64; IEND],
            q_old as *const [f64; IEND],
            work as *const [f64; IEND],
            IEND,
        );

        energycalc2(
            delvc as *const [f64; IEND],
            q_new as *mut [f64; IEND],
            compHalfStep as *const [f64; IEND],
            pHalfStep as *const [f64; IEND],
            e_new as *mut [f64; IEND],
            bvc as *const [f64; IEND],
            pbvc as *const [f64; IEND],
            ql_old as *const [f64; IEND],
            qq_old as *const [f64; IEND],
            rho0,
            IEND,
        );

        energycalc3(
            e_new as *mut [f64; IEND],
            delvc as *const [f64; IEND],
            p_old as *const [f64; IEND],
            q_old as *const [f64; IEND],
            pHalfStep as *const [f64; IEND],
            q_new as *const [f64; IEND],
            IEND,
        );

        energycalc4(
            e_new as *mut [f64; IEND],
            work as *const [f64; IEND],
            e_cut,
            emin,
            IEND,
        );

        energycalc5(
            delvc as *const [f64; IEND],
            pbvc as *const [f64; IEND],
            e_new as *mut [f64; IEND],
            vnewc as *const [f64; IEND],
            bvc as *const [f64; IEND],
            p_new as *const [f64; IEND],
            ql_old as *const [f64; IEND],
            qq_old as *const [f64; IEND],
            p_old as *const [f64; IEND],
            q_old as *const [f64; IEND],
            pHalfStep as *const [f64; IEND],
            q_new as *const [f64; IEND],
            rho0,
            e_cut,
            emin,
            IEND,
        );

        energycalc6(
            delvc as *const [f64; IEND],
            pbvc as *const [f64; IEND],
            e_new as *mut [f64; IEND],
            vnewc as *const [f64; IEND],
            bvc as *const [f64; IEND],
            p_new as *const [f64; IEND],
            q_new as *mut [f64; IEND],
            ql_old as *const [f64; IEND],
            qq_old as *const [f64; IEND],
            rho0,
            q_cut,
            IEND,
        );
        let end = get_time_ns();
        let duration_s = (end - start) as f64 / 1_000_000_000.0;
        libc::printf(c"%f\n".as_ptr(), duration_s);

        free_array(e_new);
        free_array(e_old);
        free_array(delvc);
        free_array(p_new);
        free_array(p_old);
        free_array(q_new);
        free_array(q_old);
        free_array(work);
        free_array(compHalfStep);
        free_array(pHalfStep);
        free_array(bvc);
        free_array(pbvc);
        free_array(ql_old);
        free_array(qq_old);
        free_array(vnewc);
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc1(
    e_new: *mut [f64; IEND],
    e_old: *const [f64; IEND],
    delvc: *const [f64; IEND],
    p_old: *const [f64; IEND],
    q_old: *const [f64; IEND],
    work: *const [f64; IEND],
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc1,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (e_new, e_old, delvc, p_old, q_old, work, iend),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc1(
        e_new: *mut [f64; IEND],
        e_old: *const [f64; IEND],
        delvc: *const [f64; IEND],
        p_old: *const [f64; IEND],
        q_old: *const [f64; IEND],
        work: *const [f64; IEND],
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc1(
    e_new: *mut [f64; IEND],
    e_old: *const [f64; IEND],
    delvc: *const [f64; IEND],
    p_old: *const [f64; IEND],
    q_old: *const [f64; IEND],
    work: *const [f64; IEND],
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

        if i < iend {
            (*e_new)[i] =
                (*e_old)[i] - 0.5 * (*delvc)[i] * ((*p_old)[i] + (*q_old)[i]) + 0.5 * (*work)[i];
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc2(
    delvc: *const [f64; IEND],
    q_new: *mut [f64; IEND],
    compHalfStep: *const [f64; IEND],
    pHalfStep: *const [f64; IEND],
    e_new: *mut [f64; IEND],
    bvc: *const [f64; IEND],
    pbvc: *const [f64; IEND],
    ql_old: *const [f64; IEND],
    qq_old: *const [f64; IEND],
    rho0: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc2,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (
            delvc,
            q_new,
            compHalfStep,
            pHalfStep,
            e_new,
            bvc,
            pbvc,
            ql_old,
            qq_old,
            rho0,
            iend,
        ),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc2(
        delvc: *const [f64; IEND],
        q_new: *mut [f64; IEND],
        compHalfStep: *const [f64; IEND],
        pHalfStep: *const [f64; IEND],
        e_new: *mut [f64; IEND],
        bvc: *const [f64; IEND],
        pbvc: *const [f64; IEND],
        ql_old: *const [f64; IEND],
        qq_old: *const [f64; IEND],
        rho0: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc2(
    delvc: *const [f64; IEND],
    q_new: *mut [f64; IEND],
    compHalfStep: *const [f64; IEND],
    pHalfStep: *const [f64; IEND],
    e_new: *mut [f64; IEND],
    bvc: *const [f64; IEND],
    pbvc: *const [f64; IEND],
    ql_old: *const [f64; IEND],
    qq_old: *const [f64; IEND],
    rho0: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            if (*delvc)[i] > 0.0 {
                (*q_new)[i] = 0.0;
            } else {
                let vhalf = 1.0 / (1.0 + (*compHalfStep)[i]);
                let mut ssc =
                    ((*pbvc)[i] * (*e_new)[i] + vhalf * vhalf * (*bvc)[i] * (*pHalfStep)[i]) / rho0;
                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = core::f64::math::sqrt(ssc);
                }
                (*q_new)[i] = ssc * (*ql_old)[i] + (*qq_old)[i];
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc3(
    e_new: *mut [f64; IEND],
    delvc: *const [f64; IEND],
    p_old: *const [f64; IEND],
    q_old: *const [f64; IEND],
    pHalfStep: *const [f64; IEND],
    q_new: *const [f64; IEND],
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc3,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (e_new, delvc, p_old, q_old, pHalfStep, q_new, iend),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc3(
        e_new: *mut [f64; IEND],
        delvc: *const [f64; IEND],
        p_old: *const [f64; IEND],
        q_old: *const [f64; IEND],
        pHalfStep: *const [f64; IEND],
        q_new: *const [f64; IEND],
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc3(
    e_new: *mut [f64; IEND],
    delvc: *const [f64; IEND],
    p_old: *const [f64; IEND],
    q_old: *const [f64; IEND],
    pHalfStep: *const [f64; IEND],
    q_new: *const [f64; IEND],
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            (*e_new)[i] = (*e_new)[i]
                + 0.5
                    * (*delvc)[i]
                    * (3.0 * ((*p_old)[i] + (*q_old)[i]) - 4.0 * ((*pHalfStep)[i] + (*q_new)[i]));
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc4(
    e_new: *mut [f64; IEND],
    work: *const [f64; IEND],
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc4,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (e_new, work, e_cut, emin, iend),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc4(
        e_new: *mut [f64; IEND],
        work: *const [f64; IEND],
        e_cut: f64,
        emin: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc4(
    e_new: *mut [f64; IEND],
    work: *const [f64; IEND],
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            (*e_new)[i] += 0.5 * (*work)[i];
            if ((*e_new)[i]).abs() < e_cut {
                (*e_new)[i] = 0.0;
            }
            if (*e_new)[i] < emin {
                (*e_new)[i] = emin;
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc5(
    delvc: *const [f64; IEND],
    pbvc: *const [f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: *const [f64; IEND],
    bvc: *const [f64; IEND],
    p_new: *const [f64; IEND],
    ql_old: *const [f64; IEND],
    qq_old: *const [f64; IEND],
    p_old: *const [f64; IEND],
    q_old: *const [f64; IEND],
    pHalfStep: *const [f64; IEND],
    q_new: *const [f64; IEND],
    rho0: f64,
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc5,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (
            delvc, pbvc, e_new, vnewc, bvc, p_new, ql_old, qq_old, p_old, q_old, pHalfStep, q_new,
            rho0, e_cut, emin, iend,
        ),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc5(
        delvc: *const [f64; IEND],
        pbvc: *const [f64; IEND],
        e_new: *mut [f64; IEND],
        vnewc: *const [f64; IEND],
        bvc: *const [f64; IEND],
        p_new: *const [f64; IEND],
        ql_old: *const [f64; IEND],
        qq_old: *const [f64; IEND],
        p_old: *const [f64; IEND],
        q_old: *const [f64; IEND],
        pHalfStep: *const [f64; IEND],
        q_new: *const [f64; IEND],
        rho0: f64,
        e_cut: f64,
        emin: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc5(
    delvc: *const [f64; IEND],
    pbvc: *const [f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: *const [f64; IEND],
    bvc: *const [f64; IEND],
    p_new: *const [f64; IEND],
    ql_old: *const [f64; IEND],
    qq_old: *const [f64; IEND],
    p_old: *const [f64; IEND],
    q_old: *const [f64; IEND],
    pHalfStep: *const [f64; IEND],
    q_new: *const [f64; IEND],
    rho0: f64,
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

        if i < iend {
            let q_tilde: f64;

            if (*delvc)[i] > 0.0 {
                q_tilde = 0.0;
            } else {
                let mut ssc = ((*pbvc)[i] * (*e_new)[i]
                    + (*vnewc)[i] * (*vnewc)[i] * (*bvc)[i] * (*p_new)[i])
                    / rho0;

                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = core::f64::math::sqrt(ssc);
                }

                q_tilde = ssc * (*ql_old)[i] + (*qq_old)[i];
            }

            (*e_new)[i] = (*e_new)[i]
                - (7.0 * ((*p_old)[i] + (*q_old)[i]) - 8.0 * ((*pHalfStep)[i] + (*q_new)[i])
                    + ((*p_new)[i] + q_tilde))
                    * (*delvc)[i]
                    / 6.0;

            if ((*e_new)[i]).abs() < e_cut {
                (*e_new)[i] = 0.0;
            }

            if (*e_new)[i] < emin {
                (*e_new)[i] = emin;
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc6(
    delvc: *const [f64; IEND],
    pbvc: *const [f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: *const [f64; IEND],
    bvc: *const [f64; IEND],
    p_new: *const [f64; IEND],
    q_new: *const [f64; IEND],
    ql_old: *const [f64; IEND],
    qq_old: *const [f64; IEND],
    rho0: f64,
    q_cut: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc6,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (
            delvc, pbvc, e_new, vnewc, bvc, p_new, q_new, ql_old, qq_old, rho0, q_cut, iend,
        ),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc6(
        delvc: *const [f64; IEND],
        pbvc: *const [f64; IEND],
        e_new: *mut [f64; IEND],
        vnewc: *const [f64; IEND],
        bvc: *const [f64; IEND],
        p_new: *const [f64; IEND],
        q_new: *const [f64; IEND],
        ql_old: *const [f64; IEND],
        qq_old: *const [f64; IEND],
        rho0: f64,
        q_cut: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc6(
    delvc: *const [f64; IEND],
    pbvc: *const [f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: *const [f64; IEND],
    bvc: *const [f64; IEND],
    p_new: *const [f64; IEND],
    q_new: *mut [f64; IEND],
    ql_old: *const [f64; IEND],
    qq_old: *const [f64; IEND],
    rho0: f64,
    q_cut: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            if (*delvc)[i] <= 0.0 {
                let mut ssc = ((*pbvc)[i] * (*e_new)[i]
                    + (*vnewc)[i] * (*vnewc)[i] * (*bvc)[i] * (*p_new)[i])
                    / rho0;

                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = core::f64::math::sqrt(ssc);
                }

                (*q_new)[i] = ssc * (*ql_old)[i] + (*qq_old)[i];

                if ((*q_new)[i]).abs() < q_cut {
                    (*q_new)[i] = 0.0;
                }
            }
        }
    }
}
