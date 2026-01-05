#![allow(internal_features)]
#![allow(non_snake_case)]
#![feature(abi_gpu_kernel)]
#![feature(core_float_math)]
#![feature(rustc_attrs)]
#![feature(link_llvm_intrinsics)]
#![feature(core_intrinsics)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![no_std]

#[cfg(not(target_os = "linux"))]
use ::core::f64::math::sqrt;

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
    let mut e_new = [1.0, 2.0, 3.0, 4.0];
    let e_old = [1.0, 2.0, 3.0, 4.0];
    let delvc = [1.0, 2.0, 3.0, 4.0];
    let p_new = [1.0, 2.0, 3.0, 4.0];
    let p_old = [1.0, 2.0, 3.0, 4.0];
    let mut q_new = [1.0, 2.0, 3.0, 4.0];
    let q_old = [1.0, 2.0, 3.0, 4.0];
    let work = [1.0, 2.0, 3.0, 4.0];
    let compHalfStep = [1.0, 2.0, 3.0, 4.0];
    let pHalfStep = [1.0, 2.0, 3.0, 4.0];
    let bvc = [1.0, 2.0, 3.0, 4.0];
    let pbvc = [1.0, 2.0, 3.0, 4.0];
    let ql_old = [1.0, 2.0, 3.0, 4.0];
    let qq_old = [1.0, 2.0, 3.0, 4.0];
    let vnewc = [1.0, 2.0, 3.0, 4.0];

    let rho0 = 0.0;
    let e_cut = 0.0;
    let emin = 0.0;
    let q_cut = 0.0;
    let iend = 4;

    let e_new_p = e_new.as_mut_ptr();
    let e_old_p = e_old.as_ptr();
    let delvc_p = delvc.as_ptr();
    let p_old_p = p_old.as_ptr();
    let q_old_p = q_old.as_ptr();
    let work_p = work.as_ptr();

    let q_new_p = q_new.as_mut_ptr();
    let compHalfStep_p = compHalfStep.as_ptr();
    let pHalfStep_p = pHalfStep.as_ptr();
    let bvc_p = bvc.as_ptr();
    let pbvc_p = pbvc.as_ptr();
    let ql_old_p = ql_old.as_ptr();
    let qq_old_p = qq_old.as_ptr();
    let vnewc_p = vnewc.as_ptr();
    let p_new_p = p_new.as_ptr();

    unsafe {
        energycalc1(e_new_p, e_old_p, delvc_p, p_old_p, q_old_p, work_p, iend);

        energycalc2(
            delvc_p,
            q_new_p,
            compHalfStep_p,
            pHalfStep_p,
            e_new_p,
            bvc_p,
            pbvc_p,
            ql_old_p,
            qq_old_p,
            rho0,
            iend,
        );

        energycalc3(
            e_new_p,
            delvc_p,
            p_old_p,
            q_old_p,
            pHalfStep_p,
            q_new_p,
            iend,
        );

        energycalc4(e_new_p, work_p, e_cut, emin, iend);

        energycalc5(
            delvc_p,
            pbvc_p,
            e_new_p,
            vnewc_p,
            bvc_p,
            p_new_p,
            ql_old_p,
            qq_old_p,
            p_old_p,
            q_old_p,
            pHalfStep_p,
            q_new_p,
            rho0,
            e_cut,
            emin,
            iend,
        );

        energycalc6(
            delvc_p, pbvc_p, e_new_p, vnewc_p, bvc_p, p_new_p, q_new_p, ql_old_p, qq_old_p, rho0,
            q_cut, iend,
        );
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc1(
    e_new: *mut f64,
    e_old: *const f64,
    delvc: *const f64,
    p_old: *const f64,
    q_old: *const f64,
    work: *const f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc1,
        [256, 1, 1],
        [32, 1, 1],
        (e_new, e_old, delvc, p_old, q_old, work, iend),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc1(
        e_new: *mut f64,
        e_old: *const f64,
        delvc: *const f64,
        p_old: *const f64,
        q_old: *const f64,
        work: *const f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc1(
    e_new: *mut f64,
    e_old: *const f64,
    delvc: *const f64,
    p_old: *const f64,
    q_old: *const f64,
    work: *const f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

        if i < iend {
            *e_new.add(i) = *e_old.add(i) - 0.5 * *delvc.add(i) * (*p_old.add(i) + *q_old.add(i))
                + 0.5 * *work.add(i);
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc2(
    delvc: *const f64,
    q_new: *mut f64,
    compHalfStep: *const f64,
    pHalfStep: *const f64,
    e_new: *mut f64,
    bvc: *const f64,
    pbvc: *const f64,
    ql_old: *const f64,
    qq_old: *const f64,
    rho0: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc2,
        [256, 1, 1],
        [32, 1, 1],
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
        delvc: *const f64,
        q_new: *mut f64,
        compHalfStep: *const f64,
        pHalfStep: *const f64,
        e_new: *mut f64,
        bvc: *const f64,
        pbvc: *const f64,
        ql_old: *const f64,
        qq_old: *const f64,
        rho0: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc2(
    delvc: *const f64,
    q_new: *mut f64,
    compHalfStep: *const f64,
    pHalfStep: *const f64,
    e_new: *mut f64,
    bvc: *const f64,
    pbvc: *const f64,
    ql_old: *const f64,
    qq_old: *const f64,
    rho0: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            if *delvc.add(i) > 0.0 {
                *q_new.add(i) = 0.0;
            } else {
                let vhalf = 1.0 / (1.0 + *compHalfStep.add(i));
                let mut ssc = (*pbvc.add(i) * *e_new.add(i)
                    + vhalf * vhalf * *bvc.add(i) * *pHalfStep.add(i))
                    / rho0;
                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = sqrt(ssc);
                }
                *q_new.add(i) = ssc * *ql_old.add(i) + *qq_old.add(i);
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc3(
    e_new: *mut f64,
    delvc: *const f64,
    p_old: *const f64,
    q_old: *const f64,
    pHalfStep: *const f64,
    q_new: *const f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc3,
        [256, 1, 1],
        [32, 1, 1],
        (e_new, delvc, p_old, q_old, pHalfStep, q_new, iend),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc3(
        e_new: *mut f64,
        delvc: *const f64,
        p_old: *const f64,
        q_old: *const f64,
        pHalfStep: *const f64,
        q_new: *const f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc3(
    e_new: *mut f64,
    delvc: *const f64,
    p_old: *const f64,
    q_old: *const f64,
    pHalfStep: *const f64,
    q_new: *const f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            *e_new.add(i) = *e_new.add(i)
                + 0.5
                    * *delvc.add(i)
                    * (3.0 * (*p_old.add(i) + *q_old.add(i))
                        - 4.0 * (*pHalfStep.add(i) + *q_new.add(i)));
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc4(e_new: *mut f64, work: *const f64, e_cut: f64, emin: f64, iend: usize) {
    core::intrinsics::offload(
        _energycalc4,
        [256, 1, 1],
        [32, 1, 1],
        (e_new, work, e_cut, emin, iend),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc4(e_new: *mut f64, work: *const f64, e_cut: f64, emin: f64, iend: usize);
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc4(
    e_new: *mut f64,
    work: *const f64,
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            *e_new.add(i) += 0.5 * *work.add(i);
            if (*e_new.add(i)).abs() < e_cut {
                *e_new.add(i) = 0.0;
            }
            if *e_new.add(i) < emin {
                *e_new.add(i) = emin;
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc5(
    delvc: *const f64,
    pbvc: *const f64,
    e_new: *mut f64,
    vnewc: *const f64,
    bvc: *const f64,
    p_new: *const f64,
    ql_old: *const f64,
    qq_old: *const f64,
    p_old: *const f64,
    q_old: *const f64,
    pHalfStep: *const f64,
    q_new: *const f64,
    rho0: f64,
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc5,
        [256, 1, 1],
        [32, 1, 1],
        (
            delvc, pbvc, e_new, vnewc, bvc, p_new, ql_old, qq_old, p_old, q_old, pHalfStep, q_new,
            rho0, e_cut, emin, iend,
        ),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc5(
        delvc: *const f64,
        pbvc: *const f64,
        e_new: *mut f64,
        vnewc: *const f64,
        bvc: *const f64,
        p_new: *const f64,
        ql_old: *const f64,
        qq_old: *const f64,
        p_old: *const f64,
        q_old: *const f64,
        pHalfStep: *const f64,
        q_new: *const f64,
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
    delvc: *const f64,
    pbvc: *const f64,
    e_new: *mut f64,
    vnewc: *const f64,
    bvc: *const f64,
    p_new: *const f64,
    ql_old: *const f64,
    qq_old: *const f64,
    p_old: *const f64,
    q_old: *const f64,
    pHalfStep: *const f64,
    q_new: *const f64,
    rho0: f64,
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

        if i < iend {
            let q_tilde: f64;

            if *delvc.add(i) > 0.0 {
                q_tilde = 0.0;
            } else {
                let mut ssc = (*pbvc.add(i) * *e_new.add(i)
                    + *vnewc.add(i) * *vnewc.add(i) * *bvc.add(i) * *p_new.add(i))
                    / rho0;

                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = sqrt(ssc);
                }

                q_tilde = ssc * *ql_old.add(i) + *qq_old.add(i);
            }

            *e_new.add(i) = *e_new.add(i)
                - (7.0 * (*p_old.add(i) + *q_old.add(i))
                    - 8.0 * (*pHalfStep.add(i) + *q_new.add(i))
                    + (*p_new.add(i) + q_tilde))
                    * *delvc.add(i)
                    / 6.0;

            if (*e_new.add(i)).abs() < e_cut {
                *e_new.add(i) = 0.0;
            }

            if *e_new.add(i) < emin {
                *e_new.add(i) = emin;
            }
        }
    }
}

#[cfg(target_os = "linux")]
#[inline(never)]
unsafe fn energycalc6(
    delvc: *const f64,
    pbvc: *const f64,
    e_new: *mut f64,
    vnewc: *const f64,
    bvc: *const f64,
    p_new: *const f64,
    q_new: *const f64,
    ql_old: *const f64,
    qq_old: *const f64,
    rho0: f64,
    q_cut: f64,
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc6,
        [256, 1, 1],
        [32, 1, 1],
        (
            delvc, pbvc, e_new, vnewc, bvc, p_new, q_new, ql_old, qq_old, rho0, q_cut, iend,
        ),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _energycalc6(
        delvc: *const f64,
        pbvc: *const f64,
        e_new: *mut f64,
        vnewc: *const f64,
        bvc: *const f64,
        p_new: *const f64,
        q_new: *const f64,
        ql_old: *const f64,
        qq_old: *const f64,
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
    delvc: *const f64,
    pbvc: *const f64,
    e_new: *mut f64,
    vnewc: *const f64,
    bvc: *const f64,
    p_new: *const f64,
    q_new: *mut f64,
    ql_old: *const f64,
    qq_old: *const f64,
    rho0: f64,
    q_cut: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            if *delvc.add(i) <= 0.0 {
                let mut ssc = (*pbvc.add(i) * *e_new.add(i)
                    + *vnewc.add(i) * *vnewc.add(i) * *bvc.add(i) * *p_new.add(i))
                    / rho0;

                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = sqrt(ssc);
                }

                *q_new.add(i) = ssc * *ql_old.add(i) + *qq_old.add(i);

                if (*q_new.add(i)).abs() < q_cut {
                    *q_new.add(i) = 0.0;
                }
            }
        }
    }
}
