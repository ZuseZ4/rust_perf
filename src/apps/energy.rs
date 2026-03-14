#![allow(clippy::too_many_arguments)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

const DEFAULT_PROBLEM_SIZE: usize = 1_000_000;
const DEFAULT_REPS: u32 = 130;

const IEND: usize = DEFAULT_PROBLEM_SIZE;
const THREADS_PER_BLOCK: u32 = 256;
const BLOCKS: u32 = (IEND as u32).div_ceil(THREADS_PER_BLOCK);

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{workgroup_id_x as block_idx_x, workitem_id_x as thread_idx_x};

#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workgroup.size.x"]
    fn block_dim_x() -> u32;
}

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data, alloc_and_init_data_const, calc_checksum, free, init_data_scalar,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
pub struct Energy {
    rho0: f64,
    e_cut: f64,
    emin: f64,
    q_cut: f64,
    e_new: *mut f64,
    e_old: *mut f64,
    delvc: *mut f64,
    p_new: *mut f64,
    p_old: *mut f64,
    q_new: *mut f64,
    q_old: *mut f64,
    work: *mut f64,
    comp_half_step: *mut f64,
    p_half_step: *mut f64,
    bvc: *mut f64,
    pbvc: *mut f64,
    ql_old: *mut f64,
    qq_old: *mut f64,
    vnewc: *mut f64,
}

#[cfg(target_os = "linux")]
impl Energy {
    pub const INIT: Self = Energy {
        rho0: 0.0,
        e_cut: 0.0,
        emin: 0.0,
        q_cut: 0.0,
        e_new: core::ptr::null_mut(),
        e_old: core::ptr::null_mut(),
        delvc: core::ptr::null_mut(),
        p_new: core::ptr::null_mut(),
        p_old: core::ptr::null_mut(),
        q_new: core::ptr::null_mut(),
        q_old: core::ptr::null_mut(),
        work: core::ptr::null_mut(),
        comp_half_step: core::ptr::null_mut(),
        p_half_step: core::ptr::null_mut(),
        bvc: core::ptr::null_mut(),
        pbvc: core::ptr::null_mut(),
        ql_old: core::ptr::null_mut(),
        qq_old: core::ptr::null_mut(),
        vnewc: core::ptr::null_mut(),
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Energy {
    fn name(&self) -> &'static str {
        kernel_name!("ENERGY")
    }
    fn default_problem_size(&self) -> usize {
        DEFAULT_PROBLEM_SIZE
    }
    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        self.rho0 = 0.5;
        self.e_cut = 1.0e-7;
        self.emin = -1.0e15;
        self.q_cut = 1.0e-7;

        unsafe {
            self.e_new = alloc_and_init_data_const(IEND, 0.0);
            self.e_old = alloc_and_init_data(IEND);
            self.delvc = alloc_and_init_data(IEND);
            self.p_new = alloc_and_init_data(IEND);
            self.p_old = alloc_and_init_data(IEND);
            self.q_new = alloc_and_init_data_const(IEND, 0.0);
            self.q_old = alloc_and_init_data(IEND);
            self.work = alloc_and_init_data(IEND);
            self.comp_half_step = alloc_and_init_data(IEND);
            self.p_half_step = alloc_and_init_data(IEND);
            self.bvc = alloc_and_init_data(IEND);
            self.pbvc = alloc_and_init_data(IEND);
            self.ql_old = alloc_and_init_data(IEND);
            self.qq_old = alloc_and_init_data(IEND);
            self.vnewc = alloc_and_init_data(IEND);

            self.rho0 = init_data_scalar();
            self.e_cut = init_data_scalar();
            self.emin = init_data_scalar();
            self.q_cut = init_data_scalar();
        }
    }

    fn run_kernel(&mut self) {
        unsafe {
            energycalc1(
                self.e_new as *mut [f64; IEND],
                &*(self.e_old as *const [f64; IEND]),
                &*(self.delvc as *const [f64; IEND]),
                &*(self.p_old as *const [f64; IEND]),
                &*(self.q_old as *const [f64; IEND]),
                &*(self.work as *const [f64; IEND]),
                IEND,
            );
            energycalc2(
                &*(self.delvc as *const [f64; IEND]),
                self.q_new as *mut [f64; IEND],
                &*(self.comp_half_step as *const [f64; IEND]),
                &*(self.p_half_step as *const [f64; IEND]),
                self.e_new as *mut [f64; IEND],
                &*(self.bvc as *const [f64; IEND]),
                &*(self.pbvc as *const [f64; IEND]),
                &*(self.ql_old as *const [f64; IEND]),
                &*(self.qq_old as *const [f64; IEND]),
                self.rho0,
                IEND,
            );
            energycalc3(
                self.e_new as *mut [f64; IEND],
                &*(self.delvc as *const [f64; IEND]),
                &*(self.p_old as *const [f64; IEND]),
                &*(self.q_old as *const [f64; IEND]),
                &*(self.p_half_step as *const [f64; IEND]),
                &*(self.q_new as *const [f64; IEND]),
                IEND,
            );
            energycalc4(
                self.e_new as *mut [f64; IEND],
                &*(self.work as *const [f64; IEND]),
                self.e_cut,
                self.emin,
                IEND,
            );
            energycalc5(
                &*(self.delvc as *const [f64; IEND]),
                &*(self.pbvc as *const [f64; IEND]),
                self.e_new as *mut [f64; IEND],
                &*(self.vnewc as *const [f64; IEND]),
                &*(self.bvc as *const [f64; IEND]),
                &*(self.p_new as *const [f64; IEND]),
                &*(self.ql_old as *const [f64; IEND]),
                &*(self.qq_old as *const [f64; IEND]),
                &*(self.p_old as *const [f64; IEND]),
                &*(self.q_old as *const [f64; IEND]),
                &*(self.p_half_step as *const [f64; IEND]),
                &*(self.q_new as *const [f64; IEND]),
                self.rho0,
                self.e_cut,
                self.emin,
                IEND,
            );
            energycalc6(
                &*(self.delvc as *const [f64; IEND]),
                &*(self.pbvc as *const [f64; IEND]),
                self.e_new as *mut [f64; IEND],
                &*(self.vnewc as *const [f64; IEND]),
                &*(self.bvc as *const [f64; IEND]),
                &*(self.p_new as *const [f64; IEND]),
                self.q_new as *mut [f64; IEND],
                &*(self.ql_old as *const [f64; IEND]),
                &*(self.qq_old as *const [f64; IEND]),
                self.rho0,
                self.q_cut,
                IEND,
            );
        }
    }

    fn update_checksum(&self) -> f64 {
        unsafe {
            calc_checksum(self.e_new as *const f64, IEND)
                + calc_checksum(self.q_new as *const f64, IEND)
        }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.e_new);
            self.e_new = core::ptr::null_mut();
            free(self.e_old);
            self.e_old = core::ptr::null_mut();
            free(self.delvc);
            self.delvc = core::ptr::null_mut();
            free(self.p_new);
            self.p_new = core::ptr::null_mut();
            free(self.p_old);
            self.p_old = core::ptr::null_mut();
            free(self.q_new);
            self.q_new = core::ptr::null_mut();
            free(self.q_old);
            self.q_old = core::ptr::null_mut();
            free(self.work);
            self.work = core::ptr::null_mut();
            free(self.comp_half_step);
            self.comp_half_step = core::ptr::null_mut();
            free(self.p_half_step);
            self.p_half_step = core::ptr::null_mut();
            free(self.bvc);
            self.bvc = core::ptr::null_mut();
            free(self.pbvc);
            self.pbvc = core::ptr::null_mut();
            free(self.ql_old);
            self.ql_old = core::ptr::null_mut();
            free(self.qq_old);
            self.qq_old = core::ptr::null_mut();
            free(self.vnewc);
            self.vnewc = core::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn energycalc1(
    e_new: *mut [f64; IEND],
    e_old: &[f64; IEND],
    delvc: &[f64; IEND],
    p_old: &[f64; IEND],
    q_old: &[f64; IEND],
    work: &[f64; IEND],
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
unsafe fn energycalc2(
    delvc: &[f64; IEND],
    q_new: *mut [f64; IEND],
    comp_half_step: &[f64; IEND],
    p_half_step: &[f64; IEND],
    e_new: *mut [f64; IEND],
    bvc: &[f64; IEND],
    pbvc: &[f64; IEND],
    ql_old: &[f64; IEND],
    qq_old: &[f64; IEND],
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
            comp_half_step,
            p_half_step,
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
unsafe fn energycalc3(
    e_new: *mut [f64; IEND],
    delvc: &[f64; IEND],
    p_old: &[f64; IEND],
    q_old: &[f64; IEND],
    p_half_step: &[f64; IEND],
    q_new: &[f64; IEND],
    iend: usize,
) {
    core::intrinsics::offload(
        _energycalc3,
        [BLOCKS, 1, 1],
        [THREADS_PER_BLOCK, 1, 1],
        (e_new, delvc, p_old, q_old, p_half_step, q_new, iend),
    )
}
#[cfg(target_os = "linux")]
unsafe fn energycalc4(
    e_new: *mut [f64; IEND],
    work: &[f64; IEND],
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
unsafe fn energycalc5(
    delvc: &[f64; IEND],
    pbvc: &[f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: &[f64; IEND],
    bvc: &[f64; IEND],
    p_new: &[f64; IEND],
    ql_old: &[f64; IEND],
    qq_old: &[f64; IEND],
    p_old: &[f64; IEND],
    q_old: &[f64; IEND],
    p_half_step: &[f64; IEND],
    q_new: &[f64; IEND],
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
            delvc,
            pbvc,
            e_new,
            vnewc,
            bvc,
            p_new,
            ql_old,
            qq_old,
            p_old,
            q_old,
            p_half_step,
            q_new,
            rho0,
            e_cut,
            emin,
            iend,
        ),
    )
}
#[cfg(target_os = "linux")]
unsafe fn energycalc6(
    delvc: &[f64; IEND],
    pbvc: &[f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: &[f64; IEND],
    bvc: &[f64; IEND],
    p_new: &[f64; IEND],
    q_new: *mut [f64; IEND],
    ql_old: &[f64; IEND],
    qq_old: &[f64; IEND],
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
    pub fn _energycalc1(
        e_new: *mut [f64; IEND],
        e_old: &[f64; IEND],
        delvc: &[f64; IEND],
        p_old: &[f64; IEND],
        q_old: &[f64; IEND],
        work: &[f64; IEND],
        iend: usize,
    );
    pub fn _energycalc2(
        delvc: &[f64; IEND],
        q_new: *mut [f64; IEND],
        comp_half_step: &[f64; IEND],
        p_half_step: &[f64; IEND],
        e_new: *mut [f64; IEND],
        bvc: &[f64; IEND],
        pbvc: &[f64; IEND],
        ql_old: &[f64; IEND],
        qq_old: &[f64; IEND],
        rho0: f64,
        iend: usize,
    );
    pub fn _energycalc3(
        e_new: *mut [f64; IEND],
        delvc: &[f64; IEND],
        p_old: &[f64; IEND],
        q_old: &[f64; IEND],
        p_half_step: &[f64; IEND],
        q_new: &[f64; IEND],
        iend: usize,
    );
    pub fn _energycalc4(
        e_new: *mut [f64; IEND],
        work: &[f64; IEND],
        e_cut: f64,
        emin: f64,
        iend: usize,
    );
    pub fn _energycalc5(
        delvc: &[f64; IEND],
        pbvc: &[f64; IEND],
        e_new: *mut [f64; IEND],
        vnewc: &[f64; IEND],
        bvc: &[f64; IEND],
        p_new: &[f64; IEND],
        ql_old: &[f64; IEND],
        qq_old: &[f64; IEND],
        p_old: &[f64; IEND],
        q_old: &[f64; IEND],
        p_half_step: &[f64; IEND],
        q_new: &[f64; IEND],
        rho0: f64,
        e_cut: f64,
        emin: f64,
        iend: usize,
    );
    pub fn _energycalc6(
        delvc: &[f64; IEND],
        pbvc: &[f64; IEND],
        e_new: *mut [f64; IEND],
        vnewc: &[f64; IEND],
        bvc: &[f64; IEND],
        p_new: &[f64; IEND],
        q_new: *mut [f64; IEND],
        ql_old: &[f64; IEND],
        qq_old: &[f64; IEND],
        rho0: f64,
        q_cut: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc1(
    e_new: *mut [f64; IEND],
    e_old: &[f64; IEND],
    delvc: &[f64; IEND],
    p_old: &[f64; IEND],
    q_old: &[f64; IEND],
    work: &[f64; IEND],
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

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc2(
    delvc: &[f64; IEND],
    q_new: *mut [f64; IEND],
    comp_half_step: &[f64; IEND],
    p_half_step: &[f64; IEND],
    e_new: *mut [f64; IEND],
    bvc: &[f64; IEND],
    pbvc: &[f64; IEND],
    ql_old: &[f64; IEND],
    qq_old: &[f64; IEND],
    rho0: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            if (*delvc)[i] > 0.0 {
                (*q_new)[i] = 0.0;
            } else {
                let vhalf = 1.0 / (1.0 + (*comp_half_step)[i]);
                let mut ssc = ((*pbvc)[i] * (*e_new)[i]
                    + vhalf * vhalf * (*bvc)[i] * (*p_half_step)[i])
                    / rho0;
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

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc3(
    e_new: *mut [f64; IEND],
    delvc: &[f64; IEND],
    p_old: &[f64; IEND],
    q_old: &[f64; IEND],
    p_half_step: &[f64; IEND],
    q_new: &[f64; IEND],
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            (*e_new)[i] += 0.5
                * (*delvc)[i]
                * (3.0 * ((*p_old)[i] + (*q_old)[i]) - 4.0 * ((*p_half_step)[i] + (*q_new)[i]));
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc4(
    e_new: *mut [f64; IEND],
    work: &[f64; IEND],
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            (*e_new)[i] += 0.5 * (*work)[i];
            if (*e_new)[i].abs() < e_cut {
                (*e_new)[i] = 0.0;
            }
            if (*e_new)[i] < emin {
                (*e_new)[i] = emin;
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc5(
    delvc: &[f64; IEND],
    pbvc: &[f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: &[f64; IEND],
    bvc: &[f64; IEND],
    p_new: &[f64; IEND],
    ql_old: &[f64; IEND],
    qq_old: &[f64; IEND],
    p_old: &[f64; IEND],
    q_old: &[f64; IEND],
    p_half_step: &[f64; IEND],
    q_new: &[f64; IEND],
    rho0: f64,
    e_cut: f64,
    emin: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend {
            let q_tilde = if (*delvc)[i] > 0.0 {
                0.0
            } else {
                let mut ssc = ((*pbvc)[i] * (*e_new)[i]
                    + (*vnewc)[i] * (*vnewc)[i] * (*bvc)[i] * (*p_new)[i])
                    / rho0;
                if ssc <= 0.1111111e-36 {
                    ssc = 0.3333333e-18;
                } else {
                    ssc = core::f64::math::sqrt(ssc);
                }
                ssc * (*ql_old)[i] + (*qq_old)[i]
            };
            (*e_new)[i] -= (7.0 * ((*p_old)[i] + (*q_old)[i])
                - 8.0 * ((*p_half_step)[i] + (*q_new)[i])
                + ((*p_new)[i] + q_tilde))
                * (*delvc)[i]
                / 6.0;
            if (*e_new)[i].abs() < e_cut {
                (*e_new)[i] = 0.0;
            }
            if (*e_new)[i] < emin {
                (*e_new)[i] = emin;
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _energycalc6(
    delvc: &[f64; IEND],
    pbvc: &[f64; IEND],
    e_new: *mut [f64; IEND],
    vnewc: &[f64; IEND],
    bvc: &[f64; IEND],
    p_new: &[f64; IEND],
    q_new: *mut [f64; IEND],
    ql_old: &[f64; IEND],
    qq_old: &[f64; IEND],
    rho0: f64,
    q_cut: f64,
    iend: usize,
) {
    unsafe {
        let i = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
        if i < iend && (*delvc)[i] <= 0.0 {
            let mut ssc = ((*pbvc)[i] * (*e_new)[i]
                + (*vnewc)[i] * (*vnewc)[i] * (*bvc)[i] * (*p_new)[i])
                / rho0;
            if ssc <= 0.1111111e-36 {
                ssc = 0.3333333e-18;
            } else {
                ssc = core::f64::math::sqrt(ssc);
            }
            (*q_new)[i] = ssc * (*ql_old)[i] + (*qq_old)[i];
            if (*q_new)[i].abs() < q_cut {
                (*q_new)[i] = 0.0;
            }
        }
    }
}
