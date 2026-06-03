#![allow(clippy::too_many_arguments)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

const DEFAULT_PROBLEM_SIZE: usize = 1_000_000;
const DEFAULT_REPS: u32 = 130;

const IEND: usize = DEFAULT_PROBLEM_SIZE;
const THREADS_PER_BLOCK: u32 = 256;
const BLOCKS: u32 = (IEND as u32).div_ceil(THREADS_PER_BLOCK);

use core::offload::offload_kernel;
use rustc_offload_frontend::partition::{Linear1D, PartitioningStrategy, Region};

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{PreloadMut, preload_mut};

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
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Energy {
    rho0: Real,
    e_cut: Real,
    emin: Real,
    q_cut: Real,
    e_new: *mut Real,
    e_old: *mut Real,
    delvc: *mut Real,
    p_new: *mut Real,
    p_old: *mut Real,
    q_new: *mut Real,
    q_old: *mut Real,
    work: *mut Real,
    comp_half_step: *mut Real,
    p_half_step: *mut Real,
    bvc: *mut Real,
    pbvc: *mut Real,
    ql_old: *mut Real,
    qq_old: *mut Real,
    vnewc: *mut Real,
}

#[cfg(target_os = "linux")]
impl Energy {
    pub const INIT: Self = Energy {
        rho0: to_real(0.0),
        e_cut: to_real(0.0),
        emin: to_real(0.0),
        q_cut: to_real(0.0),
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
        self.rho0 = to_real(0.5);
        self.e_cut = to_real(1.0e-7);
        self.emin = to_real(-1.0e15);
        self.q_cut = to_real(1.0e-7);

        unsafe {
            self.e_new = alloc_and_init_data_const(IEND, to_real(0.0));
            self.e_old = alloc_and_init_data(IEND);
            self.delvc = alloc_and_init_data(IEND);
            self.p_new = alloc_and_init_data(IEND);
            self.p_old = alloc_and_init_data(IEND);
            self.q_new = alloc_and_init_data_const(IEND, to_real(0.0));
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
        let mut e_new = unsafe { &mut *(self.e_new as *mut [Real; IEND]) };
        let mut q_new = unsafe { &mut *(self.q_new as *mut [Real; IEND]) };

        let p1: PreloadMut<[Real; IEND]> = preload_mut(&mut e_new);
        let p2: PreloadMut<[Real; IEND]> = preload_mut(&mut q_new);

        let mut e_new_reg = Region::<'_, _, Linear1D>::from(&p1);
        let mut q_new_reg = Region::<'_, _, Linear1D>::from(&p2);
        unsafe {
            offload! {
                kernel = energycalc1,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                    e_new_reg,
                    &*(self.e_old as *const [Real; IEND]),
                    &*(self.delvc as *const [Real; IEND]),
                    &*(self.p_old as *const [Real; IEND]),
                    &*(self.q_old as *const [Real; IEND]),
                    &*(self.work as *const [Real; IEND]),
                    IEND,
                ),
            };
            offload! {
                kernel = energycalc2,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                    &*(self.delvc as *const [Real; IEND]),
                    q_new_reg,
                    &*(self.comp_half_step as *const [Real; IEND]),
                    &*(self.p_half_step as *const [Real; IEND]),
                    e_new_reg,
                    &*(self.bvc as *const [Real; IEND]),
                    &*(self.pbvc as *const [Real; IEND]),
                    &*(self.ql_old as *const [Real; IEND]),
                    &*(self.qq_old as *const [Real; IEND]),
                    self.rho0,
                    IEND,
                ),
            };
            offload! {
                kernel = energycalc3,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                e_new_reg,
                    &*(self.delvc as *const [Real; IEND]),
                    &*(self.p_old as *const [Real; IEND]),
                    &*(self.q_old as *const [Real; IEND]),
                    &*(self.p_half_step as *const [Real; IEND]),
                    &*(self.q_new as *const [Real; IEND]),
                    IEND,
                ),
            };
            offload! {
                kernel = energycalc4,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                    e_new_reg,
                    &*(self.work as *const [Real; IEND]),
                    self.e_cut,
                    self.emin,
                    IEND,
                ),
            };
            offload! {
                kernel = energycalc5,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                    &*(self.delvc as *const [Real; IEND]),
                    &*(self.pbvc as *const [Real; IEND]),
                    e_new_reg,
                    &*(self.vnewc as *const [Real; IEND]),
                    &*(self.bvc as *const [Real; IEND]),
                    &*(self.p_new as *const [Real; IEND]),
                    &*(self.ql_old as *const [Real; IEND]),
                    &*(self.qq_old as *const [Real; IEND]),
                    &*(self.p_old as *const [Real; IEND]),
                    &*(self.q_old as *const [Real; IEND]),
                    &*(self.p_half_step as *const [Real; IEND]),
                    &*(self.q_new as *const [Real; IEND]),
                    self.rho0,
                    self.e_cut,
                    self.emin,
                    IEND,
                ),
            };
            offload! {
                kernel = energycalc6,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                    &*(self.delvc as *const [Real; IEND]),
                    &*(self.pbvc as *const [Real; IEND]),
                    e_new_reg,
                    &*(self.vnewc as *const [Real; IEND]),
                    &*(self.bvc as *const [Real; IEND]),
                    &*(self.p_new as *const [Real; IEND]),
                    q_new_reg,
                    &*(self.ql_old as *const [Real; IEND]),
                    &*(self.qq_old as *const [Real; IEND]),
                    self.rho0,
                    self.q_cut,
                    IEND,
                ),
            };
        }
    }

    fn update_checksum(&self) -> f64 {
        unsafe {
            calc_checksum(self.e_new as *const Real, IEND)
                + calc_checksum(self.q_new as *const Real, IEND)
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

#[cfg(not(target_os = "linux"))]
use crate::common::types::{Real, RealExt};

#[offload_kernel]
fn energycalc1(
    mut e_new: Region<Real, Linear1D>,
    e_old: &[Real; IEND],
    delvc: &[Real; IEND],
    p_old: &[Real; IEND],
    q_old: &[Real; IEND],
    work: &[Real; IEND],
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v) = e_new.get_mut() {
        *v = (*e_old)[i] - Real::from(0.5) * (*delvc)[i] * ((*p_old)[i] + (*q_old)[i])
            + Real::from(0.5) * (*work)[i];
    }
}

#[offload_kernel]
fn energycalc2(
    delvc: &[Real; IEND],
    mut q_new: Region<Real, Linear1D>,
    comp_half_step: &[Real; IEND],
    p_half_step: &[Real; IEND],
    mut e_new: Region<Real, Linear1D>,
    bvc: &[Real; IEND],
    pbvc: &[Real; IEND],
    ql_old: &[Real; IEND],
    qq_old: &[Real; IEND],
    rho0: Real,
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v1) = q_new.get_mut()
        && let Some(v2) = e_new.get_mut()
    {
        if ((*delvc)[i]).to_f64() > 0.0 {
            *v1 = Real::from(0.0);
        } else {
            let vhalf = Real::from(1.0) / (Real::from(1.0) + (*comp_half_step)[i]);
            let mut ssc =
                ((*pbvc)[i] * (*v2) + vhalf * vhalf * (*bvc)[i] * (*p_half_step)[i]) / rho0;
            if ssc.to_f64() <= 0.1111111e-36 {
                ssc = Real::from(0.3333333e-18);
            } else {
                ssc = ssc.sqrt();
            }
            *v1 = ssc * (*ql_old)[i] + (*qq_old)[i];
        }
    }
}

#[offload_kernel]
fn energycalc3(
    mut e_new: Region<Real, Linear1D>,
    delvc: &[Real; IEND],
    p_old: &[Real; IEND],
    q_old: &[Real; IEND],
    p_half_step: &[Real; IEND],
    q_new: &[Real; IEND],
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v) = e_new.get_mut() {
        *v += Real::from(0.5)
            * (*delvc)[i]
            * (Real::from(3.0) * ((*p_old)[i] + (*q_old)[i])
                - Real::from(4.0) * ((*p_half_step)[i] + (*q_new)[i]));
    }
}

#[offload_kernel]
fn energycalc4(
    mut e_new: Region<Real, Linear1D>,
    work: &[Real; IEND],
    e_cut: Real,
    emin: Real,
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v) = e_new.get_mut() {
        *v += Real::from(0.5) * (*work)[i];
        if (*v).abs() < e_cut {
            *v = Real::from(0.0);
        }
        if *v < emin {
            *v = emin;
        }
    }
}

#[offload_kernel]
fn energycalc5(
    delvc: &[Real; IEND],
    pbvc: &[Real; IEND],
    mut e_new: Region<Real, Linear1D>,
    vnewc: &[Real; IEND],
    bvc: &[Real; IEND],
    p_new: &[Real; IEND],
    ql_old: &[Real; IEND],
    qq_old: &[Real; IEND],
    p_old: &[Real; IEND],
    q_old: &[Real; IEND],
    p_half_step: &[Real; IEND],
    q_new: &[Real; IEND],
    rho0: Real,
    e_cut: Real,
    emin: Real,
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v) = e_new.get_mut() {
        let q_tilde = if ((*delvc)[i]).to_f64() > 0.0 {
            Real::from(0.0)
        } else {
            let mut ssc =
                ((*pbvc)[i] * (*v) + (*vnewc)[i] * (*vnewc)[i] * (*bvc)[i] * (*p_new)[i]) / rho0;
            if ssc.to_f64() <= 0.1111111e-36 {
                ssc = Real::from(0.3333333e-18);
            } else {
                ssc = ssc.sqrt();
            }
            ssc * (*ql_old)[i] + (*qq_old)[i]
        };
        *v -= (Real::from(7.0) * ((*p_old)[i] + (*q_old)[i])
            - Real::from(8.0) * ((*p_half_step)[i] + (*q_new)[i])
            + ((*p_new)[i] + q_tilde))
            * (*delvc)[i]
            / Real::from(6.0);
        if (*v).abs() < e_cut {
            *v = Real::from(0.0);
        }
        if *v < emin {
            *v = emin;
        }
    }
}

#[offload_kernel]
fn energycalc6(
    delvc: &[Real; IEND],
    pbvc: &[Real; IEND],
    mut e_new: Region<Real, Linear1D>,
    vnewc: &[Real; IEND],
    bvc: &[Real; IEND],
    p_new: &[Real; IEND],
    mut q_new: Region<Real, Linear1D>,
    ql_old: &[Real; IEND],
    qq_old: &[Real; IEND],
    rho0: Real,
    q_cut: Real,
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v1) = e_new.get_mut()
        && let Some(v2) = q_new.get_mut()
        && ((*delvc)[i]).to_f64() <= 0.0
    {
        let mut ssc =
            ((*pbvc)[i] * (*v1) + (*vnewc)[i] * (*vnewc)[i] * (*bvc)[i] * (*p_new)[i]) / rho0;
        if ssc.to_f64() <= 0.1111111e-36 {
            ssc = Real::from(0.3333333e-18);
        } else {
            ssc = ssc.sqrt();
        }
        *v2 = ssc * (*ql_old)[i] + (*qq_old)[i];
        if (*v2).abs() < q_cut {
            *v2 = Real::from(0.0);
        }
    }
}
