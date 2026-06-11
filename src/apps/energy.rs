#![allow(clippy::too_many_arguments)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]

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
use core::offload::offload::{preload, preload_mut, Preload, PreloadMut};

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data, alloc_and_init_data_const, calc_checksum, free, init_data_scalar,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{to_real, Real};

#[cfg(target_os = "linux")]
pub struct Energy<'a> {
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

    p_e_new: Option<PreloadMut<'a, [Real; IEND]>>,
    p_q_new: Option<PreloadMut<'a, [Real; IEND]>>,

    p_e_old: Option<Preload<'a, [Real; IEND]>>,
    p_delvc: Option<Preload<'a, [Real; IEND]>>,
    p_p_new: Option<Preload<'a, [Real; IEND]>>,
    p_p_old: Option<Preload<'a, [Real; IEND]>>,
    p_q_old: Option<Preload<'a, [Real; IEND]>>,
    p_work: Option<Preload<'a, [Real; IEND]>>,
    p_comp_half_step: Option<Preload<'a, [Real; IEND]>>,
    p_p_half_step: Option<Preload<'a, [Real; IEND]>>,
    p_bvc: Option<Preload<'a, [Real; IEND]>>,
    p_pbvc: Option<Preload<'a, [Real; IEND]>>,
    p_ql_old: Option<Preload<'a, [Real; IEND]>>,
    p_qq_old: Option<Preload<'a, [Real; IEND]>>,
    p_vnewc: Option<Preload<'a, [Real; IEND]>>,
}

#[cfg(target_os = "linux")]
impl<'a> Energy<'a> {
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

        p_e_new: None,
        p_q_new: None,

        p_e_old: None,
        p_delvc: None,
        p_p_new: None,
        p_p_old: None,
        p_q_old: None,
        p_work: None,
        p_comp_half_step: None,
        p_p_half_step: None,
        p_bvc: None,
        p_pbvc: None,
        p_ql_old: None,
        p_qq_old: None,
        p_vnewc: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for Energy<'a> {
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

        self.p_e_new = Some(preload_mut(unsafe {
            &mut *(self.e_new as *mut [Real; IEND])
        }));
        self.p_q_new = Some(preload_mut(unsafe {
            &mut *(self.q_new as *mut [Real; IEND])
        }));

        self.p_e_old = Some(preload(unsafe {
            &*(self.e_old as *const [Real; IEND])
        }));
        self.p_delvc = Some(preload(unsafe {
            &*(self.delvc as *const [Real; IEND])
        }));
        self.p_p_new = Some(preload(unsafe {
            &*(self.p_new as *const [Real; IEND])
        }));
        self.p_p_old = Some(preload(unsafe {
            &*(self.p_old as *const [Real; IEND])
        }));
        self.p_q_old = Some(preload(unsafe {
            &*(self.q_old as *const [Real; IEND])
        }));
        self.p_work = Some(preload(unsafe {
            &*(self.work as *const [Real; IEND])
        }));
        self.p_comp_half_step = Some(preload(unsafe {
            &*(self.comp_half_step as *const [Real; IEND])
        }));
        self.p_p_half_step = Some(preload(unsafe {
            &*(self.p_half_step as *const [Real; IEND])
        }));
        self.p_bvc = Some(preload(unsafe {
            &*(self.bvc as *const [Real; IEND])
        }));
        self.p_pbvc = Some(preload(unsafe {
            &*(self.pbvc as *const [Real; IEND])
        }));
        self.p_ql_old = Some(preload(unsafe {
            &*(self.ql_old as *const [Real; IEND])
        }));
        self.p_qq_old = Some(preload(unsafe {
            &*(self.qq_old as *const [Real; IEND])
        }));
        self.p_vnewc = Some(preload(unsafe {
            &*(self.vnewc as *const [Real; IEND])
        }));
    }

    fn run_kernel(&mut self) {
        let p_e_new = self
            .p_e_new
            .as_ref()
            .expect("ENERGY e_new was not preloaded");

        let p_q_new = self
            .p_q_new
            .as_ref()
            .expect("ENERGY q_new was not preloaded");

        self.p_e_old
            .as_ref()
            .expect("ENERGY e_old was not preloaded");
        self.p_delvc
            .as_ref()
            .expect("ENERGY delvc was not preloaded");
        self.p_p_new
            .as_ref()
            .expect("ENERGY p_new was not preloaded");
        self.p_p_old
            .as_ref()
            .expect("ENERGY p_old was not preloaded");
        self.p_q_old
            .as_ref()
            .expect("ENERGY q_old was not preloaded");
        self.p_work
            .as_ref()
            .expect("ENERGY work was not preloaded");
        self.p_comp_half_step
            .as_ref()
            .expect("ENERGY comp_half_step was not preloaded");
        self.p_p_half_step
            .as_ref()
            .expect("ENERGY p_half_step was not preloaded");
        self.p_bvc
            .as_ref()
            .expect("ENERGY bvc was not preloaded");
        self.p_pbvc
            .as_ref()
            .expect("ENERGY pbvc was not preloaded");
        self.p_ql_old
            .as_ref()
            .expect("ENERGY ql_old was not preloaded");
        self.p_qq_old
            .as_ref()
            .expect("ENERGY qq_old was not preloaded");
        self.p_vnewc
            .as_ref()
            .expect("ENERGY vnewc was not preloaded");

        let mut e_new_reg = Region::<'_, _, Linear1D>::from(p_e_new);
        let mut q_new_reg = Region::<'_, _, Linear1D>::from(p_q_new);

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

    fn tear_down(&mut self) -> f64 {
        drop(self.p_e_new.take());
        drop(self.p_q_new.take());

        drop(self.p_e_old.take());
        drop(self.p_delvc.take());
        drop(self.p_p_new.take());
        drop(self.p_p_old.take());
        drop(self.p_q_old.take());
        drop(self.p_work.take());
        drop(self.p_comp_half_step.take());
        drop(self.p_p_half_step.take());
        drop(self.p_bvc.take());
        drop(self.p_pbvc.take());
        drop(self.p_ql_old.take());
        drop(self.p_qq_old.take());
        drop(self.p_vnewc.take());
    
        let ck = self.update_checksum();

        unsafe {
            if !self.e_new.is_null() {
                free(self.e_new);
                self.e_new = core::ptr::null_mut();
            }

            if !self.e_old.is_null() {
                free(self.e_old);
                self.e_old = core::ptr::null_mut();
            }

            if !self.delvc.is_null() {
                free(self.delvc);
                self.delvc = core::ptr::null_mut();
            }

            if !self.p_new.is_null() {
                free(self.p_new);
                self.p_new = core::ptr::null_mut();
            }

            if !self.p_old.is_null() {
                free(self.p_old);
                self.p_old = core::ptr::null_mut();
            }

            if !self.q_new.is_null() {
                free(self.q_new);
                self.q_new = core::ptr::null_mut();
            }

            if !self.q_old.is_null() {
                free(self.q_old);
                self.q_old = core::ptr::null_mut();
            }

            if !self.work.is_null() {
                free(self.work);
                self.work = core::ptr::null_mut();
            }

            if !self.comp_half_step.is_null() {
                free(self.comp_half_step);
                self.comp_half_step = core::ptr::null_mut();
            }

            if !self.p_half_step.is_null() {
                free(self.p_half_step);
                self.p_half_step = core::ptr::null_mut();
            }

            if !self.bvc.is_null() {
                free(self.bvc);
                self.bvc = core::ptr::null_mut();
            }

            if !self.pbvc.is_null() {
                free(self.pbvc);
                self.pbvc = core::ptr::null_mut();
            }

            if !self.ql_old.is_null() {
                free(self.ql_old);
                self.ql_old = core::ptr::null_mut();
            }

            if !self.qq_old.is_null() {
                free(self.qq_old);
                self.qq_old = core::ptr::null_mut();
            }

            if !self.vnewc.is_null() {
                free(self.vnewc);
                self.vnewc = core::ptr::null_mut();
            }
        }
        ck
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
