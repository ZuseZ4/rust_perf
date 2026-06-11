pub const N_DEFAULT: usize = 1_000_000;
const DEFAULT_REPS: u32 = 700;

use core::offload::offload_kernel;
use rustc_offload_frontend::partition::{PartitioningStrategy, Region, Stride1D};

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
pub struct Pressure<'a> {
    n: usize,
    compression: *mut Real,
    bvc: *mut Real,
    p_new: *mut Real,
    e_old: *mut Real,
    vnewc: *mut Real,
    cls: Real,
    p_cut: Real,
    pmin: Real,
    eosvmax: Real,

    p_compression: Option<Preload<'a, [Real; N_DEFAULT]>>,
    p_bvc: Option<PreloadMut<'a, [Real; N_DEFAULT]>>,
    p_p_new: Option<PreloadMut<'a, [Real; N_DEFAULT]>>,
    p_e_old: Option<Preload<'a, [Real; N_DEFAULT]>>,
    p_vnewc: Option<Preload<'a, [Real; N_DEFAULT]>>,
}

#[cfg(target_os = "linux")]
impl<'a> Pressure<'a> {
    pub const INIT: Self = Pressure {
        n: 0,
        compression: core::ptr::null_mut(),
        bvc: core::ptr::null_mut(),
        p_new: core::ptr::null_mut(),
        e_old: core::ptr::null_mut(),
        vnewc: core::ptr::null_mut(),
        cls: to_real(0.33),
        p_cut: to_real(1.0e-7),
        pmin: to_real(1.0e-10),
        eosvmax: to_real(1.0e+10),

        p_compression: None,
        p_bvc: None,
        p_p_new: None,
        p_e_old: None,
        p_vnewc: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for Pressure<'a> {
    fn name(&self) -> &'static str {
        kernel_name!("PRESSURE")
    }

    fn default_problem_size(&self) -> usize {
        N_DEFAULT
    }

    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        self.n = self.default_problem_size();

        assert_eq!(
            self.n, N_DEFAULT,
            "PRESSURE const-array preload version requires n = {N_DEFAULT}, got {}",
            self.n
        );

        unsafe {
            self.compression = alloc_and_init_data(self.n);
            self.bvc = alloc_and_init_data(self.n);
            self.p_new = alloc_and_init_data_const(self.n, to_real(0.0));
            self.e_old = alloc_and_init_data(self.n);
            self.vnewc = alloc_and_init_data(self.n);

            self.cls = init_data_scalar();
            self.p_cut = init_data_scalar();
            self.pmin = init_data_scalar();
            self.eosvmax = init_data_scalar();
        }

        let compression_ref: &'a [Real; N_DEFAULT] =
            unsafe { &*(self.compression as *const [Real; N_DEFAULT]) };

        let bvc_ref: &'a mut [Real; N_DEFAULT] =
            unsafe { &mut *(self.bvc as *mut [Real; N_DEFAULT]) };

        let p_new_ref: &'a mut [Real; N_DEFAULT] =
            unsafe { &mut *(self.p_new as *mut [Real; N_DEFAULT]) };

        let e_old_ref: &'a [Real; N_DEFAULT] =
            unsafe { &*(self.e_old as *const [Real; N_DEFAULT]) };

        let vnewc_ref: &'a [Real; N_DEFAULT] =
            unsafe { &*(self.vnewc as *const [Real; N_DEFAULT]) };

        self.p_compression = Some(preload(compression_ref));
        self.p_bvc = Some(preload_mut(bvc_ref));
        self.p_p_new = Some(preload_mut(p_new_ref));
        self.p_e_old = Some(preload(e_old_ref));
        self.p_vnewc = Some(preload(vnewc_ref));
    }

    fn run_kernel(&mut self) {
        let n = self.n;
        let grid = [n.div_ceil(256) as u32, 1, 1];
        let block = [256, 1, 1];

        self.p_compression
            .as_ref()
            .expect("PRESSURE compression was not preloaded");

        let p_bvc = self.p_bvc.as_ref().expect("PRESSURE bvc was not preloaded");

        let p_p_new = self
            .p_p_new
            .as_ref()
            .expect("PRESSURE p_new was not preloaded");

        self.p_e_old
            .as_ref()
            .expect("PRESSURE e_old was not preloaded");

        self.p_vnewc
            .as_ref()
            .expect("PRESSURE vnewc was not preloaded");

        let mut bvc_reg = Region::<'_, _, Stride1D<256>>::from(p_bvc);
        let mut p_new_reg = Region::<'_, _, Stride1D<256>>::from(p_p_new);

        offload! {
            kernel = pressure_calc1,
            grid_dim = grid,
            block_dim = block,
            args = (
                bvc_reg,
                self.compression as *const [Real; N_DEFAULT],
                self.cls,
                n,
            ),
        };

        offload! {
            kernel = pressure_calc2,
            grid_dim = grid,
            block_dim = block,
            args = (
                p_new_reg,
                self.bvc as *const [Real; N_DEFAULT],
                self.e_old as *const [Real; N_DEFAULT],
                self.vnewc as *const [Real; N_DEFAULT],
                self.p_cut,
                self.eosvmax,
                self.pmin,
                n,
            ),
        };
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.p_new as *const Real, self.n) }
    }

    fn tear_down(&mut self) -> f64 {
        // Drop preloads first. Dropping p_bvc / p_p_new makes device writes visible on host.
        drop(self.p_bvc.take());
        drop(self.p_p_new.take());

        drop(self.p_compression.take());
        drop(self.p_e_old.take());
        drop(self.p_vnewc.take());

        let checksum = self.update_checksum();

        unsafe {
            if !self.compression.is_null() {
                free(self.compression);
                self.compression = core::ptr::null_mut();
            }

            if !self.bvc.is_null() {
                free(self.bvc);
                self.bvc = core::ptr::null_mut();
            }

            if !self.p_new.is_null() {
                free(self.p_new);
                self.p_new = core::ptr::null_mut();
            }

            if !self.e_old.is_null() {
                free(self.e_old);
                self.e_old = core::ptr::null_mut();
            }

            if !self.vnewc.is_null() {
                free(self.vnewc);
                self.vnewc = core::ptr::null_mut();
            }
        }

        self.n = 0;

        checksum
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[offload_kernel]
fn pressure_calc1(
    mut bvc: Region<Real, Stride1D<256>>,
    compression: *const [Real; N_DEFAULT],
    cls: Real,
    n: usize,
) {
    let i = Stride1D::<256>::index();

    if let Some(v) = bvc.get_mut() {
        unsafe {
            *v = cls * ((*compression)[i] + Real::from(1.0));
        }
    }
}

#[offload_kernel]
fn pressure_calc2(
    mut p_new: Region<Real, Stride1D<256>>,
    bvc: *const [Real; N_DEFAULT],
    e_old: *const [Real; N_DEFAULT],
    vnewc: *const [Real; N_DEFAULT],
    p_cut: Real,
    eosvmax: Real,
    pmin: Real,
    n: usize,
) {
    let i = Stride1D::<256>::index();

    if let Some(v) = p_new.get_mut() {
        unsafe {
            let mut p = (*bvc)[i] * (*e_old)[i];

            if p.abs() < p_cut {
                p = Real::from(0.0);
            }

            if (*vnewc)[i] >= eosvmax {
                p = Real::from(0.0);
            }

            if p < pmin {
                p = pmin;
            }

            *v = p;
        }
    }
}
