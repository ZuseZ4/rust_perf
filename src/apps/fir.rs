#![feature(stmt_expr_attributes)]
const DEFAULT_PROBLEM_SIZE: usize = 1_000_000;
const DEFAULT_REPS: u32 = 160;

const IEND: usize = DEFAULT_PROBLEM_SIZE;
pub const COEFFLEN: usize = 16;
const INLEN: usize = IEND + COEFFLEN - 1;

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
    alloc_and_init_data_const, alloc_and_init_data_rand_value, calc_checksum, free,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{to_real, Real};

#[cfg(target_os = "linux")]
pub struct Fir<'a> {
    m_in: *mut Real,
    m_out: *mut Real,
    coeff: *mut Real,

    p_m_in: Option<Preload<'a, [Real; INLEN]>>,
    p_m_out: Option<PreloadMut<'a, [Real; IEND]>>,
    p_coeff: Option<Preload<'a, [Real; COEFFLEN]>>,
}

#[cfg(target_os = "linux")]
impl<'a> Fir<'a> {
    pub const INIT: Self = Fir {
        m_in: core::ptr::null_mut(),
        m_out: core::ptr::null_mut(),
        coeff: core::ptr::null_mut(),

        p_m_in: None,
        p_m_out: None,
        p_coeff: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for Fir<'a> {
    fn name(&self) -> &'static str {
        kernel_name!("FIR")
    }

    fn default_problem_size(&self) -> usize {
        DEFAULT_PROBLEM_SIZE
    }

    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        unsafe {
            self.m_in = alloc_and_init_data_rand_value(INLEN);
            self.m_out = alloc_and_init_data_const(IEND, to_real(0.0));

            // Allocate coeff on the heap as well, so the Preload handle does not
            // borrow from inside `self`.
            //
            // This random-inits first, then we overwrite with the hard-coded FIR
            // coefficients below.
            self.coeff = alloc_and_init_data_rand_value(COEFFLEN);

            let coeff: &mut [Real; COEFFLEN] = &mut *(self.coeff as *mut [Real; COEFFLEN]);

            *coeff = [
                to_real(3.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(3.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(3.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(-1.0),
                to_real(3.0),
            ];
        }

        let m_in_ref: &'a [Real; INLEN] = unsafe { &*(self.m_in as *const [Real; INLEN]) };

        let m_out_ref: &'a mut [Real; IEND] = unsafe { &mut *(self.m_out as *mut [Real; IEND]) };

        let coeff_ref: &'a [Real; COEFFLEN] = unsafe { &*(self.coeff as *const [Real; COEFFLEN]) };

        self.p_m_in = Some(preload(m_in_ref));
        self.p_m_out = Some(preload_mut(m_out_ref));
        self.p_coeff = Some(preload(coeff_ref));
    }

    fn run_kernel(&mut self) {
        let Some(_p_m_in) = self.p_m_in.as_ref() else {
            return;
        };
        let Some(p_m_out) = self.p_m_out.as_ref() else {
            return;
        };
        let Some(_p_coeff) = self.p_coeff.as_ref() else {
            return;
        };

        let mut m_out_reg = Region::<'_, _, Linear1D>::from(p_m_out);

        let m_in_ref: &[Real; INLEN] = unsafe { &*(self.m_in as *const [Real; INLEN]) };

        let coeff_ref: &[Real; COEFFLEN] = unsafe { &*(self.coeff as *const [Real; COEFFLEN]) };

        offload! {
            kernel = fir,
            grid_dim = [BLOCKS, 1, 1],
            block_dim = [THREADS_PER_BLOCK, 1, 1],
            args = (
                m_out_reg,
                m_in_ref,
                coeff_ref,
                IEND,
            ),
        };
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.m_out as *const Real, IEND) }
    }

    fn tear_down(&mut self) -> f64 {
        // Drop all offload mappings before freeing the corresponding host memory.
        drop(self.p_m_out.take());
        drop(self.p_m_in.take());
        drop(self.p_coeff.take());
        let ck = self.update_checksum();

        unsafe {
            if !self.m_in.is_null() {
                free(self.m_in);
                self.m_in = core::ptr::null_mut();
            }

            if !self.m_out.is_null() {
                free(self.m_out);
                self.m_out = core::ptr::null_mut();
            }

            if !self.coeff.is_null() {
                free(self.coeff);
                self.coeff = core::ptr::null_mut();
            }
        }
        ck
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[offload_kernel]
fn fir(
    mut m_out: Region<Real, Linear1D>,
    m_in: &[Real; INLEN],
    coeff: &[Real; COEFFLEN],
    iend: usize,
) {
    let i = Linear1D::index();

    if i < iend {
        if let Some(v) = m_out.get_mut() {
            let mut sum: Real = Real::from(0.0);
            let mut j = 0;

            //#[unroll(2)]
            //for j in 0..COEFFLEN {
            //#[rustc_unroll(4)]
            while j < COEFFLEN {
                unsafe {
                    sum += (*coeff)[j] * (*m_in)[i + j];
                }
                j += 1;
            }

            *v = sum;
        }
    }
}
