const DEFAULT_PROBLEM_SIZE: usize = 1_000_000;
const DEFAULT_REPS: u32 = 160;

const IEND: usize = DEFAULT_PROBLEM_SIZE;
pub const COEFFLEN: usize = 16;
const THREADS_PER_BLOCK: u32 = 256;
const BLOCKS: u32 = (IEND as u32).div_ceil(THREADS_PER_BLOCK);

use core::offload::offload_kernel;
use rustc_offload_frontend::partition::{Linear1D, PartitioningStrategy, Region};

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{PreloadMut, preload_mut};

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data_const, alloc_and_init_data_rand_value, calc_checksum, free,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Fir {
    m_in: *mut Real,
    m_out: *mut Real,
    coeff: [Real; COEFFLEN],
}

#[cfg(target_os = "linux")]
impl Fir {
    pub const INIT: Self = Fir {
        m_in: core::ptr::null_mut(),
        m_out: core::ptr::null_mut(),
        coeff: [const { to_real(0.0) }; COEFFLEN],
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Fir {
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
        self.coeff = [
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

        unsafe {
            self.m_in = alloc_and_init_data_rand_value(IEND + COEFFLEN - 1);
            self.m_out = alloc_and_init_data_const(IEND, to_real(0.0));
        }
    }

    fn run_kernel(&mut self) {
        let mut m_out = unsafe { &mut *(self.m_out as *mut [Real; IEND]) };
        let p: PreloadMut<[Real; IEND]> = preload_mut(&mut m_out);
        let mut m_out_reg = Region::<'_, _, Linear1D>::from(&p);
        offload! {
            kernel = fir,
            grid_dim = [BLOCKS, 1, 1],
            block_dim = [THREADS_PER_BLOCK, 1, 1],
            args = (
                m_out_reg,
                unsafe { &*(self.m_in as *const [Real; IEND + COEFFLEN]) },
                unsafe { &self.coeff as &[Real; COEFFLEN] },
                IEND,
            ),
        };
        drop(p);
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.m_out as *const Real, IEND) }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.m_in);
            self.m_in = core::ptr::null_mut();
            free(self.m_out);
            self.m_out = core::ptr::null_mut();
        }
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[offload_kernel]
fn fir(
    mut m_out: Region<Real, Linear1D>,
    m_in: &[Real; IEND + COEFFLEN],
    coeff: &[Real; COEFFLEN],
    iend: usize,
) {
    let i = Linear1D::index();
    if let Some(v) = m_out.get_mut() {
        let mut sum: Real = Real::from(0.0);
        let mut j = 0;
        while j < COEFFLEN {
            unsafe {
                sum += (*coeff)[j] * (*m_in)[i + j];
            }
            j += 1;
        }
        *v = sum;
    }
}
