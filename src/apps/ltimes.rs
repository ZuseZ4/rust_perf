pub const NUM_D: usize = 64;
pub const NUM_G: usize = 32;
pub const NUM_M: usize = 25;
const DEFAULT_REPS: u32 = 50;

use core::offload::offload_kernel;
use rustc_offload_frontend::partition::{PartitioningStrategy, Region, Stride3D};

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{preload_mut, PreloadMut};

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{
    workgroup_id_x as block_idx_x, workgroup_id_y as block_idx_y, workgroup_id_z as block_idx_z,
    workitem_id_x as thread_idx_x, workitem_id_y as thread_idx_y, workitem_id_z as thread_idx_z,
};
#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_idx_x as block_idx_x, _block_idx_y as block_idx_y, _block_idx_z as block_idx_z,
    _thread_idx_x as thread_idx_x, _thread_idx_y as thread_idx_y, _thread_idx_z as thread_idx_z,
};

#[cfg(target_os = "linux")]
use libc::printf;

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data, alloc_and_init_data_const, calc_checksum, free,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{to_real, Real};

#[cfg(target_os = "linux")]
pub struct LTimes {
    num_z: usize,
    phidat: *mut Real,
    elldat: *mut Real,
    psidat: *mut Real,
}

#[cfg(target_os = "linux")]
impl LTimes {
    pub const INIT: Self = LTimes {
        num_z: 0,
        phidat: core::ptr::null_mut(),
        elldat: core::ptr::null_mut(),
        psidat: core::ptr::null_mut(),
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for LTimes {
    fn name(&self) -> &'static str {
        kernel_name!("LTIMES")
    }

    fn default_problem_size(&self) -> usize {
        let num_z_default = (1_000_000 + (NUM_D * NUM_G) / 2) / (NUM_D * NUM_G);
        num_z_default * NUM_D * NUM_G
    }

    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        let prob_size = self.default_problem_size();
        self.num_z = (prob_size + (NUM_D * NUM_G) / 2) / (NUM_D * NUM_G);

        let philen = NUM_M * NUM_G * self.num_z;
        let elllen = NUM_D * NUM_M;
        let psilen = NUM_D * NUM_G * self.num_z;

        unsafe {
            self.phidat = alloc_and_init_data_const(philen, to_real(0.0));
            self.elldat = alloc_and_init_data(elllen);
            self.psidat = alloc_and_init_data(psilen);
        }
    }

    fn run_kernel(&mut self) {
        let num_z = self.num_z;

        let m_block = 32;
        let g_block = 8;
        let z_block = 1;

        let grid_x = NUM_M.div_ceil(m_block);
        let grid_y = NUM_G.div_ceil(g_block);
        let grid_z = num_z.div_ceil(z_block);

        let mut phidat = unsafe { &mut *(self.phidat as *mut [Real; 390400]) };
        let p: PreloadMut<[Real; 390400]> = preload_mut(&mut phidat);
        let mut phidat_reg = Region::<'_, _, Stride3D<32, 8, 1, 25, 32>>::from(&p);

        offload! {
            kernel = ltimes,
            grid_dim = [grid_x as u32, grid_y as u32, grid_z as u32],
            block_dim = [m_block as u32, g_block as u32, z_block as u32],
            args = (
                phidat_reg,
                self.elldat as *const [Real; 1600],
                self.psidat as *const [Real; 999424],
                NUM_D,
                NUM_M,
                NUM_G,
                num_z,
            ),
        };
    }

    fn update_checksum(&self) -> f64 {
        let philen = NUM_M * NUM_G * self.num_z;
        unsafe { calc_checksum(self.phidat as *const Real, philen) }
    }

    fn tear_down(&mut self) -> f64 {
        unsafe {
            free(self.phidat);
            free(self.elldat);
            free(self.psidat);
            self.phidat = core::ptr::null_mut();
            self.elldat = core::ptr::null_mut();
            self.psidat = core::ptr::null_mut();
        }
        self.num_z = 0;
        self.update_checksum()
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[offload_kernel]
fn ltimes(
    mut phi: Region<Real, Stride3D<32, 8, 1, 25, 32>>,
    ell: &[Real; 1600],
    psi: &[Real; 999424],
    num_d: usize,
    num_m: usize,
    num_g: usize,
    num_z: usize,
) {
    let num_d = NUM_D;
    let num_g = NUM_G;

    let m = unsafe { (block_idx_x() * 32 + thread_idx_x()) as usize };
    let g = unsafe { (block_idx_y() * 8 + thread_idx_y()) as usize };
    let z = unsafe { (block_idx_z() * 1 + thread_idx_z()) as usize };

    if let Some(v) = phi.get_mut() {
        for d in 0..num_d {
            let ell_idx = d + num_d * m;
            let psi_idx = d + num_d * (g + num_g * z);

            *v += ell[ell_idx] * psi[psi_idx];
        }
    }
}
