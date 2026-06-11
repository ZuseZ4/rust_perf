pub const NUM_D: usize = 64;
pub const NUM_G: usize = 32;
pub const NUM_M: usize = 25;

const DEFAULT_PROBLEM_SIZE: usize = 1_000_000;
const DEFAULT_REPS: u32 = 50;

const DEFAULT_NUM_Z: usize = (DEFAULT_PROBLEM_SIZE + (NUM_D * NUM_G) / 2) / (NUM_D * NUM_G);

const PHILEN: usize = NUM_M * NUM_G * DEFAULT_NUM_Z;
const ELLLEN: usize = NUM_D * NUM_M;
const PSILEN: usize = NUM_D * NUM_G * DEFAULT_NUM_Z;

use core::offload::offload_kernel;
use rustc_offload_frontend::partition::{PartitioningStrategy, Region, Stride3D};

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{preload, preload_mut, Preload, PreloadMut};

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
pub struct LTimes<'a> {
    num_z: usize,

    phidat: *mut Real,
    elldat: *mut Real,
    psidat: *mut Real,

    p_phidat: Option<PreloadMut<'a, [Real; PHILEN]>>,
    p_elldat: Option<Preload<'a, [Real; ELLLEN]>>,
    p_psidat: Option<Preload<'a, [Real; PSILEN]>>,
}

#[cfg(target_os = "linux")]
impl<'a> LTimes<'a> {
    pub const INIT: Self = LTimes {
        num_z: 0,

        phidat: core::ptr::null_mut(),
        elldat: core::ptr::null_mut(),
        psidat: core::ptr::null_mut(),

        p_phidat: None,
        p_elldat: None,
        p_psidat: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for LTimes<'a> {
    fn name(&self) -> &'static str {
        kernel_name!("LTIMES")
    }

    fn default_problem_size(&self) -> usize {
        let num_z_default = (DEFAULT_PROBLEM_SIZE + (NUM_D * NUM_G) / 2) / (NUM_D * NUM_G);
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

        assert_eq!(
            self.num_z, DEFAULT_NUM_Z,
            "LTIMES const-array preload version requires num_z = {DEFAULT_NUM_Z}, got {}",
            self.num_z,
        );

        assert_eq!(
            philen, PHILEN,
            "LTIMES const-array preload version requires PHILEN = {PHILEN}, got {philen}",
        );

        assert_eq!(
            elllen, ELLLEN,
            "LTIMES const-array preload version requires ELLLEN = {ELLLEN}, got {elllen}",
        );

        assert_eq!(
            psilen, PSILEN,
            "LTIMES const-array preload version requires PSILEN = {PSILEN}, got {psilen}",
        );

        unsafe {
            self.phidat = alloc_and_init_data_const(philen, to_real(0.0));
            self.elldat = alloc_and_init_data(elllen);
            self.psidat = alloc_and_init_data(psilen);
        }

        let phidat_ref: &'a mut [Real; PHILEN] =
            unsafe { &mut *(self.phidat as *mut [Real; PHILEN]) };

        let elldat_ref: &'a [Real; ELLLEN] = unsafe { &*(self.elldat as *const [Real; ELLLEN]) };

        let psidat_ref: &'a [Real; PSILEN] = unsafe { &*(self.psidat as *const [Real; PSILEN]) };

        self.p_phidat = Some(preload_mut(phidat_ref));
        self.p_elldat = Some(preload(elldat_ref));
        self.p_psidat = Some(preload(psidat_ref));
    }

    fn run_kernel(&mut self) {
        let num_z = self.num_z;

        let m_block = 32;
        let g_block = 8;
        let z_block = 1;

        let grid_x = NUM_M.div_ceil(m_block);
        let grid_y = NUM_G.div_ceil(g_block);
        let grid_z = num_z.div_ceil(z_block);

        let p_phidat = self
            .p_phidat
            .as_ref()
            .expect("LTIMES phidat was not preloaded");

        let _p_elldat = self
            .p_elldat
            .as_ref()
            .expect("LTIMES elldat was not preloaded");

        let _p_psidat = self
            .p_psidat
            .as_ref()
            .expect("LTIMES psidat was not preloaded");

        let mut phidat_reg = Region::<'_, _, Stride3D<32, 8, 1, 25, 32>>::from(p_phidat);

        let elldat_ref: &[Real; ELLLEN] = unsafe { &*(self.elldat as *const [Real; ELLLEN]) };

        let psidat_ref: &[Real; PSILEN] = unsafe { &*(self.psidat as *const [Real; PSILEN]) };

        offload! {
            kernel = ltimes,
            grid_dim = [grid_x as u32, grid_y as u32, grid_z as u32],
            block_dim = [m_block as u32, g_block as u32, z_block as u32],
            args = (
                phidat_reg,
                elldat_ref,
                psidat_ref,
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
        // Drop output mapping first, so device writes are copied back.
        drop(self.p_phidat.take());

        let checksum = self.update_checksum();

        // Drop read-only mappings before freeing host allocations.
        drop(self.p_elldat.take());
        drop(self.p_psidat.take());

        unsafe {
            if !self.phidat.is_null() {
                free(self.phidat);
                self.phidat = core::ptr::null_mut();
            }

            if !self.elldat.is_null() {
                free(self.elldat);
                self.elldat = core::ptr::null_mut();
            }

            if !self.psidat.is_null() {
                free(self.psidat);
                self.psidat = core::ptr::null_mut();
            }
        }

        self.num_z = 0;

        checksum
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[offload_kernel]
fn ltimes(
    mut phi: Region<Real, Stride3D<32, 8, 1, 25, 32>>,
    ell: &[Real; ELLLEN],
    psi: &[Real; PSILEN],
    num_d: usize,
    num_m: usize,
    num_g: usize,
    num_z: usize,
) {
    let m = unsafe { (block_idx_x() * 32 + thread_idx_x()) as usize };
    let g = unsafe { (block_idx_y() * 8 + thread_idx_y()) as usize };
    let z = unsafe { (block_idx_z() * 1 + thread_idx_z()) as usize };

    if m < num_m && g < num_g && z < num_z {
        if let Some(v) = phi.get_mut() {
            for d in 0..num_d {
                let ell_idx = d + num_d * m;
                let psi_idx = d + num_d * (g + num_g * z);

                unsafe {
                    *v += *ell.get_unchecked(ell_idx) * *psi.get_unchecked(psi_idx);
                }
            }
        }
    }
}
