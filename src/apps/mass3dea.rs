pub const MEA_D1D: usize = 4;
pub const MEA_Q1D: usize = 5;
pub const NE_DEFAULT: usize = 125;
const DEFAULT_REPS: u32 = 1;

const EA_MAT: usize = MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D * MEA_D1D;

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_idx_x as block_idx_x, _syncthreads, _thread_idx_x as thread_idx_x,
    _thread_idx_y as thread_idx_y, _thread_idx_z as thread_idx_z,
};
#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn thread_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.y"]
    fn thread_idx_y() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.z"]
    fn thread_idx_z() -> i32;
    #[link_name = "llvm.amdgcn.s.barrier"]
    fn _syncthreads();
}

#[cfg(target_os = "linux")]
use crate::common::data_utils::{alloc_and_init_data_const, calc_checksum, free};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Mass3DEA {
    ne: usize,
    b: *mut Real,
    d: *mut Real,
    m: *mut Real,
}

#[cfg(target_os = "linux")]
impl Mass3DEA {
    pub const INIT: Self = Mass3DEA {
        ne: 0,
        b: core::ptr::null_mut(),
        d: core::ptr::null_mut(),
        m: core::ptr::null_mut(),
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Mass3DEA {
    fn name(&self) -> &'static str {
        kernel_name!("MASS3DEA")
    }

    fn default_problem_size(&self) -> usize {
        NE_DEFAULT * EA_MAT
    }

    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        self.ne = 125;

        unsafe {
            self.b = alloc_and_init_data_const(MEA_Q1D * MEA_D1D, to_real(1.0));
            self.d = alloc_and_init_data_const(MEA_Q1D * MEA_Q1D * MEA_Q1D * self.ne, to_real(1.0));
            self.m = alloc_and_init_data_const(EA_MAT * self.ne, to_real(0.0));
        }
    }

    fn run_kernel(&mut self) {
        let ne = self.ne;
        core::intrinsics::offload::<_, _, ()>(
            _mass3dea,
            [ne as u32, 1, 1],
            [MEA_D1D as u32, MEA_D1D as u32, MEA_D1D as u32],
            (
                self.b as *const [Real; MEA_Q1D * MEA_D1D],
                self.d as *const [Real; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_DEFAULT],
                self.m as *mut [Real; EA_MAT * NE_DEFAULT],
                ne,
            ),
        );
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.m as *const Real, EA_MAT * self.ne) }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.b);
            self.b = core::ptr::null_mut();
            free(self.d);
            self.d = core::ptr::null_mut();
            free(self.m);
            self.m = core::ptr::null_mut();
        }
        self.ne = 0;
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _mass3dea(
        B: *const [Real; MEA_Q1D * MEA_D1D],
        D: *const [Real; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_DEFAULT],
        M: *mut [Real; EA_MAT * NE_DEFAULT],
        NE: usize,
    );
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _mass3dea(
    B: *const [Real; MEA_Q1D * MEA_D1D],
    D: *const [Real; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_DEFAULT],
    M: *mut [Real; EA_MAT * NE_DEFAULT],
    NE: usize,
) {
    let e = block_idx_x() as usize;

    if e < NE {
        let tx = thread_idx_x() as usize;
        let ty = thread_idx_y() as usize;
        let tz = thread_idx_z() as usize;

        // TODO(Sa4dUs): RAJA_TEAM_SHARED
        let mut s_B = [[0.0; MEA_D1D]; MEA_Q1D];
        // TODO(Sa4dUs): RAJA_TEAM_SHARED
        let mut s_D = [[[0.0; MEA_Q1D]; MEA_Q1D]; MEA_Q1D];

        if tz == 0 && tx < MEA_D1D && ty < MEA_Q1D {
            let q = ty;
            let d = tx;
            s_B[q][d] = (*B)[q + MEA_Q1D * d];
        }

        if tx < MEA_Q1D && ty < MEA_Q1D && tz < MEA_Q1D {
            let k1 = tx;
            let k2 = ty;
            let k3 = tz;
            let d_idx =
                k1 + MEA_Q1D * k2 + (MEA_Q1D * MEA_Q1D) * k3 + (MEA_Q1D * MEA_Q1D * MEA_Q1D) * e;
            s_D[k1][k2][k3] = (*D)[d_idx];
        }

        _syncthreads();

        if tx < MEA_D1D && ty < MEA_D1D && tz < MEA_D1D {
            let i1 = tx;
            let i2 = ty;
            let i3 = tz;

            for j1 in 0..MEA_D1D {
                for j2 in 0..MEA_D1D {
                    for j3 in 0..MEA_D1D {
                        let mut val: Real = 0.0;

                        for k1 in 0..MEA_Q1D {
                            let b_val1 = s_B[k1][i1] * s_B[k1][j1];
                            for k2 in 0..MEA_Q1D {
                                let b_val2 = s_B[k2][i2] * s_B[k2][j2];
                                for k3 in 0..MEA_Q1D {
                                    let b_val3 = s_B[k3][i3] * s_B[k3][j3];

                                    val += b_val1 * b_val2 * b_val3 * s_D[k1][k2][k3];
                                }
                            }
                        }

                        let m_idx = i1
                            + MEA_D1D
                                * (i2
                                    + MEA_D1D
                                        * (i3
                                            + MEA_D1D
                                                * (j1
                                                    + MEA_D1D
                                                        * (j2 + MEA_D1D * (j3 + MEA_D1D * e)))));
                        (*M)[m_idx] = val;
                    }
                }
            }
        }
    }
}
