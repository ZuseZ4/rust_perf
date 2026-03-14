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
pub struct Mass3DEA {
    ne: usize,
    b: *mut f64,
    d: *mut f64,
    m: *mut f64,
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
            self.b = alloc_and_init_data_const(MEA_Q1D * MEA_D1D, 1.0);
            self.d = alloc_and_init_data_const(MEA_Q1D * MEA_Q1D * MEA_Q1D * self.ne, 1.0);
            self.m = alloc_and_init_data_const(EA_MAT * self.ne, 0.0);
        }
    }

    fn run_kernel(&mut self) {
        let ne = self.ne;
        core::intrinsics::offload::<_, _, ()>(
            _mass3dea,
            [ne as u32, 1, 1],
            [MEA_D1D as u32, MEA_D1D as u32, MEA_D1D as u32],
            (
                self.b as *const [f64; MEA_Q1D * MEA_D1D],
                self.d as *const [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_DEFAULT],
                self.m as *mut [f64; EA_MAT * NE_DEFAULT],
                ne,
            ),
        );
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.m as *const f64, EA_MAT * self.ne) }
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
        B: *const [f64; MEA_Q1D * MEA_D1D],
        D: *const [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_DEFAULT],
        M: *mut [f64; EA_MAT * NE_DEFAULT],
        NE: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _mass3dea(
    B: *const [f64; MEA_Q1D * MEA_D1D],
    D: *const [f64; MEA_Q1D * MEA_Q1D * MEA_Q1D * NE_DEFAULT],
    M: *mut [f64; EA_MAT * NE_DEFAULT],
    NE: usize,
) {
    let e = unsafe { block_idx_x() as usize };
    if e >= NE {
        return;
    }

    let x = unsafe { thread_idx_x() as usize };
    let y = unsafe { thread_idx_y() as usize };
    let z = unsafe { thread_idx_z() as usize };

    if x < MEA_D1D && y < MEA_D1D && z < MEA_D1D {
        let (i1, i2, i3) = (x, y, z);
        for j1 in 0..MEA_D1D {
            for j2 in 0..MEA_D1D {
                for j3 in 0..MEA_D1D {
                    let mut val: f64 = 0.0;
                    for k1 in 0..MEA_Q1D {
                        let (b_k1_i1, b_k1_j1) =
                            unsafe { ((*B)[k1 + MEA_Q1D * i1], (*B)[k1 + MEA_Q1D * j1]) };
                        for k2 in 0..MEA_Q1D {
                            let (b_k2_i2, b_k2_j2) =
                                unsafe { ((*B)[k2 + MEA_Q1D * i2], (*B)[k2 + MEA_Q1D * j2]) };
                            for k3 in 0..MEA_Q1D {
                                let (b_k3_i3, b_k3_j3) =
                                    unsafe { ((*B)[k3 + MEA_Q1D * i3], (*B)[k3 + MEA_Q1D * j3]) };

                                let d_val = unsafe {
                                    (*D)[k1 + MEA_Q1D * (k2 + MEA_Q1D * (k3 + MEA_Q1D * e))]
                                };

                                val += b_k1_i1
                                    * b_k1_j1
                                    * b_k2_i2
                                    * b_k2_j2
                                    * b_k3_i3
                                    * b_k3_j3
                                    * d_val;
                            }
                        }
                    }
                    let idx = i1
                        + MEA_D1D
                            * (i2
                                + MEA_D1D
                                    * (i3
                                        + MEA_D1D
                                            * (j1
                                                + MEA_D1D * (j2 + MEA_D1D * (j3 + MEA_D1D * e)))));
                    unsafe {
                        (*M)[idx] = val;
                    }
                }
            }
        }
    }
}
