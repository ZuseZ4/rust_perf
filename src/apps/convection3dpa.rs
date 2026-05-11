pub const CONV_D1D: usize = 3;
pub const CONV_Q1D: usize = 4;
pub const CONV_VDIM: usize = 3;
pub const NE_DEFAULT: usize = 15625;
const DEFAULT_REPS: u32 = 50;

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _syncthreads,
    _thread_idx_x as thread_idx_x, _thread_idx_y as thread_idx_y, _thread_idx_z as thread_idx_z,
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
    #[link_name = "llvm.amdgcn.workgroup.size.x"]
    fn block_dim_x() -> i32;
}

#[cfg(target_os = "linux")]
use crate::common::data_utils::{alloc_and_init_data_const, calc_checksum, free};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Convection3DPA {
    ne: usize,
    b: *mut Real,
    bt: *mut Real,
    g: *mut Real,
    d: *mut Real,
    x: *mut Real,
    y: *mut Real,
}

#[cfg(target_os = "linux")]
impl Convection3DPA {
    pub const INIT: Self = Convection3DPA {
        ne: 0,
        b: core::ptr::null_mut(),
        bt: core::ptr::null_mut(),
        g: core::ptr::null_mut(),
        d: core::ptr::null_mut(),
        x: core::ptr::null_mut(),
        y: core::ptr::null_mut(),
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Convection3DPA {
    fn name(&self) -> &'static str {
        kernel_name!("CONVECTION3DPA")
    }

    fn default_problem_size(&self) -> usize {
        NE_DEFAULT * CONV_D1D * CONV_D1D * CONV_D1D
    }

    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        self.ne = NE_DEFAULT;

        unsafe {
            self.b = alloc_and_init_data_const(CONV_Q1D * CONV_D1D, to_real(1.0));
            self.bt = alloc_and_init_data_const(CONV_D1D * CONV_Q1D, to_real(1.0));
            self.g = alloc_and_init_data_const(CONV_Q1D * CONV_D1D, to_real(1.0));
            self.d = alloc_and_init_data_const(
                CONV_Q1D * CONV_Q1D * CONV_Q1D * CONV_VDIM * self.ne,
                to_real(1.0),
            );
            self.x =
                alloc_and_init_data_const(CONV_D1D * CONV_D1D * CONV_D1D * self.ne, to_real(1.0));
            self.y =
                alloc_and_init_data_const(CONV_D1D * CONV_D1D * CONV_D1D * self.ne, to_real(0.0));
        }
    }

    fn run_kernel(&mut self) {
        let ne = self.ne;

        core::intrinsics::offload::<_, _, ()>(
            _convection3dpa,
            [ne as u32, 1, 1],
            [CONV_Q1D as u32, CONV_Q1D as u32, CONV_Q1D as u32],
            (6 * CONV_Q1D * CONV_Q1D * CONV_Q1D * 8) as u32,
            (
                self.b as *const [Real; CONV_Q1D * CONV_D1D],
                self.bt as *const [Real; CONV_D1D * CONV_Q1D],
                self.g as *const [Real; CONV_Q1D * CONV_D1D],
                self.d as *const [Real; CONV_Q1D * CONV_Q1D * CONV_Q1D * CONV_VDIM * NE_DEFAULT],
                self.x as *const [Real; CONV_D1D * CONV_D1D * CONV_D1D * NE_DEFAULT],
                self.y as *mut [Real; CONV_D1D * CONV_D1D * CONV_D1D * NE_DEFAULT],
                ne,
            ),
        )
    }

    fn update_checksum(&self) -> f64 {
        unsafe {
            calc_checksum(
                self.y as *const Real,
                CONV_D1D * CONV_D1D * CONV_D1D * self.ne,
            )
        }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.b);
            free(self.bt);
            free(self.g);
            free(self.d);
            free(self.x);
            free(self.y);
            self.b = core::ptr::null_mut();
            self.bt = core::ptr::null_mut();
            self.g = core::ptr::null_mut();
            self.d = core::ptr::null_mut();
            self.x = core::ptr::null_mut();
            self.y = core::ptr::null_mut();
        }
        self.ne = 0;
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _convection3dpa(
        B: *const [Real; CONV_Q1D * CONV_D1D],
        Bt: *const [Real; CONV_D1D * CONV_Q1D],
        G: *const [Real; CONV_Q1D * CONV_D1D],
        D: *const [Real; CONV_Q1D * CONV_Q1D * CONV_Q1D * CONV_VDIM * NE_DEFAULT],
        X: *const [Real; CONV_D1D * CONV_D1D * CONV_D1D * NE_DEFAULT],
        Y: *mut [Real; CONV_D1D * CONV_D1D * CONV_D1D * NE_DEFAULT],
        NE: usize,
    );
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _convection3dpa(
    B: *const [Real; CONV_Q1D * CONV_D1D],
    Bt: *const [Real; CONV_D1D * CONV_Q1D],
    G: *const [Real; CONV_Q1D * CONV_D1D],
    D: *const [Real; CONV_Q1D * CONV_Q1D * CONV_Q1D * CONV_VDIM * NE_DEFAULT],
    X: *const [Real; CONV_D1D * CONV_D1D * CONV_D1D * NE_DEFAULT],
    Y: *mut [Real; CONV_D1D * CONV_D1D * CONV_D1D * NE_DEFAULT],
    NE: usize,
) {
    let e = block_idx_x() as usize;
    if e >= NE {
        return;
    }

    let tx = thread_idx_x() as usize;
    let ty = thread_idx_y() as usize;
    let tz = thread_idx_z() as usize;

    const MD: usize = CONV_D1D;
    const MQ: usize = CONV_Q1D;
    const SM_SIZE: usize = MQ * MQ * MQ;

    let shmem =
        core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<[u8; 6 * MQ * MQ * MQ * 8]>();
    let sm0_ptr = shmem as *mut Real;
    let sm1_ptr = sm0_ptr.add(SM_SIZE);
    let sm2_ptr = sm1_ptr.add(SM_SIZE);
    let sm3_ptr = sm2_ptr.add(SM_SIZE);
    let sm4_ptr = sm3_ptr.add(SM_SIZE);
    let sm5_ptr = sm4_ptr.add(SM_SIZE);

    if tz < MD && ty < MD && tx < MD {
        let idx = tx + MD * (ty + MD * (tz + MD * e));
        *sm0_ptr.add(tx + MD * (ty + MD * tz)) = (*X)[idx];
    }
    _syncthreads();

    if tz < MD && ty < MD && tx < MQ {
        let mut Bu_val = Real::from(0.0);
        let mut Gu_val = Real::from(0.0);
        for dx in 0..MD {
            let bx = (*B)[tx + MQ * dx];
            let gx = (*G)[tx + MQ * dx];
            let val = *sm0_ptr.add(dx + MD * (ty + MD * tz));
            Bu_val += bx * val;
            Gu_val += gx * val;
        }
        *sm1_ptr.add(tx + MQ * (ty + MD * tz)) = Bu_val;
        *sm2_ptr.add(tx + MQ * (ty + MD * tz)) = Gu_val;
    }
    _syncthreads();

    if tz < MD && ty < MQ && tx < MQ {
        let mut BBu_val = Real::from(0.0);
        let mut GBu_val = Real::from(0.0);
        let mut BGu_val = Real::from(0.0);
        for dy in 0..MD {
            let by = (*B)[ty + MQ * dy];
            let gy = (*G)[ty + MQ * dy];
            BBu_val += by * (*sm1_ptr.add(tx + MQ * (dy + MD * tz)));
            GBu_val += gy * (*sm1_ptr.add(tx + MQ * (dy + MD * tz)));
            BGu_val += by * (*sm2_ptr.add(tx + MQ * (dy + MD * tz)));
        }
        *sm3_ptr.add(tx + MQ * (ty + MQ * tz)) = BBu_val;
        *sm4_ptr.add(tx + MQ * (ty + MQ * tz)) = GBu_val;
        *sm5_ptr.add(tx + MQ * (ty + MQ * tz)) = BGu_val;
    }
    _syncthreads();

    if tz < MQ && ty < MQ && tx < MQ {
        let mut GBBu_val = Real::from(0.0);
        let mut BGBu_val = Real::from(0.0);
        let mut BBGu_val = Real::from(0.0);
        for dz in 0..MD {
            let bz = (*B)[tz + MQ * dz];
            let gz = (*G)[tz + MQ * dz];
            GBBu_val += gz * (*sm3_ptr.add(tx + MQ * (ty + MQ * dz)));
            BGBu_val += bz * (*sm4_ptr.add(tx + MQ * (ty + MQ * dz)));
            BBGu_val += bz * (*sm5_ptr.add(tx + MQ * (ty + MQ * dz)));
        }
        let o1 = (*D)[tx + MQ * (ty + MQ * (tz + MQ * (0 + CONV_VDIM * e)))];
        let o2 = (*D)[tx + MQ * (ty + MQ * (tz + MQ * (1 + CONV_VDIM * e)))];
        let o3 = (*D)[tx + MQ * (ty + MQ * (tz + MQ * (2 + CONV_VDIM * e)))];
        *sm3_ptr.add(tx + MQ * (ty + MQ * tz)) = o1 * BBGu_val + o2 * BGBu_val + o3 * GBBu_val;
    }
    _syncthreads();

    if tz < MD && ty < MQ && tx < MQ {
        let mut val = Real::from(0.0);
        for qz in 0..MQ {
            val += (*Bt)[tz + MD * qz] * (*sm3_ptr.add(tx + MQ * (ty + MQ * qz)));
        }
        *sm4_ptr.add(tx + MQ * (ty + MQ * tz)) = val;
    }
    _syncthreads();

    if tz < MD && ty < MD && tx < MQ {
        let mut val = Real::from(0.0);
        for qy in 0..MQ {
            val += (*Bt)[ty + MD * qy] * (*sm4_ptr.add(tx + MQ * (qy + MQ * tz)));
        }
        *sm5_ptr.add(tx + MQ * (ty + MD * tz)) = val;
    }
    _syncthreads();

    if tz < MD && ty < MD && tx < MD {
        let mut val = Real::from(0.0);
        for qx in 0..MQ {
            val += (*Bt)[tx + MD * qx] * (*sm5_ptr.add(qx + MQ * (ty + MD * tz)));
        }
        let idx = tx + MD * (ty + MD * (tz + MD * e));
        (*Y)[idx] += val;
    }
}
