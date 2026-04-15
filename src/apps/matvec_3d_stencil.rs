pub const N_DEFAULT: usize = 1000000;
const DEFAULT_REPS: u32 = 100;

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{_block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x};

#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn thread_idx_x() -> i32;
}

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data, alloc_and_init_data_const, calc_checksum, calc_multiplier, free,
    inc_data_init_count,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Matvec3DStencil {
    n: usize,
    x: *mut Real,
    b: *mut Real,
    matrix: [*mut Real; 14],
    real_zones: *mut u64,
    jp: usize,
    kp: usize,
    fpz: usize,
    lpz: usize,
}

#[cfg(target_os = "linux")]
impl Matvec3DStencil {
    pub const INIT: Self = Matvec3DStencil {
        n: 0,
        x: core::ptr::null_mut(),
        b: core::ptr::null_mut(),
        matrix: [core::ptr::null_mut(); 14],
        real_zones: core::ptr::null_mut(),
        jp: 0,
        kp: 0,
        fpz: 0,
        lpz: 0,
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Matvec3DStencil {
    fn name(&self) -> &'static str {
        kernel_name!("MATVEC_3D_STENCIL")
    }

    fn default_problem_size(&self) -> usize {
        N_DEFAULT
    }

    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        self.n = N_DEFAULT;
        let m: usize = 100;
        let npnl = 2;
        let npnr = 1;
        let dim_size = m + 1;
        let nn_dim = dim_size + npnl + npnr;
        self.jp = nn_dim;
        self.kp = nn_dim * nn_dim;
        let nnalls = nn_dim * nn_dim * nn_dim;

        let imin = npnl;
        let imax = npnl + dim_size - 1;
        let jmin = npnl;
        let jmax = npnl + dim_size - 1;
        let kmin = npnl;
        let kmax = npnl + dim_size - 1;

        let npzl = npnl - 1;
        let npzr = npnr + 1 - 1;

        self.fpz = (kmin - npzl) * self.kp + (jmin - npzl) * self.jp + (imin - npzl);
        self.lpz = (kmax - 1 + npzr) * self.kp + (jmax - 1 + npzr) * self.jp + (imax - 1 + npzr);

        unsafe {
            self.b = alloc_and_init_data_const(nnalls, to_real(0.0));
            self.x = alloc_and_init_data(nnalls);

            for i in 0..14 {
                self.matrix[i] = alloc_and_init_data(nnalls);
            }

            self.real_zones = crate::common::data_utils::alloc::<u64>(self.n);
            for i in 0..self.n {
                *self.real_zones.add(i) = u64::MAX;
            }
            inc_data_init_count();

            let j_stride = imax - imin;
            let k_stride = j_stride * (jmax - jmin);

            for k in kmin..kmax {
                for j in jmin..jmax {
                    for i in imin..imax {
                        let iz = i + j * self.jp + k * self.kp;
                        let il = (i - imin) + (j - jmin) * j_stride + (k - kmin) * k_stride;
                        *self.real_zones.add(il) = iz as u64;
                    }
                }
            }
        }
    }

    fn run_kernel(&mut self) {
        let n = self.n;
        let jp = self.jp;
        let kp = self.kp;

        core::intrinsics::offload::<_, _, ()>(
            _matvec3dstencil,
            [n.div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            (
                self.x as *const [Real; 1124864],
                self.b as *mut [Real; 1124864],
                self.matrix[0] as *const [Real; 1124864],
                self.matrix[1] as *const [Real; 1124864],
                self.matrix[2] as *const [Real; 1124864],
                self.matrix[3] as *const [Real; 1124864],
                self.matrix[4] as *const [Real; 1124864],
                self.matrix[5] as *const [Real; 1124864],
                self.matrix[6] as *const [Real; 1124864],
                self.matrix[7] as *const [Real; 1124864],
                self.matrix[8] as *const [Real; 1124864],
                self.matrix[9] as *const [Real; 1124864],
                self.matrix[10] as *const [Real; 1124864],
                self.matrix[11] as *const [Real; 1124864],
                self.matrix[12] as *const [Real; 1124864],
                self.matrix[13] as *const [Real; 1124864],
                self.real_zones as *const [u64; 1000000],
                jp,
                kp,
                n,
            ),
        );
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.b as *const Real, 1124864) }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.x);
            free(self.b);
            for i in 0..14 {
                free(self.matrix[i]);
                self.matrix[i] = core::ptr::null_mut();
            }
            free(self.real_zones);
            self.x = core::ptr::null_mut();
            self.b = core::ptr::null_mut();
            self.real_zones = core::ptr::null_mut();
        }
        self.n = 0;
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _matvec3dstencil(
        x: *const [Real; 1124864],
        b: *mut [Real; 1124864],
        m0: *const [Real; 1124864],
        m1: *const [Real; 1124864],
        m2: *const [Real; 1124864],
        m3: *const [Real; 1124864],
        m4: *const [Real; 1124864],
        m5: *const [Real; 1124864],
        m6: *const [Real; 1124864],
        m7: *const [Real; 1124864],
        m8: *const [Real; 1124864],
        m9: *const [Real; 1124864],
        m10: *const [Real; 1124864],
        m11: *const [Real; 1124864],
        m12: *const [Real; 1124864],
        m13: *const [Real; 1124864],
        real_zones: *const [u64; 1000000],
        jp: usize,
        kp: usize,
        n: usize,
    );
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _matvec3dstencil(
    x: *const [Real; 1124864],
    b: *mut [Real; 1124864],
    m0: *const [Real; 1124864],
    m1: *const [Real; 1124864],
    m2: *const [Real; 1124864],
    m3: *const [Real; 1124864],
    m4: *const [Real; 1124864],
    m5: *const [Real; 1124864],
    m6: *const [Real; 1124864],
    m7: *const [Real; 1124864],
    m8: *const [Real; 1124864],
    m9: *const [Real; 1124864],
    m10: *const [Real; 1124864],
    m11: *const [Real; 1124864],
    m12: *const [Real; 1124864],
    m13: *const [Real; 1124864],
    real_zones: *const [u64; 1000000],
    jp: usize,
    kp: usize,
    n: usize,
) {
    let i = unsafe { (block_idx_x() * 256 + thread_idx_x()) as usize };
    if i >= n {
        return;
    }

    let iz = unsafe { (*real_zones)[i] } as usize;
    let s_jp = jp as isize;
    let s_kp = kp as isize;

    unsafe {
        let mut b0 = (*m0)[iz] * (*x)[(iz as isize - 1 - s_jp - s_kp) as usize];
        b0 += (*m1)[iz] * (*x)[(iz as isize - s_jp - s_kp) as usize];
        b0 += (*m2)[iz] * (*x)[(iz as isize + 1 - s_jp - s_kp) as usize];
        b0 += (*m3)[iz] * (*x)[(iz as isize - 1 - s_kp) as usize];
        b0 += (*m4)[iz] * (*x)[(iz as isize - s_kp) as usize];
        b0 += (*m5)[iz] * (*x)[(iz as isize + 1 - s_kp) as usize];
        b0 += (*m6)[iz] * (*x)[(iz as isize - 1 + s_jp - s_kp) as usize];
        b0 += (*m7)[iz] * (*x)[(iz as isize + s_jp - s_kp) as usize];
        b0 += (*m8)[iz] * (*x)[(iz as isize + 1 + s_jp - s_kp) as usize];

        b0 += (*m9)[iz] * (*x)[(iz as isize - 1 - s_jp) as usize];
        b0 += (*m10)[iz] * (*x)[(iz as isize - s_jp) as usize];
        b0 += (*m11)[iz] * (*x)[(iz as isize + 1 - s_jp) as usize];
        b0 += (*m12)[iz] * (*x)[(iz as isize - 1) as usize];
        b0 += (*m13)[iz] * (*x)[iz];
        b0 += (*m12)[iz + 1] * (*x)[iz + 1];
        b0 += (*m11)[iz - 1 + jp] * (*x)[iz - 1 + jp];
        b0 += (*m10)[iz + jp] * (*x)[iz + jp];
        b0 += (*m9)[iz + 1 + jp] * (*x)[iz + 1 + jp];

        b0 += (*m8)[iz - 1 - jp + kp] * (*x)[(iz as isize - 1 - s_jp + s_kp) as usize];
        b0 += (*m7)[iz - jp + kp] * (*x)[(iz as isize - s_jp + s_kp) as usize];
        b0 += (*m6)[iz + 1 - jp + kp] * (*x)[(iz as isize + 1 - s_jp + s_kp) as usize];
        b0 += (*m5)[iz - 1 + kp] * (*x)[(iz as isize - 1 + s_kp) as usize];
        b0 += (*m4)[iz + kp] * (*x)[(iz as isize + s_kp) as usize];
        b0 += (*m3)[iz + 1 + kp] * (*x)[(iz as isize + 1 + s_kp) as usize];
        b0 += (*m2)[iz - 1 + jp + kp] * (*x)[(iz as isize - 1 + s_jp + s_kp) as usize];
        b0 += (*m1)[iz + jp + kp] * (*x)[(iz as isize + s_jp + s_kp) as usize];
        b0 += (*m0)[iz + 1 + jp + kp] * (*x)[(iz as isize + 1 + s_jp + s_kp) as usize];

        (*b)[iz] = b0;
    }
}
