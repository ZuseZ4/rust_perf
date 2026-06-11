pub const N_DEFAULT: usize = 1_000_000;
const DEFAULT_REPS: u32 = 100;

const M_DEFAULT: usize = 100;
const NPNL: usize = 2;
const NPNR: usize = 1;

const DIM_SIZE: usize = M_DEFAULT + 1;
const NN_DIM: usize = DIM_SIZE + NPNL + NPNR;
const NNALLS: usize = NN_DIM * NN_DIM * NN_DIM;

use core::offload::offload_kernel;

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{workgroup_id_x as block_idx_x, workitem_id_x as thread_idx_x};

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{_block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x};

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{preload, preload_mut, Preload, PreloadMut};

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data, alloc_and_init_data_const, calc_checksum, free, inc_data_init_count,
};

#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;

#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{to_real, Real};

#[cfg(target_os = "linux")]
pub struct Matvec3DStencil<'a> {
    n: usize,
    x: *mut Real,
    b: *mut Real,
    matrix: [*mut Real; 14],
    real_zones: *mut u64,
    jp: usize,
    kp: usize,
    fpz: usize,
    lpz: usize,

    p_x: Option<Preload<'a, [Real; NNALLS]>>,
    p_b: Option<PreloadMut<'a, [Real; NNALLS]>>,
    p_matrix: [Option<Preload<'a, [Real; NNALLS]>>; 14],
    p_real_zones: Option<Preload<'a, [u64; N_DEFAULT]>>,
}

#[cfg(target_os = "linux")]
impl<'a> Matvec3DStencil<'a> {
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

        p_x: None,
        p_b: None,
        p_matrix: [const { None }; 14],
        p_real_zones: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for Matvec3DStencil<'a> {
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
        self.n = self.default_problem_size();

        assert_eq!(
            self.n, N_DEFAULT,
            "MATVEC_3D_STENCIL const-array preload version requires n = {N_DEFAULT}, got {}",
            self.n
        );

        let m: usize = M_DEFAULT;
        let npnl = NPNL;
        let npnr = NPNR;

        let dim_size = m + 1;
        let nn_dim = dim_size + npnl + npnr;
        let nnalls = nn_dim * nn_dim * nn_dim;

        assert_eq!(
            nnalls, NNALLS,
            "MATVEC_3D_STENCIL const-array preload version requires NNALLS = {NNALLS}, got {nnalls}",
        );

        self.jp = nn_dim;
        self.kp = nn_dim * nn_dim;

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
            self.b = alloc_and_init_data_const(NNALLS, to_real(0.0));
            self.x = alloc_and_init_data(NNALLS);

            for i in 0..14 {
                self.matrix[i] = alloc_and_init_data(NNALLS);
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

        let x_ref: &'a [Real; NNALLS] = unsafe { &*(self.x as *const [Real; NNALLS]) };

        let b_ref: &'a mut [Real; NNALLS] = unsafe { &mut *(self.b as *mut [Real; NNALLS]) };

        let real_zones_ref: &'a [u64; N_DEFAULT] =
            unsafe { &*(self.real_zones as *const [u64; N_DEFAULT]) };

        self.p_x = Some(preload(x_ref));
        self.p_b = Some(preload_mut(b_ref));
        self.p_real_zones = Some(preload(real_zones_ref));

        for i in 0..14 {
            let matrix_ref: &'a [Real; NNALLS] =
                unsafe { &*(self.matrix[i] as *const [Real; NNALLS]) };

            self.p_matrix[i] = Some(preload(matrix_ref));
        }
    }

    fn run_kernel(&mut self) {
        let n = self.n;
        let jp = self.jp;
        let kp = self.kp;

        self.p_x
            .as_ref()
            .expect("MATVEC_3D_STENCIL x was not preloaded");

        self.p_b
            .as_ref()
            .expect("MATVEC_3D_STENCIL b was not preloaded");

        self.p_real_zones
            .as_ref()
            .expect("MATVEC_3D_STENCIL real_zones was not preloaded");

        for p in &self.p_matrix {
            p.as_ref()
                .expect("MATVEC_3D_STENCIL matrix entry was not preloaded");
        }

        offload! {
            kernel = matvec3dstencil,
            grid_dim = [n.div_ceil(256) as u32, 1, 1],
            block_dim = [256, 1, 1],
            args = (
                self.x as *const [Real; NNALLS],
                self.b as *mut [Real; NNALLS],
                self.matrix[0] as *const [Real; NNALLS],
                self.matrix[1] as *const [Real; NNALLS],
                self.matrix[2] as *const [Real; NNALLS],
                self.matrix[3] as *const [Real; NNALLS],
                self.matrix[4] as *const [Real; NNALLS],
                self.matrix[5] as *const [Real; NNALLS],
                self.matrix[6] as *const [Real; NNALLS],
                self.matrix[7] as *const [Real; NNALLS],
                self.matrix[8] as *const [Real; NNALLS],
                self.matrix[9] as *const [Real; NNALLS],
                self.matrix[10] as *const [Real; NNALLS],
                self.matrix[11] as *const [Real; NNALLS],
                self.matrix[12] as *const [Real; NNALLS],
                self.matrix[13] as *const [Real; NNALLS],
                self.real_zones as *const [u64; N_DEFAULT],
                jp,
                kp,
                n,
            ),
        };
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.b as *const Real, NNALLS) }
    }

    fn tear_down(&mut self) -> f64 {
        // Drop preloads first. Dropping p_b copies device output back to host.
        drop(self.p_b.take());
        drop(self.p_x.take());
        drop(self.p_real_zones.take());

        for p in &mut self.p_matrix {
            drop(p.take());
        }

        let checksum = self.update_checksum();

        unsafe {
            if !self.x.is_null() {
                free(self.x);
                self.x = core::ptr::null_mut();
            }

            if !self.b.is_null() {
                free(self.b);
                self.b = core::ptr::null_mut();
            }

            for i in 0..14 {
                if !self.matrix[i].is_null() {
                    free(self.matrix[i]);
                    self.matrix[i] = core::ptr::null_mut();
                }
            }

            if !self.real_zones.is_null() {
                free(self.real_zones);
                self.real_zones = core::ptr::null_mut();
            }
        }

        self.n = 0;
        self.jp = 0;
        self.kp = 0;
        self.fpz = 0;
        self.lpz = 0;

        checksum
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[offload_kernel]
fn matvec3dstencil(
    x: *const [Real; NNALLS],
    b: *mut [Real; NNALLS],
    m0: *const [Real; NNALLS],
    m1: *const [Real; NNALLS],
    m2: *const [Real; NNALLS],
    m3: *const [Real; NNALLS],
    m4: *const [Real; NNALLS],
    m5: *const [Real; NNALLS],
    m6: *const [Real; NNALLS],
    m7: *const [Real; NNALLS],
    m8: *const [Real; NNALLS],
    m9: *const [Real; NNALLS],
    m10: *const [Real; NNALLS],
    m11: *const [Real; NNALLS],
    m12: *const [Real; NNALLS],
    m13: *const [Real; NNALLS],
    real_zones: *const [u64; N_DEFAULT],
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
