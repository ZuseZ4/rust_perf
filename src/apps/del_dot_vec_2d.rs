const RZMAX: usize = 1001;
const DEFAULT_REPS: u32 = 100;

const NPNL: usize = 2;
const NPNR: usize = 1;

const IMIN: usize = NPNL;
const IMAX: usize = NPNL + RZMAX - 1;
const JMIN: usize = NPNL;
const JMAX: usize = NPNL + RZMAX - 1;

const NNALLS_1D: usize = IMAX + 1 - IMIN + NPNL + NPNR;
const JP: usize = NNALLS_1D;
const NNALLS: usize = NNALLS_1D * (JMAX + 1 - JMIN + NPNL + NPNR);
const N_REAL_ZONES: usize = (IMAX - IMIN) * (JMAX - JMIN);

const THREADS_PER_BLOCK: u32 = 256;
const BLOCKS: u32 = (N_REAL_ZONES as u32).div_ceil(THREADS_PER_BLOCK);

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{workgroup_id_x as block_idx_x, workitem_id_x as thread_idx_x};
#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};
#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workgroup.size.x"]
    fn block_dim_x() -> u32;
}

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc, alloc_and_init_data, alloc_and_init_data_const, calc_checksum, free, inc_data_init_count,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct DelDotVec2D {
    x: *mut Real,
    y: *mut Real,
    xdot: *mut Real,
    ydot: *mut Real,
    div: *mut Real,
    real_zones: *mut usize,
}

#[cfg(target_os = "linux")]
impl DelDotVec2D {
    pub const INIT: Self = DelDotVec2D {
        x: core::ptr::null_mut(),
        y: core::ptr::null_mut(),
        xdot: core::ptr::null_mut(),
        ydot: core::ptr::null_mut(),
        div: core::ptr::null_mut(),
        real_zones: core::ptr::null_mut(),
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for DelDotVec2D {
    fn name(&self) -> &'static str {
        kernel_name!("DEL_DOT_VEC_2D")
    }
    fn default_problem_size(&self) -> usize {
        N_REAL_ZONES
    }
    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        unsafe {
            self.x = alloc_and_init_data_const(NNALLS, to_real(0.0));
            self.y = alloc_and_init_data_const(NNALLS, to_real(0.0));
        }
        self.real_zones = unsafe { alloc::<usize>(N_REAL_ZONES) };
        for i in 0..N_REAL_ZONES {
            unsafe {
                *self.real_zones.add(i) = usize::MAX;
            }
        }
        inc_data_init_count();

        let dx: f64 = 0.2;
        let dy: f64 = 0.1;
        let jstart = JMIN as isize - NPNL as isize;
        let jend = JMAX as isize + 1 + NPNR as isize;
        let istart = IMIN as isize - NPNL as isize;
        let iend = IMAX as isize + 1 + NPNR as isize;
        for j in jstart..jend {
            for i in istart..iend {
                let idx = (i + j * JP as isize) as usize;
                unsafe {
                    *self.x.add(idx) = to_real(i as f64 * dx);
                    *self.y.add(idx) = to_real(j as f64 * dy);
                }
            }
        }

        let j_stride = (IMAX - IMIN) as isize;
        for j in JMIN..JMAX {
            for i in IMIN..IMAX {
                let iz = i + j * JP;
                let il = (i - IMIN) + (j - JMIN) * j_stride as usize;
                unsafe {
                    *self.real_zones.add(il) = iz;
                }
            }
        }

        unsafe {
            self.xdot = alloc_and_init_data(NNALLS);
            self.ydot = alloc_and_init_data(NNALLS);
            self.div = alloc_and_init_data_const(NNALLS, to_real(0.0));
        }
    }

    fn run_kernel(&mut self) {
        let ptiny = to_real(1.0e-20);
        let half = to_real(0.5);

        let x1 = self.x as *const [Real; NNALLS];
        let x2 = unsafe { self.x.add(1) as *const Real };
        let x3 = unsafe { self.x.add(1 + JP) as *const Real };
        let x4 = unsafe { self.x.add(JP) as *const Real };

        let y1 = self.y as *const [Real; NNALLS];
        let y2 = unsafe { self.y.add(1) as *const Real };
        let y3 = unsafe { self.y.add(1 + JP) as *const Real };
        let y4 = unsafe { self.y.add(JP) as *const Real };

        let fx1 = self.xdot as *const [Real; NNALLS];
        let fx2 = unsafe { self.xdot.add(1) as *const Real };
        let fx3 = unsafe { self.xdot.add(1 + JP) as *const Real };
        let fx4 = unsafe { self.xdot.add(JP) as *const Real };

        let fy1 = self.ydot as *const [Real; NNALLS];
        let fy2 = unsafe { self.ydot.add(1) as *const Real };
        let fy3 = unsafe { self.ydot.add(1 + JP) as *const Real };
        let fy4 = unsafe { self.ydot.add(JP) as *const Real };

        unsafe {
            core::intrinsics::offload::<_, _, ()>(
                _del_dot_vec_2d,
                [BLOCKS, 1, 1],
                [THREADS_PER_BLOCK, 1, 1],
                (
                    self.div as *mut [Real; NNALLS],
                    &*x1,
                    x2,
                    x3,
                    x4,
                    &*y1,
                    y2,
                    y3,
                    y4,
                    &*fx1,
                    fx2,
                    fx3,
                    fx4,
                    &*fy1,
                    fy2,
                    fy3,
                    fy4,
                    &*(self.real_zones as *const [usize; N_REAL_ZONES]),
                    half,
                    ptiny,
                    N_REAL_ZONES,
                ),
            );
        }
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.div as *const Real, NNALLS) }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.x);
            self.x = core::ptr::null_mut();
            free(self.y);
            self.y = core::ptr::null_mut();
            free(self.xdot);
            self.xdot = core::ptr::null_mut();
            free(self.ydot);
            self.ydot = core::ptr::null_mut();
            free(self.div);
            self.div = core::ptr::null_mut();
            free(self.real_zones);
            self.real_zones = core::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _del_dot_vec_2d(
        div: *mut [Real; NNALLS],
        x1: &[Real; NNALLS],
        x2: *const Real,
        x3: *const Real,
        x4: *const Real,
        y1: &[Real; NNALLS],
        y2: *const Real,
        y3: *const Real,
        y4: *const Real,
        fx1: &[Real; NNALLS],
        fx2: *const Real,
        fx3: *const Real,
        fx4: *const Real,
        fy1: &[Real; NNALLS],
        fy2: *const Real,
        fy3: *const Real,
        fy4: *const Real,
        real_zones: &[usize; N_REAL_ZONES],
        half: Real,
        ptiny: Real,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::{Real, RealExt};

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _del_dot_vec_2d(
    div: *mut [Real; NNALLS],
    x1: &[Real; NNALLS],
    x2: *const Real,
    x3: *const Real,
    x4: *const Real,
    y1: &[Real; NNALLS],
    y2: *const Real,
    y3: *const Real,
    y4: *const Real,
    fx1: &[Real; NNALLS],
    fx2: *const Real,
    fx3: *const Real,
    fx4: *const Real,
    fy1: &[Real; NNALLS],
    fy2: *const Real,
    fy3: *const Real,
    fy4: *const Real,
    real_zones: &[usize; N_REAL_ZONES],
    half: Real,
    ptiny: Real,
    iend: usize,
) {
    let ii = unsafe { (block_idx_x() * block_dim_x() + thread_idx_x()) as usize };
    if ii < iend {
        let i = real_zones[ii];

        let xi = half * (x1[i] + *x2.add(i) - *x3.add(i) - *x4.add(i));
        let xj = half * (*x2.add(i) + *x3.add(i) - *x4.add(i) - x1[i]);

        let yi = half * (y1[i] + *y2.add(i) - *y3.add(i) - *y4.add(i));
        let yj = half * (*y2.add(i) + *y3.add(i) - *y4.add(i) - y1[i]);

        let fxi = half * (fx1[i] + *fx2.add(i) - *fx3.add(i) - *fx4.add(i));
        let fxj = half * (*fx2.add(i) + *fx3.add(i) - *fx4.add(i) - fx1[i]);

        let fyi = half * (fy1[i] + *fy2.add(i) - *fy3.add(i) - *fy4.add(i));
        let fyj = half * (*fy2.add(i) + *fy3.add(i) - *fy4.add(i) - fy1[i]);

        let rarea = Real::from(1.0) / (xi * yj - xj * yi + ptiny);

        let dfxdx = rarea * (fxi * yj - fxj * yi);

        let dfydy = rarea * (fyj * xi - fyi * xj);

        let affine = (fy1[i] + *fy2.add(i) + *fy3.add(i) + *fy4.add(i))
            / (y1[i] + *y2.add(i) + *y3.add(i) + *y4.add(i));

        (*div)[i] = dfxdx + dfydy + affine;
    }
}
