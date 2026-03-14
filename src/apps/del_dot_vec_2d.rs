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
pub struct DelDotVec2D {
    x: *mut f64,
    y: *mut f64,
    xdot: *mut f64,
    ydot: *mut f64,
    div: *mut f64,
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
            self.x = alloc_and_init_data_const(NNALLS, 0.0);
            self.y = alloc_and_init_data_const(NNALLS, 0.0);
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
                    *self.x.add(idx) = i as f64 * dx;
                    *self.y.add(idx) = j as f64 * dy;
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
            self.div = alloc_and_init_data_const(NNALLS, 0.0);
        }
    }

    fn run_kernel(&mut self) {
        const PTINY: f64 = 1.0e-20;
        const HALF: f64 = 0.5;
        unsafe {
            core::intrinsics::offload::<_, _, ()>(
                _del_dot_vec_2d,
                [BLOCKS, 1, 1],
                [THREADS_PER_BLOCK, 1, 1],
                (
                    self.div as *mut [f64; NNALLS],
                    &*(self.x as *const [f64; NNALLS]),
                    &*(self.y as *const [f64; NNALLS]),
                    &*(self.xdot as *const [f64; NNALLS]),
                    &*(self.ydot as *const [f64; NNALLS]),
                    &*(self.real_zones as *const [usize; N_REAL_ZONES]),
                    JP,
                    HALF,
                    PTINY,
                    N_REAL_ZONES,
                ),
            );
        }
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.div as *const f64, NNALLS) }
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
        div: *mut [f64; NNALLS],
        x: &[f64; NNALLS],
        y: &[f64; NNALLS],
        xdot: &[f64; NNALLS],
        ydot: &[f64; NNALLS],
        real_zones: &[usize; N_REAL_ZONES],
        jp: usize,
        half: f64,
        ptiny: f64,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _del_dot_vec_2d(
    div: &mut [f64; NNALLS],
    x: &[f64; NNALLS],
    y: &[f64; NNALLS],
    xdot: &[f64; NNALLS],
    ydot: &[f64; NNALLS],
    real_zones: &[usize; N_REAL_ZONES],
    jp: usize,
    half: f64,
    ptiny: f64,
    iend: usize,
) {
    let ii = unsafe { (block_idx_x() * block_dim_x() + thread_idx_x()) as usize };
    if ii < iend {
        let i = real_zones[ii];

        let x1 = x[i];
        let x2 = x[i + 1];
        let x3 = x[i + 1 + jp];
        let x4 = x[i + jp];

        let y1 = y[i];
        let y2 = y[i + 1];
        let y3 = y[i + 1 + jp];
        let y4 = y[i + jp];

        let fx1 = xdot[i];
        let fx2 = xdot[i + 1];
        let fx3 = xdot[i + 1 + jp];
        let fx4 = xdot[i + jp];

        let fy1 = ydot[i];
        let fy2 = ydot[i + 1];
        let fy3 = ydot[i + 1 + jp];
        let fy4 = ydot[i + jp];

        let xi = half * (x2 - x4 + x1 - x3);
        let xj = half * (x3 - x1 + x2 - x4);
        let yi = half * (y2 - y4 + y1 - y3);
        let yj = half * (y3 - y1 + y2 - y4);

        let fxi = half * (fx2 - fx4 + fx1 - fx3);
        let fxj = half * (fx3 - fx1 + fx2 - fx4);
        let fyi = half * (fy2 - fy4 + fy1 - fy3);
        let fyj = half * (fy3 - fy1 + fy2 - fy4);

        let rarea = 1.0 / (xi * yj - xj * yi + ptiny);
        let dfxdx = rarea * (fxi * yj - fxj * yi);
        let dfydy = rarea * (fyj * xi - fyi * xj);
        let affine = (fy1 + fy2 + fy3 + fy4) / (y1 + y2 + y3 + y4);

        div[i] = dfxdx + dfydy + affine;
    }
}
