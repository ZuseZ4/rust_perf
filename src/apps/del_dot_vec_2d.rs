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

use core::offload::offload_kernel;

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{preload, preload_mut, Preload, PreloadMut};

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn block_dim_x() -> u32 {
    use rustc_offload_frontend::gpu::*;
    let dispatch = dispatch_ptr();
    let x = (*dispatch).workgroup_size_x as u32;
    x
}

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{workgroup_id_x as block_idx_x, workitem_id_x as thread_idx_x};

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc, alloc_and_init_data, alloc_and_init_data_const, calc_checksum, free, inc_data_init_count,
};

#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;

#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{to_real, Real};

#[cfg(target_os = "linux")]
pub struct DelDotVec2D<'a> {
    x: *mut Real,
    y: *mut Real,
    xdot: *mut Real,
    ydot: *mut Real,
    div: *mut Real,
    real_zones: *mut usize,

    p_x: Option<Preload<'a, [Real; NNALLS]>>,
    p_y: Option<Preload<'a, [Real; NNALLS]>>,
    p_xdot: Option<Preload<'a, [Real; NNALLS]>>,
    p_ydot: Option<Preload<'a, [Real; NNALLS]>>,
    p_real_zones: Option<Preload<'a, [usize; N_REAL_ZONES]>>,
    p_div: Option<PreloadMut<'a, [Real; NNALLS]>>,
}

#[cfg(target_os = "linux")]
impl<'a> DelDotVec2D<'a> {
    pub const INIT: Self = DelDotVec2D {
        x: core::ptr::null_mut(),
        y: core::ptr::null_mut(),
        xdot: core::ptr::null_mut(),
        ydot: core::ptr::null_mut(),
        div: core::ptr::null_mut(),
        real_zones: core::ptr::null_mut(),

        p_x: None,
        p_y: None,
        p_xdot: None,
        p_ydot: None,
        p_real_zones: None,
        p_div: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for DelDotVec2D<'a> {
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

        let x_ref: &'a [Real; NNALLS] = unsafe { &*(self.x as *const [Real; NNALLS]) };

        let y_ref: &'a [Real; NNALLS] = unsafe { &*(self.y as *const [Real; NNALLS]) };

        let xdot_ref: &'a [Real; NNALLS] = unsafe { &*(self.xdot as *const [Real; NNALLS]) };

        let ydot_ref: &'a [Real; NNALLS] = unsafe { &*(self.ydot as *const [Real; NNALLS]) };

        let real_zones_ref: &'a [usize; N_REAL_ZONES] =
            unsafe { &*(self.real_zones as *const [usize; N_REAL_ZONES]) };

        let div_ref: &'a mut [Real; NNALLS] = unsafe { &mut *(self.div as *mut [Real; NNALLS]) };

        self.p_x = Some(preload(x_ref));
        self.p_y = Some(preload(y_ref));
        self.p_xdot = Some(preload(xdot_ref));
        self.p_ydot = Some(preload(ydot_ref));
        self.p_real_zones = Some(preload(real_zones_ref));
        self.p_div = Some(preload_mut(div_ref));
    }

    fn run_kernel(&mut self) {
        let ptiny = to_real(1.0e-20);
        let half = to_real(0.5);

        let _p_x = self
            .p_x
            .as_ref()
            .expect("DEL_DOT_VEC_2D x was not preloaded");

        let _p_y = self
            .p_y
            .as_ref()
            .expect("DEL_DOT_VEC_2D y was not preloaded");

        let _p_xdot = self
            .p_xdot
            .as_ref()
            .expect("DEL_DOT_VEC_2D xdot was not preloaded");

        let _p_ydot = self
            .p_ydot
            .as_ref()
            .expect("DEL_DOT_VEC_2D ydot was not preloaded");

        let _p_real_zones = self
            .p_real_zones
            .as_ref()
            .expect("DEL_DOT_VEC_2D real_zones was not preloaded");

        let _p_div = self
            .p_div
            .as_ref()
            .expect("DEL_DOT_VEC_2D div was not preloaded");

        let x = unsafe { &*(self.x as *const [Real; NNALLS]) };
        let y = unsafe { &*(self.y as *const [Real; NNALLS]) };
        let xdot = unsafe { &*(self.xdot as *const [Real; NNALLS]) };
        let ydot = unsafe { &*(self.ydot as *const [Real; NNALLS]) };
        let real_zones = unsafe { &*(self.real_zones as *const [usize; N_REAL_ZONES]) };

        unsafe {
            offload! {
                kernel = del_dot_vec_2d,
                grid_dim = [BLOCKS, 1, 1],
                block_dim = [THREADS_PER_BLOCK, 1, 1],
                args = (
                    self.div as *mut [Real; NNALLS],
                    x,
                    y,
                    xdot,
                    ydot,
                    real_zones,
                    half,
                    ptiny,
                    N_REAL_ZONES,
                ),
            };
        }
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.div as *const Real, NNALLS) }
    }

    fn tear_down(&mut self) -> f64 {
        // Drop output preload first so device writes are copied back before checksum.
        drop(self.p_div.take());

        let checksum = self.update_checksum();

        // Drop read-only preloads before freeing their host allocations.
        drop(self.p_x.take());
        drop(self.p_y.take());
        drop(self.p_xdot.take());
        drop(self.p_ydot.take());
        drop(self.p_real_zones.take());

        unsafe {
            if !self.x.is_null() {
                free(self.x);
                self.x = core::ptr::null_mut();
            }

            if !self.y.is_null() {
                free(self.y);
                self.y = core::ptr::null_mut();
            }

            if !self.xdot.is_null() {
                free(self.xdot);
                self.xdot = core::ptr::null_mut();
            }

            if !self.ydot.is_null() {
                free(self.ydot);
                self.ydot = core::ptr::null_mut();
            }

            if !self.div.is_null() {
                free(self.div);
                self.div = core::ptr::null_mut();
            }

            if !self.real_zones.is_null() {
                free(self.real_zones);
                self.real_zones = core::ptr::null_mut();
            }
        }

        checksum
    }
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::{Real, RealExt};

#[offload_kernel]
fn del_dot_vec_2d(
    div: *mut [Real; NNALLS],
    x: &[Real; NNALLS],
    y: &[Real; NNALLS],
    xdot: &[Real; NNALLS],
    ydot: &[Real; NNALLS],
    real_zones: &[usize; N_REAL_ZONES],
    half: Real,
    ptiny: Real,
    iend: usize,
) {
    let ii = unsafe { (block_idx_x() * block_dim_x() + thread_idx_x()) as usize };

    if ii < iend {
        let i = real_zones[ii];

        unsafe {
            let x1 = x[i];
            let x2 = x[i + 1];
            let x3 = x[i + 1 + JP];
            let x4 = x[i + JP];

            let y1 = y[i];
            let y2 = y[i + 1];
            let y3 = y[i + 1 + JP];
            let y4 = y[i + JP];

            let fx1 = xdot[i];
            let fx2 = xdot[i + 1];
            let fx3 = xdot[i + 1 + JP];
            let fx4 = xdot[i + JP];

            let fy1 = ydot[i];
            let fy2 = ydot[i + 1];
            let fy3 = ydot[i + 1 + JP];
            let fy4 = ydot[i + JP];

            let xi = half * (x1 + x2 - x3 - x4);
            let xj = half * (x2 + x3 - x4 - x1);

            let yi = half * (y1 + y2 - y3 - y4);
            let yj = half * (y2 + y3 - y4 - y1);

            let fxi = half * (fx1 + fx2 - fx3 - fx4);
            let fxj = half * (fx2 + fx3 - fx4 - fx1);

            let fyi = half * (fy1 + fy2 - fy3 - fy4);
            let fyj = half * (fy2 + fy3 - fy4 - fy1);

            let rarea = Real::from(1.0) / (xi * yj - xj * yi + ptiny);

            let dfxdx = rarea * (fxi * yj - fxj * yi);
            let dfydy = rarea * (fyj * xi - fyi * xj);

            let affine = (fy1 + fy2 + fy3 + fy4) / (y1 + y2 + y3 + y4);

            (*div)[i] = dfxdx + dfydy + affine;
        }
    }
}
