pub const N_DEFAULT: usize = 1_000_000;
const DEFAULT_REPS: u32 = 100;

const M_DEFAULT: usize = 100;
const NPNL: usize = 2;
const NPNR: usize = 1;

const DIM_SIZE: usize = M_DEFAULT + 1;
const NN_DIM: usize = DIM_SIZE + NPNL + NPNR;
const NNALLS: usize = NN_DIM * NN_DIM * NN_DIM;

use core::offload::offload_kernel;
use rustc_offload_frontend::partition::{OffsetStride1D, PartitioningStrategy, Region};

#[cfg(target_os = "linux")]
use rustc_offload_frontend::offload;

#[cfg(target_os = "linux")]
use core::offload::offload::{preload, preload_mut, Preload, PreloadMut};

#[cfg(target_os = "linux")]
use crate::common::data_utils::{alloc_and_init_data_const, calc_checksum, free};

#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;

#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{to_real, Real};

#[cfg(target_os = "linux")]
pub struct Vol3D<'a> {
    n: usize,
    x: *mut Real,
    y: *mut Real,
    z: *mut Real,
    vol: *mut Real,
    vnormq: Real,
    jp: usize,
    kp: usize,
    fpz: usize,
    lpz: usize,

    p_x: Option<Preload<'a, [Real; NNALLS]>>,
    p_y: Option<Preload<'a, [Real; NNALLS]>>,
    p_z: Option<Preload<'a, [Real; NNALLS]>>,
    p_vol: Option<PreloadMut<'a, [Real; NNALLS]>>,
}

#[cfg(target_os = "linux")]
impl<'a> Vol3D<'a> {
    pub const INIT: Self = Vol3D {
        n: 0,
        x: core::ptr::null_mut(),
        y: core::ptr::null_mut(),
        z: core::ptr::null_mut(),
        vol: core::ptr::null_mut(),
        vnormq: to_real(0.08333333333333),
        jp: 0,
        kp: 0,
        fpz: 0,
        lpz: 0,

        p_x: None,
        p_y: None,
        p_z: None,
        p_vol: None,
    };
}

#[cfg(target_os = "linux")]
impl<'a> KernelBase for Vol3D<'a> {
    fn name(&self) -> &'static str {
        kernel_name!("VOL3D")
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
            "VOL3D const-array preload version requires n = {N_DEFAULT}, got {}",
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
            "VOL3D const-array preload version requires NNALLS = {NNALLS}, got {nnalls}",
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
            self.x = alloc_and_init_data_const(NNALLS, to_real(0.0));
            self.y = alloc_and_init_data_const(NNALLS, to_real(0.0));
            self.z = alloc_and_init_data_const(NNALLS, to_real(0.0));
            self.vol = alloc_and_init_data_const(NNALLS, to_real(0.0));

            let dx = 0.3;
            let dy = 0.2;
            let dz = 0.1;

            for k in (kmin - npnl)..(kmax + 1 + npnr) {
                for j in (jmin - npnl)..(jmax + 1 + npnr) {
                    for i in (imin - npnl)..(imax + 1 + npnr) {
                        let inn = i + j * self.jp + k * self.kp;

                        *self.x.add(inn) = to_real(i as f64 * dx);
                        *self.y.add(inn) = to_real(j as f64 * dy);
                        *self.z.add(inn) = to_real(k as f64 * dz);
                    }
                }
            }
        }

        let x_ref: &'a [Real; NNALLS] = unsafe { &*(self.x as *const [Real; NNALLS]) };

        let y_ref: &'a [Real; NNALLS] = unsafe { &*(self.y as *const [Real; NNALLS]) };

        let z_ref: &'a [Real; NNALLS] = unsafe { &*(self.z as *const [Real; NNALLS]) };

        let vol_ref: &'a mut [Real; NNALLS] = unsafe { &mut *(self.vol as *mut [Real; NNALLS]) };

        self.p_x = Some(preload(x_ref));
        self.p_y = Some(preload(y_ref));
        self.p_z = Some(preload(z_ref));
        self.p_vol = Some(preload_mut(vol_ref));
    }

    fn run_kernel(&mut self) {
        let fpz = self.fpz;
        let lpz = self.lpz;
        let jp = self.jp;
        let kp = self.kp;
        let count = lpz + 1 - fpz;

        self.p_x.as_ref().expect("VOL3D x was not preloaded");

        self.p_y.as_ref().expect("VOL3D y was not preloaded");

        self.p_z.as_ref().expect("VOL3D z was not preloaded");

        let p_vol = self.p_vol.as_ref().expect("VOL3D vol was not preloaded");

        let mut vol_reg = Region::<'_, _, OffsetStride1D<256>>::from(p_vol);

        offload! {
            kernel = vol3d,
            grid_dim = [count.div_ceil(256) as u32, 1, 1],
            block_dim = [256, 1, 1],
            args = (
                self.x as *const [Real; NNALLS],
                self.y as *const [Real; NNALLS],
                self.z as *const [Real; NNALLS],
                vol_reg,
                self.vnormq,
                jp,
                kp,
                fpz,
                lpz,
            ),
        };
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.vol as *const Real, NNALLS) }
    }

    fn tear_down(&mut self) -> f64 {
        // Drop preloads first. Dropping p_vol copies device output back to host.
        drop(self.p_vol.take());
        drop(self.p_x.take());
        drop(self.p_y.take());
        drop(self.p_z.take());

        let checksum = self.update_checksum();

        unsafe {
            if !self.x.is_null() {
                free(self.x);
                self.x = core::ptr::null_mut();
            }

            if !self.y.is_null() {
                free(self.y);
                self.y = core::ptr::null_mut();
            }

            if !self.z.is_null() {
                free(self.z);
                self.z = core::ptr::null_mut();
            }

            if !self.vol.is_null() {
                free(self.vol);
                self.vol = core::ptr::null_mut();
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
fn vol3d(
    x: *const [Real; NNALLS],
    y: *const [Real; NNALLS],
    z: *const [Real; NNALLS],
    mut vol: Region<Real, OffsetStride1D<256>>,
    vnormq: Real,
    jp: usize,
    kp: usize,
    fpz: usize,
    lpz: usize,
) {
    let idx = OffsetStride1D::<256>::index();
    let i = fpz + idx;

    if i > lpz {
        return;
    }

    if let Some(mut vvol) = vol.get_mut() {
        unsafe {
            let i0 = i;
            let i1 = i + 1;
            let i2 = i + jp;
            let i3 = i + 1 + jp;
            let i4 = i + kp;
            let i5 = i + 1 + kp;
            let i6 = i + jp + kp;
            let i7 = i + 1 + jp + kp;

            let x71 = (*x)[i7] - (*x)[i1];
            let x72 = (*x)[i7] - (*x)[i2];
            let x74 = (*x)[i7] - (*x)[i4];
            let x30 = (*x)[i3] - (*x)[i0];
            let x50 = (*x)[i5] - (*x)[i0];
            let x60 = (*x)[i6] - (*x)[i0];

            let y71 = (*y)[i7] - (*y)[i1];
            let y72 = (*y)[i7] - (*y)[i2];
            let y74 = (*y)[i7] - (*y)[i4];
            let y30 = (*y)[i3] - (*y)[i0];
            let y50 = (*y)[i5] - (*y)[i0];
            let y60 = (*y)[i6] - (*y)[i0];

            let z71 = (*z)[i7] - (*z)[i1];
            let z72 = (*z)[i7] - (*z)[i2];
            let z74 = (*z)[i7] - (*z)[i4];
            let z30 = (*z)[i3] - (*z)[i0];
            let z50 = (*z)[i5] - (*z)[i0];
            let z60 = (*z)[i6] - (*z)[i0];

            let mut xps = x71 + x60;
            let mut yps = y71 + y60;
            let mut zps = z71 + z60;

            let mut cyz = y72 * z30 - z72 * y30;
            let mut czx = z72 * x30 - x72 * z30;
            let mut cxy = x72 * y30 - y72 * x30;
            let mut v = xps * cyz + yps * czx + zps * cxy;

            xps = x72 + x50;
            yps = y72 + y50;
            zps = z72 + z50;

            cyz = y74 * z60 - z74 * y60;
            czx = z74 * x60 - x74 * z60;
            cxy = x74 * y60 - y74 * x60;
            v += xps * cyz + yps * czx + zps * cxy;

            xps = x74 + x30;
            yps = y74 + y30;
            zps = z74 + z30;

            cyz = y71 * z50 - z71 * y50;
            czx = z71 * x50 - x71 * z50;
            cxy = x71 * y50 - y71 * x50;
            v += xps * cyz + yps * czx + zps * cxy;

            vvol.set(fpz, v * vnormq);
        }
    }
}
