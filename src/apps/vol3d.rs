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
use crate::common::data_utils::{alloc_and_init_data_const, calc_checksum, free};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Vol3D {
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
}

#[cfg(target_os = "linux")]
impl Vol3D {
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
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Vol3D {
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
            self.x = alloc_and_init_data_const(nnalls, to_real(0.0));
            self.y = alloc_and_init_data_const(nnalls, to_real(0.0));
            self.z = alloc_and_init_data_const(nnalls, to_real(0.0));
            self.vol = alloc_and_init_data_const(nnalls, to_real(0.0));

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
    }

    fn run_kernel(&mut self) {
        let fpz = self.fpz;
        let lpz = self.lpz;
        let jp = self.jp;
        let kp = self.kp;
        let count = lpz + 1 - fpz;

        core::intrinsics::offload::<_, _, ()>(
            _vol3d,
            [((count + 255) / 256) as u32, 1, 1],
            [256, 1, 1],
            (
                self.x as *const [Real; 1124864],
                self.y as *const [Real; 1124864],
                self.z as *const [Real; 1124864],
                self.vol as *mut [Real; 1124864],
                self.vnormq,
                jp,
                kp,
                fpz,
                lpz,
            ),
        );
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.vol as *const Real, 1124864) }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.x);
            free(self.y);
            free(self.z);
            free(self.vol);
            self.x = core::ptr::null_mut();
            self.y = core::ptr::null_mut();
            self.z = core::ptr::null_mut();
            self.vol = core::ptr::null_mut();
        }
        self.n = 0;
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _vol3d(
        x: *const [Real; 1124864],
        y: *const [Real; 1124864],
        z: *const [Real; 1124864],
        vol: *mut [Real; 1124864],
        vnormq: Real,
        jp: usize,
        kp: usize,
        fpz: usize,
        lpz: usize,
    );
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _vol3d(
    x: *const [Real; 1124864],
    y: *const [Real; 1124864],
    z: *const [Real; 1124864],
    vol: *mut [Real; 1124864],
    vnormq: Real,
    jp: usize,
    kp: usize,
    fpz: usize,
    lpz: usize,
) {
    let idx = unsafe { (block_idx_x() * 256 + thread_idx_x()) as usize };
    let i = fpz + idx;
    if i > lpz {
        return;
    }

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

        (*vol)[i] = v * vnormq;
    }
}
