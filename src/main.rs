#![allow(internal_features)]
#![allow(linker_messages)]
#![allow(improper_ctypes)]
#![allow(improper_gpu_kernel_arg)]
#![allow(improper_ctypes_definitions)]
#![feature(gpu_offload)]
#![cfg_attr(target_os = "linux", feature(core_intrinsics, offload))]
#![cfg_attr(
    any(target_arch = "nvptx64", target_arch = "amdgpu"),
    feature(abi_gpu_kernel)
)]
#![cfg_attr(any(target_arch = "nvptx64", target_arch = "amdgpu"), no_std)]
#![cfg_attr(any(target_arch = "nvptx64", target_arch = "amdgpu"), no_main)]

use rust_perf;

#[cfg(target_os = "linux")]
extern crate libc;

#[cfg(all(target_os = "linux", feature = "del_dot_vec_2d"))]
use rust_perf::apps::del_dot_vec_2d::DelDotVec2D;
#[cfg(all(target_os = "linux", feature = "energy"))]
use rust_perf::apps::energy::Energy;
#[cfg(all(target_os = "linux", feature = "fir"))]
use rust_perf::apps::fir::Fir;

#[cfg(all(target_os = "linux", feature = "ltimes"))]
use rust_perf::apps::ltimes::LTimes;

#[cfg(all(target_os = "linux", feature = "matvec_3d_stencil"))]
use rust_perf::apps::matvec_3d_stencil::Matvec3DStencil;

#[cfg(all(target_os = "linux", feature = "pressure"))]
use rust_perf::apps::pressure::Pressure;

#[cfg(all(target_os = "linux", feature = "vol3d"))]
use rust_perf::apps::vol3d::Vol3D;

#[cfg(all(target_os = "linux", feature = "energy"))]
static mut K_ENERGY: Energy = Energy::INIT;
#[cfg(all(target_os = "linux", feature = "fir"))]
static mut K_FIR: Fir = Fir::INIT;
#[cfg(all(target_os = "linux", feature = "del_dot_vec_2d"))]
static mut K_DEL: DelDotVec2D = DelDotVec2D::INIT;
#[cfg(all(target_os = "linux", feature = "ltimes"))]
static mut K_LTIMES: LTimes = LTimes::INIT;
#[cfg(all(target_os = "linux", feature = "matvec_3d_stencil"))]
static mut K_MATVEC3DSTENCIL: Matvec3DStencil = Matvec3DStencil::INIT;
#[cfg(all(target_os = "linux", feature = "pressure"))]
static mut K_PRESSURE: Pressure = Pressure::INIT;
#[cfg(all(target_os = "linux", feature = "vol3d"))]
static mut K_VOL3D: Vol3D = Vol3D::INIT;

#[cfg(target_os = "linux")]
fn main() {
    use core::mem::MaybeUninit;
    use rust_perf::common::executor::{Executor, KernelResult, MAX_KERNELS};
    use rust_perf::common::kernel_base::KernelBase;

    let mut k_links: [Option<&mut dyn KernelBase>; MAX_KERNELS] = [const { None }; MAX_KERNELS];
    let mut count = 0;

    #[cfg(feature = "energy")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_ENERGY) });
        count += 1;
    }
    #[cfg(feature = "fir")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_FIR) });
        count += 1;
    }
    #[cfg(feature = "del_dot_vec_2d")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_DEL) });
        count += 1;
    }
    #[cfg(feature = "ltimes")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_LTIMES) });
        count += 1;
    }
    #[cfg(feature = "matvec_3d_stencil")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_MATVEC3DSTENCIL) });
        count += 1;
    }

    #[cfg(feature = "pressure")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_PRESSURE) });
        count += 1;
    }

    #[cfg(feature = "vol3d")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_VOL3D) });
        count += 1;
    }

    let mut kernel_refs: [MaybeUninit<&mut dyn KernelBase>; MAX_KERNELS] =
        [const { MaybeUninit::uninit() }; MAX_KERNELS];

    for i in 0..count {
        kernel_refs[i] = MaybeUninit::new(k_links[i].take().unwrap());
    }

    let kernels_slice = unsafe {
        core::slice::from_raw_parts_mut(kernel_refs.as_mut_ptr() as *mut &mut dyn KernelBase, count)
    };

    let mut suite = Executor::new(kernels_slice);

    static mut RESULT_BUF: [MaybeUninit<KernelResult>; MAX_KERNELS] =
        [const { MaybeUninit::uninit() }; MAX_KERNELS];

    let results = suite.run_suite(unsafe { &mut *(&raw mut RESULT_BUF) });

    Executor::print_report(results);
    Executor::export_csv(results, c"results.csv".as_ptr());
}
