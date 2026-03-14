#![allow(internal_features)]
#![allow(non_snake_case)]
#![allow(clippy::deref_addrof)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
#![allow(unused_features)]
#![feature(abi_gpu_kernel)]
#![feature(core_float_math)]
#![feature(core_intrinsics)]
#![feature(float_algebraic)]
#![feature(rustc_attrs)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

pub mod apps;
pub mod common;

#[cfg(all(target_os = "linux", feature = "del_dot_vec_2d"))]
use apps::del_dot_vec_2d::DelDotVec2D;
#[cfg(all(target_os = "linux", feature = "energy"))]
use apps::energy::Energy;
#[cfg(all(target_os = "linux", feature = "fir"))]
use apps::fir::Fir;
#[cfg(all(target_os = "linux", feature = "mass3dea"))]
use apps::mass3dea::Mass3DEA;

#[cfg(all(target_os = "linux", feature = "energy"))]
static mut K_ENERGY: Energy = Energy::INIT;
#[cfg(all(target_os = "linux", feature = "fir"))]
static mut K_FIR: Fir = Fir::INIT;
#[cfg(all(target_os = "linux", feature = "del_dot_vec_2d"))]
static mut K_DEL: DelDotVec2D = DelDotVec2D::INIT;
#[cfg(all(target_os = "linux", feature = "mass3dea"))]
static mut K_MASS3DEA: Mass3DEA = Mass3DEA::INIT;

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
fn main() {
    use crate::common::executor::{Executor, KernelResult, MAX_KERNELS};
    use crate::common::kernel_base::KernelBase;
    use core::mem::MaybeUninit;

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
    #[cfg(feature = "mass3dea")]
    {
        k_links[count] = Some(unsafe { &mut *(&raw mut K_MASS3DEA) });
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
}
