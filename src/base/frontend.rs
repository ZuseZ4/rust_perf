#![allow(internal_features)]
#![allow(linker_messages)]
#![allow(improper_ctypes)]
#![allow(improper_gpu_kernel_arg)]
#![allow(improper_ctypes_definitions)]
#![feature(gpu_offload)]
#![cfg_attr(target_os = "linux", feature(core_intrinsics))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch = "nvptx64", feature(abi_gpu_kernel))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

use core::offload::offload_kernel;

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// library
struct Linear1D;

pub trait PartitioningStrategy {
    fn get_mut<'a, T>(data: &'a mut [T]) -> Option<&'a mut T>;
}

impl PartitioningStrategy for Linear1D {
    fn get_mut<'a, T>(data: &'a mut [T]) -> Option<&'a mut T> {
        #[cfg(target_arch = "nvptx64")]
        let i = unsafe { (block_idx_x() * block_dim_x() + thread_idx_x()) as usize };
        #[cfg(target_os = "linux")]
        let i = 0;
        if i < data.len() {
            Some(&mut data[i])
        } else {
            None
        }
    }
}

struct Region<'a, T, S> {
    data: &'a mut [T],
    _marker: core::marker::PhantomData<S>,
}

impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self {
            data,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        S::get_mut(self.data)
    }
}

// source code
#[offload_kernel]
fn foo(x: &mut Region<f64, Linear1D>) {
    if let Some(e) = x.get_mut() {
        *e = 42.0 as f64;
    }
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
fn main() {
    let mut x = [0.0f64; 256];
    let mut reg = Region::<_, Linear1D>::new(&mut x);
    core::intrinsics::offload::<_, _, ()>(foo, [1, 1, 1], [256, 1, 1], (&mut reg,));
    for i in 0..x.len() {
        assert_eq!(x[i], 42.0 as f64);
    }
}
