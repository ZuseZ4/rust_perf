#![allow(internal_features)]
#![allow(linker_messages)]
#![allow(improper_ctypes)]
#![allow(improper_gpu_kernel_arg)]
#![allow(improper_ctypes_definitions)]
#![feature(gpu_offload, offload)]
#![feature(float_algebraic, core_float_math)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx, abi_gpu_kernel))]
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![feature(rustc_attrs, core_intrinsics)]

pub mod apps;
pub mod common;
