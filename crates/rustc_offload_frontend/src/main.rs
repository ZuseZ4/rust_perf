#![allow(internal_features)]
#![allow(linker_messages)]
#![allow(improper_ctypes)]
#![allow(improper_gpu_kernel_arg)]
#![allow(improper_ctypes_definitions)]
#![feature(gpu_offload)]
#![cfg_attr(target_os = "linux", feature(core_intrinsics))]
#![cfg_attr(target_arch = "nvptx64", feature(abi_gpu_kernel))]
#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", no_main)]

#[cfg(target_os = "linux")]
extern crate libc;

use rustc_offload_frontend::offload_kernel;
use rustc_offload_frontend::partition::{Linear1D, Linear2D, Region, Stencil2D, Stride2D};

#[offload_kernel]
fn linear1d(x: &mut Region<f64, Linear1D>) {
    if let Some(e) = x.get_mut() {
        *e = 42.0;
    }
}

#[offload_kernel]
fn stride2d(grid: &mut Region<f64, Stride2D<2, 2, 4, 4>>) {
    if let Some(mut view) = grid.get_mut() {
        view.set(0, 0, 42.0);
        view.set(1, 1, 42.0);
    }
}

#[offload_kernel]
fn conv_blur2d(input: &Region<f64, Stencil2D<1>>, output: &mut Region<f64, Linear2D>) {
    if let (Some(in_view), Some(out_cell)) = (input.get(), output.get_mut()) {
        let mut sum = 0.0;

        for dy in -1..=1 {
            for dx in -1..=1 {
                sum += in_view.get_neighbour(dx, dy);
            }
        }

        *out_cell = sum / 9.0;
    }
}

#[cfg(target_os = "linux")]
fn main() {
    use rustc_offload_frontend::offload;

    // linear1d
    let mut x = [0.0f64; 256];
    let mut reg = Region::<_, Linear1D>::new(&mut x, ());
    // core::intrinsics::offload::<_, _, ()>(linear1d, [1, 1, 1], [256, 1, 1], (&mut reg,));
    offload! {
        kernel = linear1d,
        grid_dim = [256, 1, 1],
        args = (&mut reg,),
    };
    for i in 0..x.len() {
        assert_eq!(x[i], 42.0 as f64);
    }

    // stride2d
    let mut blocks = [0.0; 64];
    let mut reg_stride = Region::<_, Stride2D<2, 2, 4, 4>>::new(&mut blocks, (8, 8));
    // core::intrinsics::offload::<_, _, ()>(stride2d, [1, 1, 1], [2, 2, 1], (&mut reg_stride,));
    offload! {
        kernel = stride2d,
        block_dim = [2, 2, 1],
        args = (&mut reg_stride,),
    };
    // thread (0, 0, 0) takes a 2x2 block and writes on the diagonal elements
    assert_eq!(blocks[0], 42.0);
    assert_eq!(blocks[9], 42.0);

    // conv_blur2d
    let mut input_data = [
        0.0, 0.0, 0.0, 0.0, //
        0.0, 9.0, 9.0, 0.0, //
        0.0, 9.0, 9.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
    ];
    let mut output_data = [0.0f64; 16];

    let reg_input = Region::<_, Stencil2D<1>>::new(&mut input_data, (4, 4));
    let mut reg_output = Region::<_, Linear2D>::new(&mut output_data, (4, 4));

    offload! {
        kernel = conv_blur2d,
        block_dim = [4, 4, 1],
        args = (&reg_input, &mut reg_output,),
    };

    println!("{:#?}", output_data);

    println!("all checks passed!");
}
