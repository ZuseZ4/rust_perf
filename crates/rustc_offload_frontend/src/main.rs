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

#[offload_kernel]
fn saxpy_kernel(alpha: f32, x: &Region<f32, Linear1D>, y: &mut Region<f32, Linear1D>) {
    if let (Some(val_x), Some(val_y)) = (x.get(), y.get_mut()) {
        *val_y = alpha * (*val_x) + (*val_y);
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
    println!("GPU bits: {:064b} value: {:?}", x[0].to_bits(), x[0]);
    println!("CPU bits: {:064b} value: {:?}", 42.0f64.to_bits(), 42.0);
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
    let mut input = [
        0.0, 0.0, 0.0, 0.0, //
        0.0, 9.0, 9.0, 0.0, //
        0.0, 9.0, 9.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
    ];
    let mut output = [0.0f64; 16];

    let reg_input = Region::<_, Stencil2D<1>>::new(&mut input, (4, 4));
    let mut reg_output = Region::<_, Linear2D>::new(&mut output, (4, 4));
    offload! {
        kernel = conv_blur2d,
        block_dim = [4, 4, 1],
        args = (&reg_input, &mut reg_output,),
    };

    let expected = [
        1.0, 2.0, 2.0, 1.0, //
        2.0, 4.0, 4.0, 2.0, //
        2.0, 4.0, 4.0, 2.0, //
        1.0, 2.0, 2.0, 1.0, //
    ];
    assert_eq!(output, expected);

    // saxpy
    const N: usize = 512;
    let alpha: f32 = 2.5;
    let x: [f32; N] = [2.0; N];
    let mut y: [f32; N] = [1.0; N];

    let reg_x = Region::<_, Linear1D>::new(&x, ());
    let mut reg_y = Region::<_, Linear1D>::new(&mut y, ());

    offload! {
        kernel = saxpy_kernel,
        grid_dim = [N as u32, 1, 1],        args = (alpha, &reg_x, &mut reg_y,),
    };

    for i in 0..N {
        assert_eq!(y[i], 6.0f32);
    }

    println!("all checks passed!");
}
