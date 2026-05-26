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

use rustc_offload_frontend::{offload_kernel};
use rustc_offload_frontend::partition::{Region, Linear1D, Stencil2D, Stride2D};

#[offload_kernel]
fn linear1d(x: &mut Region<f64, Linear1D>) {
    if let Some(e) = x.get_mut() {
        *e = 42.0;
    }
}

#[offload_kernel]
fn stencil2d(grid: &mut Region<f64, Stencil2D<1>>) {
    if let Some(mut view) = grid.get_mut() {
        let mid = *view.get_neighbour(0, 0);
        let left = *view.get_neighbour(-1, 0);
        let right = *view.get_neighbour(1, 0);
        view.set_center((left + mid + right) / 3.0);
    }
}

#[offload_kernel]
fn stride2d(grid: &mut Region<f64, Stride2D<2, 2, 4, 4>>) {
    if let Some(mut view) = grid.get_mut() {
        view.set(0, 0, 42.0);
        view.set(1, 1, 42.0);
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

    // stencil2d
    let mut grid = [
        1.0, 1.0, 1.0, 1.0, //
        1.0, 4.0, 1.0, 1.0, // cargo fmt don't merge this lines
        1.0, 1.0, 1.0, 1.0, //
        1.0, 1.0, 1.0, 1.0,
    ];
    let mut reg_stencil = Region::<_, Stencil2D<1>>::new(&mut grid, (4, 4));
    // core::intrinsics::offload::<_, _, ()>(stencil2d, [1, 1, 1], [2, 2, 1], (&mut reg_stencil,));
    offload! {
        kernel = stencil2d,
        block_dim = [2, 2, 1],
        args = (&mut reg_stencil,),
    };
    // thread (0, 0, 0) will have center on (x, y) = 1 (index = 5), so (1 + 4 + 1) / 3 = 2
    assert_eq!(grid[5], 2.0);

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

    println!("all checks passed!");
}
