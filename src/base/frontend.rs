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

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/*
* macro
*/
#[macro_export]
macro_rules! offload {
    ( $($field:ident = $val:expr),* $(,)? ) => {
        $crate::offload!(@munch
            [ $($field = $val),* ];
            kernel = NONE;
            grid_dim = ([1, 1, 1]);
            block_dim = ([1, 1, 1]);
            args = NONE
        );
    };

    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = (SOME $val); grid_dim = $g; block_dim = $b; args = $a);
    };
    (@munch [grid_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = ($val); block_dim = $b; args = $a);
    };
    (@munch [block_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = ($val); args = $a);
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = $b; args = (SOME $val));
    };

    (@munch [$invalid:ident = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        compile_error!(concat!("unkown field ", stringify!($invalid)));
    };

    (@munch []; kernel = NONE; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        compile_error!("missing `kernel`");
    };
    (@munch []; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = NONE) => {
        compile_error!("missing `args`");
    };
    (@munch []; kernel = (SOME $kernel:expr); grid_dim = ($grid_dim:expr); block_dim = ($block_dim:expr); args = (SOME $args:expr)) => {
        core::intrinsics::offload::<_, _, ()>(
            $kernel,
            $grid_dim,
            $block_dim,
            $args,
        )
    };
}

/*
* library
*/

// index helpers for mental sanity xd
#[derive(Clone, Copy)]
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

fn global_thread_dim() -> Dim3 {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        use core::arch::nvptx::*;
        Dim3 {
            x: (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize,
            y: (_block_idx_y() * _block_dim_y() + _thread_idx_y()) as usize,
            z: (_block_idx_z() * _block_dim_z() + _thread_idx_z()) as usize,
        }
    }
    #[cfg(target_os = "linux")]
    Dim3 { x: 0, y: 0, z: 0 }
}

pub trait PartitioningStrategy {
    type Shape: Copy;
    type View<'a, T: 'a>;
    type ViewMut<'a, T: 'a>;

    unsafe fn get<'a, T>(
        ptr: *const T,
        len: usize,
        shape: Self::Shape,
    ) -> Option<Self::View<'a, T>>;
    unsafe fn get_mut<'a, T>(
        ptr: *mut T,
        len: usize,
        shape: Self::Shape,
    ) -> Option<Self::ViewMut<'a, T>>;
}

pub struct Region<'a, T, S: PartitioningStrategy> {
    ptr: *mut T,
    len: usize,
    pub shape: S::Shape,
    _marker: core::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    pub fn new(data: &'a mut [T], shape: S::Shape) -> Self {
        Self {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            shape,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn get(&self) -> Option<S::View<'_, T>> {
        unsafe { S::get(self.ptr as *const T, self.len, self.shape) }
    }

    pub fn get_mut(&mut self) -> Option<S::ViewMut<'_, T>> {
        unsafe { S::get_mut(self.ptr, self.len, self.shape) }
    }
}

// linear1d
pub struct Linear1D;
impl PartitioningStrategy for Linear1D {
    type Shape = ();
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    unsafe fn get<'a, T>(ptr: *const T, len: usize, _: Self::Shape) -> Option<Self::View<'a, T>> {
        let tid = global_thread_dim().x;
        if tid < len {
            Some(unsafe { &*ptr.add(tid) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(
        ptr: *mut T,
        len: usize,
        _: Self::Shape,
    ) -> Option<Self::ViewMut<'a, T>> {
        let tid = global_thread_dim().x;
        if tid < len {
            Some(unsafe { &mut *ptr.add(tid) })
        } else {
            None
        }
    }
}

// stencil2d
pub struct StencilViewMut<'a, T> {
    base_ptr: *mut T,
    center_idx: usize,
    cols: usize,
    _marker: core::marker::PhantomData<&'a mut T>,
}
impl<'a, T> StencilViewMut<'a, T> {
    pub fn set_center(&mut self, val: T) {
        unsafe {
            *self.base_ptr.add(self.center_idx) = val;
        }
    }

    pub fn get_neighbour(&self, ox: isize, oy: isize) -> &T {
        unsafe {
            &*self
                .base_ptr
                .offset((self.center_idx as isize) + (oy * self.cols as isize) + ox)
        }
    }
}

pub struct Stencil2D<const RADIUS: usize>;
impl<const R: usize> PartitioningStrategy for Stencil2D<R> {
    type Shape = (usize, usize);
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = StencilViewMut<'a, T>;

    unsafe fn get<'a, T>(_: *const T, _: usize, _: Self::Shape) -> Option<Self::View<'a, T>> {
        unimplemented!()
    }
    unsafe fn get_mut<'a, T>(
        ptr: *mut T,
        len: usize,
        shape: Self::Shape,
    ) -> Option<Self::ViewMut<'a, T>> {
        let tid = global_thread_dim();
        let x = tid.x + R;
        let y = tid.y + R;
        if x < shape.0 - R && y < shape.1 - R {
            let center_idx = y * shape.0 + x;
            if center_idx < len {
                Some(StencilViewMut {
                    base_ptr: ptr,
                    center_idx,
                    cols: shape.0,
                    _marker: core::marker::PhantomData,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

// stride
pub struct StrideViewMut<'a, T> {
    block_ptr: *mut T,
    stride: usize,
    width: usize,
    height: usize,
    _marker: core::marker::PhantomData<&'a mut T>,
}
impl<'a, T> StrideViewMut<'a, T> {
    pub fn set(&mut self, x: usize, y: usize, val: T) {
        if x < self.width && y < self.height {
            unsafe {
                *self.block_ptr.add(y * self.stride + x) = val;
            }
        }
    }
}

pub struct Stride2D<const W: usize, const H: usize, const SX: usize, const SY: usize>;
impl<const W: usize, const H: usize, const SX: usize, const SY: usize> PartitioningStrategy
    for Stride2D<W, H, SX, SY>
{
    type Shape = (usize, usize);
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = StrideViewMut<'a, T>;

    unsafe fn get<'a, T>(_: *const T, _: usize, _: Self::Shape) -> Option<Self::View<'a, T>> {
        unimplemented!()
    }
    unsafe fn get_mut<'a, T>(
        ptr: *mut T,
        _: usize,
        shape: Self::Shape,
    ) -> Option<Self::ViewMut<'a, T>> {
        let tid = global_thread_dim();
        let start_x = tid.x * SX;
        let start_y = tid.y * SY;
        if start_x + W <= shape.0 && start_y + H <= shape.1 {
            Some(StrideViewMut {
                block_ptr: unsafe { ptr.add(start_y * shape.0 + start_x) },
                stride: shape.0,
                width: W,
                height: H,
                _marker: core::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

// source code
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
#[unsafe(no_mangle)]
fn main() {
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
    // thread (0, 0, 0) will have center on (x, y) = 1 (index = 5), so (1 + 4 + 1) / 3 = 20
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
}
