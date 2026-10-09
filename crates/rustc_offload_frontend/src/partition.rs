use crate::gpu::{block_idx, global_thread_dim, thread_idx};
use core::offload::PreloadMut;
pub use core::offload::{LaunchError, PartitioningStrategy, RawRegion, Region};
use core::prelude::v1::*;
use core::ptr::NonNull;

/// Builds a `Region` over a preloaded array. The preload only registers the host pointer with
/// the runtime; the kernel launch maps the region through that same pointer.
pub fn region_mut<'a, T, const N: usize, S: PartitioningStrategy>(
    p: &'a PreloadMut<'_, [T; N]>,
) -> Region<'a, T, S> {
    Region::new(unsafe { &mut *p.cpu_ptr })
}

// linear1d
#[derive(Copy, Clone)]
pub struct Linear1D;
unsafe impl PartitioningStrategy for Linear1D {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }

    fn index() -> usize {
        global_thread_dim().x
    }
    unsafe fn get<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::View<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &*ptr.as_ptr().add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &mut *ptr.as_ptr().add(idx) })
        } else {
            None
        }
    }
}

// linear2d
#[derive(Copy, Clone)]
pub struct Linear2D<const W: usize>;
unsafe impl<const W: usize> PartitioningStrategy for Linear2D<W> {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }

    fn index() -> usize {
        let tid = global_thread_dim();
        tid.y * W + tid.x
    }
    unsafe fn get<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::View<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &*ptr.as_ptr().add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &mut *ptr.as_ptr().add(idx) })
        } else {
            None
        }
    }
}

// stride1d
#[derive(Copy, Clone)]
pub struct Stride1D<const STRIDE: usize>;
unsafe impl<const STRIDE: usize> PartitioningStrategy for Stride1D<STRIDE> {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }

    fn index() -> usize {
        let bidx = block_idx().x;
        let tidx = thread_idx().x;
        bidx * STRIDE + tidx
    }
    unsafe fn get<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::View<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &*ptr.as_ptr().add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &mut *ptr.as_ptr().add(idx) })
        } else {
            None
        }
    }
}

// stride2d
#[derive(Copy, Clone)]
pub struct StrideViewMut<'a, T> {
    block_ptr: *mut T,
    stride: usize,
    _marker: core::marker::PhantomData<&'a mut T>,
}
impl<'a, T> StrideViewMut<'a, T> {
    pub fn set(&mut self, x: usize, y: usize, val: T) {
        unsafe {
            *self.block_ptr.add(y * self.stride + x) = val;
        }
    }
}

#[derive(Copy, Clone)]
pub struct Stride2D<
    const W: usize,
    const H: usize,
    const SX: usize,
    const SY: usize,
    const STRIDE: usize,
>;
unsafe impl<const W: usize, const H: usize, const SX: usize, const SY: usize, const STRIDE: usize>
    PartitioningStrategy for Stride2D<W, H, SX, SY, STRIDE>
{
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = StrideViewMut<'a, T>;

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }

    fn index() -> usize {
        let tid = global_thread_dim();
        tid.y * SY * STRIDE + tid.x * SX
    }
    unsafe fn get<'a, T>(_: NonNull<T>, _: usize) -> Option<Self::View<'a, T>> {
        unimplemented!()
    }
    unsafe fn get_mut<'a, T>(ptr: NonNull<T>, _: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        Some(StrideViewMut {
            block_ptr: unsafe { ptr.as_ptr().add(idx) },
            stride: STRIDE,
            _marker: core::marker::PhantomData,
        })
    }
}

// some custom patterns needed for `rust_perf`

// for vol3d
#[derive(Copy, Clone)]
pub struct OffsetStrideViewMut<'a, T> {
    base_ptr: *mut T,
    idx: usize,
    len: usize,
    _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T> OffsetStrideViewMut<'a, T> {
    pub fn set(&mut self, offset: usize, val: T) {
        if let Some(final_idx) = self.idx.checked_add(offset) {
            if final_idx < self.len {
                unsafe {
                    *self.base_ptr.add(final_idx) = val;
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct OffsetStride1D<const STRIDE: usize>;

unsafe impl<const STRIDE: usize> PartitioningStrategy for OffsetStride1D<STRIDE> {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = OffsetStrideViewMut<'a, T>;

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }

    fn index() -> usize {
        let bidx = block_idx().x;
        let tidx = thread_idx().x;
        bidx * STRIDE + tidx
    }

    unsafe fn get<'a, T>(_: NonNull<T>, _: usize) -> Option<Self::View<'a, T>> {
        unimplemented!("write only")
    }

    unsafe fn get_mut<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();

        if idx < len {
            Some(OffsetStrideViewMut {
                base_ptr: ptr.as_ptr(),
                idx,
                len,
                _marker: core::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

// for ltimes
#[derive(Copy, Clone)]
pub struct Stride3D<
    const BX: usize,
    const BY: usize,
    const BZ: usize,
    const MAX_X: usize,
    const MAX_Y: usize,
>;

unsafe impl<const BX: usize, const BY: usize, const BZ: usize, const MAX_X: usize, const MAX_Y: usize>
    PartitioningStrategy for Stride3D<BX, BY, BZ, MAX_X, MAX_Y>
{
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn check_launch(_len: usize, _grid: [u32; 3], _block: [u32; 3]) -> Result<(), LaunchError> {
        Ok(())
    }

    fn index() -> usize {
        let mx = (block_idx().x * BX) + thread_idx().x;
        let gy = (block_idx().y * BY) + thread_idx().y;
        let zz = (block_idx().z * BZ) + thread_idx().z;

        mx + MAX_X * (gy + MAX_Y * zz)
    }

    unsafe fn get<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::View<'a, T>> {
        let mx = (block_idx().x * BX) + thread_idx().x;
        let gy = (block_idx().y * BY) + thread_idx().y;
        let zz = (block_idx().z * BZ) + thread_idx().z;

        if mx < MAX_X && gy < MAX_Y {
            let idx = mx + MAX_X * (gy + MAX_Y * zz);
            if idx < len {
                return Some(unsafe { &*ptr.as_ptr().add(idx) });
            }
        }
        None
    }

    unsafe fn get_mut<'a, T>(ptr: NonNull<T>, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let mx = (block_idx().x * BX) + thread_idx().x;
        let gy = (block_idx().y * BY) + thread_idx().y;
        let zz = (block_idx().z * BZ) + thread_idx().z;

        if mx < MAX_X && gy < MAX_Y {
            let idx = mx + MAX_X * (gy + MAX_Y * zz);
            if idx < len {
                return Some(unsafe { &mut *ptr.as_ptr().add(idx) });
            }
        }
        None
    }
}
