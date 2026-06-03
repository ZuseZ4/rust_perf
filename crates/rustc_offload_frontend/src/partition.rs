use crate::gpu::{ global_thread_dim, block_idx, block_dim, thread_idx };
use core::convert::From;
use core::prelude::v1::*;
use core::offload::offload::PreloadMut;

pub unsafe trait PartitioningStrategy {
    type View<'a, T: 'a>;
    type ViewMut<'a, T: 'a>;

    fn index() -> usize;
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>>;
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>>;
}

#[derive(Debug, Copy, Clone)]
pub struct Region<'a, T, S: PartitioningStrategy> {
    ptr: *mut T,
    len: usize,
    _marker: core::marker::PhantomData<(&'a mut [T], S)>,
}

impl<'a, T, const N: usize, S> From<&PreloadMut<'a, [T; N]>> for Region<'a, T, S>
where
    S: PartitioningStrategy,
{
    fn from(p: &PreloadMut<'a, [T; N]>) -> Self {
        Self {
            ptr: p.cpu_ptr as *mut T,
            len: N,
            _marker: core::marker::PhantomData,
        }
    }
}

pub struct RawRegion<'a, T> {
    pub ptr: *mut T,
    pub len: usize,
    _marker: core::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T> From<&'a mut [T]> for RawRegion<'a, T> {
    fn from(data: &'a mut [T]) -> Self {
        Self {
            ptr: data.as_mut_ptr(),
            len: data.len(),
            _marker: core::marker::PhantomData,
        }
    }
}

impl<'a, T, const N: usize> From<&'a mut [T; N]> for RawRegion<'a, T> {
    fn from(data: &'a mut [T; N]) -> Self {
        Self {
            ptr: data.as_mut_ptr(),
            len: N,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    pub fn new<D>(data: D) -> Self
    where
        D: Into<RawRegion<'a, T>>,
    {
        let raw = data.into();
        Self {
            ptr: raw.ptr,
            len: raw.len,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn get(&self) -> Option<S::View<'_, T>> {
        unsafe { S::get(self.ptr as *const T, self.len) }
    }

    pub fn get_mut(&mut self) -> Option<S::ViewMut<'_, T>> {
        unsafe { S::get_mut(self.ptr, self.len) }
    }
}

// linear1d
#[derive(Debug, Copy, Clone)]
pub struct Linear1D;
unsafe impl PartitioningStrategy for Linear1D {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        global_thread_dim().x
    }
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &*ptr.add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &mut *ptr.add(idx) })
        } else {
            None
        }
    }
}

// linear2d
#[derive(Debug, Copy, Clone)]
pub struct Linear2D<const W: usize>;
unsafe impl<const W: usize> PartitioningStrategy for Linear2D<W> {
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        let tid = global_thread_dim();
        tid.y * W + tid.x
    }
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &*ptr.add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &mut *ptr.add(idx) })
        } else {
            None
        }
    }
}

// stride1d
#[derive(Debug, Copy, Clone)]
pub struct Stride1D<
    const STRIDE: usize,
>;
unsafe impl<const STRIDE: usize>
    PartitioningStrategy for Stride1D<STRIDE>
{
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    fn index() -> usize {
        let bidx = block_idx().x;
        let tidx = thread_idx().x;
        bidx * STRIDE + tidx
    }
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &*ptr.add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        if idx < len {
            Some(unsafe { &mut *ptr.add(idx) })
        } else {
            None
        }
    }
}


// stride2d
#[derive(Debug, Copy, Clone)]
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

#[derive(Debug, Copy, Clone)]
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

    fn index() -> usize {
        let tid = global_thread_dim();
        tid.y * SY * STRIDE + tid.x * SX
    }
    unsafe fn get<'a, T>(_: *const T, _: usize) -> Option<Self::View<'a, T>> {
        unimplemented!()
    }
    unsafe fn get_mut<'a, T>(ptr: *mut T, _: usize) -> Option<Self::ViewMut<'a, T>> {
        let idx = Self::index();
        Some(StrideViewMut {
            block_ptr: unsafe { ptr.add(idx) },
            stride: STRIDE,
            _marker: core::marker::PhantomData,
        })
    }
}
