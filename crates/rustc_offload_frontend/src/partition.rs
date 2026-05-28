use crate::gpu::global_thread_dim;
use core::convert::From;
use core::prelude::v1::*;

pub unsafe trait PartitioningStrategy {
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

impl<'a, T> From<&'a [T]> for RawRegion<'a, T> {
    fn from(data: &'a [T]) -> Self {
        Self {
            ptr: data.as_ptr() as *mut T,
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

impl<'a, T, const N: usize> From<&'a [T; N]> for RawRegion<'a, T> {
    fn from(data: &'a [T; N]) -> Self {
        Self {
            ptr: data.as_ptr() as *mut T,
            len: N,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    pub fn new<D>(data: D, shape: S::Shape) -> Self
    where
        D: Into<RawRegion<'a, T>>,
    {
        let raw = data.into();
        Self {
            ptr: raw.ptr,
            len: raw.len,
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
unsafe impl PartitioningStrategy for Linear1D {
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

// linear2d
pub struct Linear2D;
unsafe impl PartitioningStrategy for Linear2D {
    type Shape = (usize, usize);
    type View<'a, T: 'a> = &'a T;
    type ViewMut<'a, T: 'a> = &'a mut T;

    unsafe fn get<'a, T>(
        ptr: *const T,
        len: usize,
        shape: Self::Shape,
    ) -> Option<Self::View<'a, T>> {
        let tid = global_thread_dim();
        let idx = tid.y * shape.0 + tid.x;
        if idx < len {
            Some(unsafe { &*ptr.add(idx) })
        } else {
            None
        }
    }
    unsafe fn get_mut<'a, T>(
        ptr: *mut T,
        len: usize,
        shape: Self::Shape,
    ) -> Option<Self::ViewMut<'a, T>> {
        let tid = global_thread_dim();
        let idx = tid.y * shape.0 + tid.x;
        if idx < len {
            Some(unsafe { &mut *ptr.add(idx) })
        } else {
            None
        }
    }
}

// stencil2d
pub struct Stencil2D<const RADIUS: usize>;

pub struct StencilView<'a, T> {
    base_ptr: *const T,
    center_idx: usize,
    cols: usize,
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T> StencilView<'a, T> {
    pub fn get_neighbour(&self, ox: isize, oy: isize) -> &T {
        unsafe {
            &*self
                .base_ptr
                .offset((self.center_idx as isize) + (oy * self.cols as isize) + ox)
        }
    }
}

unsafe impl<const R: usize> PartitioningStrategy for Stencil2D<R> {
    type Shape = (usize, usize);
    type View<'a, T: 'a> = StencilView<'a, T>;
    type ViewMut<'a, T: 'a> = core::marker::PhantomData<&'a mut T>;

    unsafe fn get<'a, T>(
        ptr: *const T,
        len: usize,
        shape: Self::Shape,
    ) -> Option<Self::View<'a, T>> {
        let (cols, _rows) = shape;
        let tid = global_thread_dim();

        let center_idx = tid.y * cols + tid.x;

        if center_idx < len {
            Some(StencilView {
                base_ptr: ptr,
                center_idx,
                cols,
                _marker: core::marker::PhantomData,
            })
        } else {
            None
        }
    }

    unsafe fn get_mut<'a, T>(_: *mut T, _: usize, _: Self::Shape) -> Option<Self::ViewMut<'a, T>> {
        None
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
unsafe impl<const W: usize, const H: usize, const SX: usize, const SY: usize> PartitioningStrategy
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
