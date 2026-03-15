#[cfg(target_os = "linux")]
extern crate libc;

#[cfg(target_os = "linux")]
pub const DATA_INIT_SEED: u32 = 4793;

#[cfg(target_os = "linux")]
static mut DATA_INIT_COUNT: usize = 0;

#[cfg(target_os = "linux")]
pub fn reset_data_init_count() {
    unsafe { DATA_INIT_COUNT = 0 };
}

#[cfg(target_os = "linux")]
pub fn inc_data_init_count() {
    unsafe { DATA_INIT_COUNT += 1 };
}

#[cfg(target_os = "linux")]
pub fn get_data_init_count() -> usize {
    unsafe { DATA_INIT_COUNT }
}

#[cfg(target_os = "linux")]
use crate::common::types::{Real, from_real, to_real};

#[cfg(target_os = "linux")]
pub unsafe fn init_data(ptr: *mut Real, len: usize) {
    let factor = if !get_data_init_count().is_multiple_of(2) {
        0.1
    } else {
        0.2
    };
    for i in 0..len {
        unsafe {
            *ptr.add(i) = to_real(factor * (i as f64 + 1.1) / (i as f64 + 1.12345));
        }
    }
    inc_data_init_count();
}

#[cfg(target_os = "linux")]
pub unsafe fn alloc_and_init_data(len: usize) -> *mut Real {
    let ptr = unsafe { alloc::<Real>(len) };
    unsafe { init_data(ptr, len) };
    ptr
}

#[cfg(target_os = "linux")]
pub unsafe fn init_data_const(ptr: *mut Real, len: usize, val: Real) {
    for i in 0..len {
        unsafe {
            *ptr.add(i) = val;
        }
    }
    inc_data_init_count();
}

#[cfg(target_os = "linux")]
pub unsafe fn alloc_and_init_data_const(len: usize, val: Real) -> *mut Real {
    let ptr = unsafe { alloc::<Real>(len) };
    unsafe { init_data_const(ptr, len, val) };
    ptr
}

#[cfg(target_os = "linux")]
pub unsafe fn init_data_rand_value(ptr: *mut Real, len: usize) {
    unsafe { libc::srand(DATA_INIT_SEED) };
    for i in 0..len {
        let r = unsafe { libc::rand() } as f64;
        let rmax = libc::RAND_MAX as f64;
        unsafe {
            *ptr.add(i) = to_real(r / rmax);
        }
    }
    inc_data_init_count();
}

#[cfg(target_os = "linux")]
pub unsafe fn alloc_and_init_data_rand_value(len: usize) -> *mut Real {
    let ptr = unsafe { alloc::<Real>(len) };
    unsafe { init_data_rand_value(ptr, len) };
    ptr
}

#[cfg(target_os = "linux")]
pub fn init_data_scalar() -> Real {
    let factor = if !get_data_init_count().is_multiple_of(2) {
        0.1
    } else {
        0.2
    };
    let val = to_real(factor * 1.1 / 1.12345);
    inc_data_init_count();
    val
}

#[cfg(target_os = "linux")]
pub unsafe fn calc_checksum(ptr: *const Real, len: usize) -> f64 {
    let mut chk = KahanSum::new(0.0);

    for j in 0..len {
        let val = from_real(unsafe { *ptr.add(j) });
        chk.add(calc_multiplier(j as f64, if val >= 0.0 { 1.0 } else { 0.5 }) * val.abs());
    }

    chk.get()
}

#[cfg(target_os = "linux")]
#[inline]
pub fn calc_multiplier(index: f64, offset: f64) -> f64 {
    const PI_INV: f64 = core::f64::consts::FRAC_1_PI;
    const HALF: f64 = 0.5;

    let val = (index + offset) * PI_INV;

    val - core::f64::math::floor(val) + HALF
}

#[cfg(target_os = "linux")]
pub struct KahanSum {
    sum: f64,
    c: f64,
}

#[cfg(target_os = "linux")]
impl KahanSum {
    pub const fn new(val: f64) -> Self {
        Self { sum: val, c: 0.0 }
    }

    pub fn add(&mut self, val: f64) {
        let y = val - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    pub fn get(&self) -> f64 {
        self.sum
    }
}

#[cfg(target_os = "linux")]
pub unsafe fn alloc<T>(len: usize) -> *mut T {
    let bytes = len * core::mem::size_of::<T>();
    let ptr = unsafe { libc::malloc(bytes) } as *mut T;
    if ptr.is_null() {
        panic!("Failed to allocate {bytes} bytes")
    }
    unsafe { core::ptr::write_bytes(ptr, 0, len) };
    ptr
}

#[cfg(target_os = "linux")]
pub unsafe fn free<T>(ptr: *mut T) {
    unsafe { libc::free(ptr as *mut libc::c_void) };
}
