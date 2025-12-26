#![allow(internal_features, non_camel_case_types, non_snake_case)]
#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum rocblas_operation {
    rocblas_operation_none = 111,
    rocblas_operation_transpose = 112,
    rocblas_operation_conjugate_transpose = 113,
}

// Usually from rocblas-types.h; typically i32 on host.
pub type rocblas_int = i32;

// Usually from rocblas-types.h; typically an opaque pointer type.
#[repr(C)]
pub struct rocblas_handle__ {
    _private: [u8; 0],
}
pub type rocblas_handle = *mut rocblas_handle__;

// You likely already have the full set of status codes elsewhere.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum rocblas_status {
    rocblas_status_success = 0,
    // ... add other rocblas_status_* as needed
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
unsafe fn main() {
    let mut handle: rocblas_handle = core::ptr::null_mut();
    let st = unsafe { rocblas_create_handle(&mut handle as *mut rocblas_handle) };
    assert_eq!(st, rocblas_status::rocblas_status_success);

    // 2) dimensions and op
    let trans = rocblas_operation::rocblas_operation_none;
    let m: rocblas_int = 2;
    let n: rocblas_int = 3;
    let incx: rocblas_int = 1;
    let incy: rocblas_int = 1;

    let A = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let x = [1.0, 1.0, 1.0];
    let y = [0.0, 0.0];

    let lda = 1;

    // 3) scalars (host pointers)
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    unsafe {
        core::intrinsics::offload::<_, _, ()>(
            rocblas_sgemv_wrapper,
            (
                //&handle as *const rocblas_handle,
                //&trans as *const rocblas_operation,
                &m as *const i32,
                &n as *const i32,
                &alpha as *const f32,
                A.as_ptr(),
                &lda as *const i32,
                x.as_ptr(),
                &incx as *const i32,
                &beta as *const f32,
                y.as_ptr(),
                &incy as *const i32,
            ),
        );
    };
}

unsafe extern "C" {
    fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;
    pub fn rocblas_sgemv(
        handle: rocblas_handle,
        trans: rocblas_operation,
        m: rocblas_int,
        n: rocblas_int,
        alpha: f32,
        A: *const f32,
        lda: rocblas_int,
        x: *const f32,
        incx: rocblas_int,
        beta: f32,
        y: *mut f32,
        incy: rocblas_int,
    );
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn rocblas_sgemv_wrapper(
        //handle: *const rocblas_handle,
        //trans: *const rocblas_operation,
        m: *const rocblas_int,
        n: *const rocblas_int,
        alpha: *const f32,
        A: *const f32,
        lda: *const rocblas_int,
        x: *const f32,
        incx: *const rocblas_int,
        beta: *const f32,
        y: *mut f32,
        incy: *const rocblas_int,
    ) -> ();
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn rocblas_sgemv_wrapper(
    //handle: *const rocblas_handle,
    //trans: *const rocblas_operation,
    m: *const rocblas_int,
    n: *const rocblas_int,
    alpha: *const f32,
    A: *const f32,
    lda: *const rocblas_int,
    x: *const f32,
    incx: *const rocblas_int,
    beta: *const f32,
    y: *mut f32,
    incy: *const rocblas_int,
) -> () {
    unsafe {
        let trans = rocblas_operation::rocblas_operation_none;
        let mut handle: rocblas_handle = core::ptr::null_mut();
        rocblas_sgemv(
            handle, trans, *m, *n, *alpha, A, *lda, x, *incx, *beta, y, *incy,
        );
    };
}
