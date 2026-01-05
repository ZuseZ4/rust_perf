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

// Usually from rocblas-types.h; typically an opaque pointer type.
#[repr(C)]
pub struct rocblas_handle__ {
    _private: [u8; 0],
}
pub type rocblas_handle = *mut rocblas_handle__;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum rocblas_status {
    rocblas_status_success = 0,              //**< Success */
    rocblas_status_invalid_handle = 1,       //**< Handle not initialized, invalid or null */
    rocblas_status_not_implemented = 2,      //**< Function is not implemented */
    rocblas_status_invalid_pointer = 3,      //**< Invalid pointer argument */
    rocblas_status_invalid_size = 4,         //**< Invalid size argument */
    rocblas_status_memory_error = 5, //**< Failed internal memory allocation, copy or dealloc */
    rocblas_status_internal_error = 6, //**< Other internal library failure */
    rocblas_status_perf_degraded = 7, //**< Performance degraded due to low device memory */
    rocblas_status_size_query_mismatch = 8, //**< Unmatched start/stop size query */
    rocblas_status_size_increased = 9, //**< Queried device memory size increased */
    rocblas_status_size_unchanged = 10, //**< Queried device memory size unchanged */
    rocblas_status_invalid_value = 11, //**< Passed argument not valid */
    rocblas_status_continue = 12,    //**< Nothing preventing function to proceed */
    rocblas_status_check_numerics_fail = 13, //**< Will be set if the vector/matrix has a NaN/Infinity/denormal value */
    rocblas_status_excluded_from_build = 14, //**< Function is not available in build, likely a function requiring Tensile built without Tensile */
    rocblas_status_arch_mismatch = 15, //**< The function requires a feature absent from the device architecture */
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
unsafe fn main() {
    let A: [f32; 6] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let x: [f32; 3] = [1.0, 1.0, 1.0];
    let mut y: [f32; 2] = [0.0, 0.0];
    unsafe {
        core::intrinsics::offload_args::<_, _, ()>(
            rocblas_sgemv_wrapper,
            [256, 1, 1],
            [32, 1, 1],
            (A.as_ptr(), x.as_ptr(), y.as_mut_ptr()),
        );
    };
    use libc::{c_double, printf};
    unsafe { printf(c"y1 %f\n".as_ptr(), y[0] as c_double) };
    unsafe { printf(c"y0 %f\n".as_ptr(), y[1] as c_double) };
}

#[repr(C)]
pub struct hipStream_t__ {
    _private: [u8; 0],
}
pub type hipStream_t = *mut hipStream_t__;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum hipError_t {
    hipSuccess = 0, /* ... */
}
unsafe extern "C" {
    fn hipSetDevice(dev: i32) -> hipError_t;
    fn rocblas_create_handle(handle: *mut rocblas_handle) -> rocblas_status;
    fn rocblas_destroy_handle(handle: rocblas_handle) -> rocblas_status;
    fn rocblas_get_stream(handle: rocblas_handle, stream: *mut hipStream_t) -> rocblas_status;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn rocblas_sgemv(
        handle: rocblas_handle,
        trans: rocblas_operation,
        m: i32,
        n: i32,
        alpha: *const f32,
        A: *const [f32; 6],
        lda: i32,
        x: *const [f32; 3],
        incx: i32,
        beta: *const f32,
        y: *mut [f32; 2],
        incy: i32,
    ) -> rocblas_status;
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
pub fn rocblas_sgemv_wrapper(A: *const [f32; 6], x: *const [f32; 3], y: *mut [f32; 2]) -> () {
    let m: i32 = 2;
    let n: i32 = 3;
    let incx: i32 = 1;
    let incy: i32 = 1;
    let lda = m;
    // those two by default should be host ptr:
    let alpha: f32 = 1.0;
    let beta: f32 = 1.0;
    unsafe {
        assert_eq!(hipSetDevice(0), hipError_t::hipSuccess);
        let trans = rocblas_operation::rocblas_operation_none;
        let mut handle: rocblas_handle = core::ptr::null_mut();
        let st = rocblas_create_handle(&mut handle as *mut rocblas_handle);
        assert_eq!(st, rocblas_status::rocblas_status_success);
        let st_res = rocblas_sgemv(
            handle,
            trans,
            m,
            n,
            &alpha as *const f32,
            A,
            lda,
            x,
            incx,
            &beta as *const f32,
            y,
            incy,
        );
        assert_eq!(st_res, rocblas_status::rocblas_status_success);

        let mut stream: hipStream_t = core::ptr::null_mut();
        let st_stream = rocblas_get_stream(handle, &mut stream as *mut _);
        assert_eq!(st_stream, rocblas_status::rocblas_status_success);

        let he = hipStreamSynchronize(stream);
        assert_eq!(he, hipError_t::hipSuccess);
        let st2 = rocblas_destroy_handle(handle);
        assert_eq!(st2, rocblas_status::rocblas_status_success);
    };
}
