#![feature(abi_gpu_kernel)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;
#[cfg(target_os = "linux")]
use libc::c_char;

#[cfg(target_os = "linux")]
use core::mem;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
fn main() {
    let output = c"The first element is zero %f\n";
    let output2 = c"The first element is NOT zero %f\n";
    let output3 = c"The second element is %f\n";

    let array_c: *mut [f64; 256] =
        unsafe { libc::calloc(256, (mem::size_of::<f64>()) as libc::size_t) as *mut [f64; 256] };

    unsafe {
        let val: *const c_char = if (*array_c)[0] < 0.1 {
            output.as_ptr()
        } else {
            output2.as_ptr()
        };
        libc::printf(val, (*array_c)[0]);
    }

    unsafe {
        kernel_1(array_c);
    }
    core::hint::black_box(&array_c);
    unsafe {
        let val: *const c_char = if (*array_c)[0] < 0.1 {
            output.as_ptr()
        } else {
            output2.as_ptr()
        };
        libc::printf(val, (*array_c)[0]);
        libc::printf(output3.as_ptr(), (*array_c)[1]);
    }
}

#[inline(never)]
unsafe fn kernel_1(x: *mut [f64; 256]) {
    core::intrinsics::offload(_kernel_1, (x,))
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _kernel_1(array_b: *mut [f64; 256]);
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub extern "gpu-kernel" fn _kernel_1(x: *mut [f64; 256]) {
    unsafe { (*x)[0] = 21.0 };
}