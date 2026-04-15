#![feature(asm_experimental_arch)]
#![allow(internal_features)]
#![feature(abi_gpu_kernel)]
#![feature(link_llvm_intrinsics)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics, intrinsics, gpu_intrinsics)]
#![cfg_attr(
    target_os = "amdhsa",
    feature(stdarch_amdgpu, gpu_launch_sized_workgroup_mem)
)]
#![no_std]
#![no_main]

#[cfg(not(target_os = "linux"))]
use core::arch::amdgpu::*;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
}

#[repr(C)]
#[cfg(target_os = "linux")]
#[derive(Copy, Clone)]
struct Timespec {
    tv_sec: libc::time_t,
    tv_nsec: libc::c_long,
}

#[cfg(target_os = "linux")]
const CLOCK_MONOTONIC: i32 = 1;

#[cfg(target_os = "linux")]
unsafe fn get_time_ns() -> u64 {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn kernel_1(dbg1: *mut [usize; 40], dbg2: *const [usize; 40]);
}

//*********************************************************************
//function name: gpu_square_matrix_mult
//
//description: dot product of two matrix (not only square) in GPU
//
//parameters:
//      &a GPU device pointer to a n X n matrix (A)
//      &b GPU device pointer to a n X n matrix (B)
//      &c GPU device output purpose pointer to a n X n matrix (C)
//      to store the result
//Note:
//  grid and block should be configured as:
//
//    dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
//    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
//
//return: none
//*********************************************************************
//*/
const BLOCK_SIZE: u32 = 16;

#[cfg(target_arch = "amdgpu")]
use core::arch::amdgpu::{
    s_barrier, workgroup_id_x as block_idx_x, workgroup_id_y as block_idx_y,
    workitem_id_x as thread_idx_x, workitem_id_y as thread_idx_y,
};

#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.s.barrier"]
    fn _syncthreads();
}

const DIM: usize = 256 * 256;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn gpu_square_matrix_mult(
        d_a: *const [i32; DIM],
        d_b: *const [i32; DIM],
        d_result: *mut [i32; DIM],
        n: *const u32,
        reps: *const u32,
    );
}
#[unsafe(no_mangle)]
#[cfg(not(target_os = "linux"))]
#[inline(never)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn gpu_square_matrix_mult(
    d_a: *const i32,
    d_b: *const i32,
    d_result: *mut i32,
    n: *const u32,
    reps: *const u32,
) {
    unsafe {
        let n = *n;
        let r = *reps;
        let tile_a = core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<i32>()
            as *mut [i32; (BLOCK_SIZE * BLOCK_SIZE) as usize];
        let tile_b = core::intrinsics::gpu::gpu_launch_sized_workgroup_mem::<i32>()
            .add((BLOCK_SIZE * BLOCK_SIZE) as usize)
            as *mut [i32; (BLOCK_SIZE * BLOCK_SIZE) as usize];
        for i in 0..r {
            let blockIdxX: u32 = block_idx_x();
            let blockIdxY: u32 = block_idx_y();
            let threadIdxX: u32 = thread_idx_x();
            let threadIdxY: u32 = thread_idx_y();
            let gridDimX: u32 = BLOCK_SIZE; //block_dim_x();

            let row = blockIdxY * BLOCK_SIZE + threadIdxY;
            let col = blockIdxX * BLOCK_SIZE + threadIdxX;
            let mut tmp = 0;
            let mut idx;
            for sub in 0..gridDimX {
                idx = row * n + sub * BLOCK_SIZE + threadIdxX;
                let left_idx = (threadIdxY * BLOCK_SIZE + threadIdxX) as usize;
                if (idx >= n * n) {
                    // n may not divisible by BLOCK_SIZE
                    (*tile_a)[left_idx] = 0;
                } else {
                    (*tile_a)[left_idx] = *d_a.add(idx as usize);
                }

                idx = (sub * BLOCK_SIZE + threadIdxY) * n + col;
                let left_idx = (threadIdxY * BLOCK_SIZE + threadIdxX) as usize;
                if (idx >= n * n) {
                    (*tile_b)[left_idx] = 0;
                } else {
                    (*tile_b)[left_idx] = *d_b.add(idx as usize);
                }
                s_barrier();

                for k in 0..BLOCK_SIZE {
                    let left_idx = (threadIdxY * BLOCK_SIZE + k) as usize;
                    let left_idx2 = (k * BLOCK_SIZE + threadIdxX) as usize;
                    tmp += (*tile_a)[left_idx] * (*tile_b)[left_idx2];
                }
                s_barrier();
            }

            if (row < n && col < n) {
                *(d_result.add((row * n + col) as usize)) = tmp;
            }
        }
    }
}
fn cpu_matrix_mult(
    h_a: *const i32,
    h_b: *const i32,
    h_result: *mut i32,
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..k {
            let mut tmp = 0;
            for h in 0..n {
                tmp += unsafe { *h_a.add(i * n + h) * *h_b.add(h * k + j) };
            }
            unsafe { *h_result.add(i * k + j) = tmp };
        }
    }
}
#[cfg(target_os = "linux")]
unsafe fn alloc_array<T>(len: i32) -> *mut T {
    let len = len as usize;
    let size = len * core::mem::size_of::<T>();
    let ptr = unsafe { libc::malloc(size) } as *mut T;
    if ptr.is_null() {
        panic!();
    }
    ptr
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn main(argc: libc::c_int, argv: *const *const libc::c_char) -> libc::c_int {
    let m = libc::atoi(*argv.add(1));
    let n = libc::atoi(*argv.add(2));
    let k = libc::atoi(*argv.add(3));
    let reps = libc::atoi(*argv.add(4)) as u32;
    let check = libc::atoi(*argv.add(5));
    let check: bool = if check == 1 { true } else { false };

    if m != n || n != k {
        panic!();
    }

    /* Fixed seed for illustration */
    libc::srand(3333);

    // allocate memory in host RAM, h_cc is used to store CPU result
    let h_a: *mut [i32; 4 * DIM] = alloc_array(4 * m * n) as *mut [i32; 4 * DIM];
    let h_b: *mut [i32; 4 * DIM] = alloc_array(4 * n * k) as *mut [i32; 4 * DIM];
    let mut h_c: *mut [i32; 4 * DIM] = alloc_array(4 * m * k) as *mut [i32; 4 * DIM];
    let mut h_cc: *mut i32 = alloc_array(4 * m * k);

    // random initialize matrix A
    for i in 0..m {
        for j in 0..n {
            (*h_a)[(i * n + j) as usize] = libc::rand() % 1024;
        }
    }

    // random initialize matrix B
    for i in 0..m {
        for j in 0..n {
            (*h_b)[(i * k + j) as usize] = libc::rand() % 1024;
        }
    }

    let mut gpu_elapsed_time_total_s = 0.0;
    let mut gpu_elapsed_time_s: usize;
    let mut cpu_elapsed_time_s: usize;

    let grid_rows = (m + BLOCK_SIZE as i32 - 1) / BLOCK_SIZE as i32;
    let grid_cols = (k + BLOCK_SIZE as i32 - 1) / BLOCK_SIZE as i32;

    let nthreads = BLOCK_SIZE;

    //for r in 0..=reps {
    // start to count execution time of GPU version
    //start = get_time();
    let start = get_time_ns();

    let n = n as u32;
    core::intrinsics::offload::<_, _, ()>(
        gpu_square_matrix_mult,
        [grid_cols as u32, grid_rows as u32, 1],
        [nthreads, nthreads, 1],
        2 * 1024 as u32,
        (
            h_a as *const [i32; DIM],
            h_b as *const [i32; DIM],
            h_c as *mut [i32; DIM],
            &n as *const u32,
            &reps as *const u32,
        ),
    );
    //gpu_square_matrix_mult(h_a, h_b, h_c, n);

    // time counting terminate
    //stop = get_time();
    //gpu_elapsed_time_s = stop - start;
    //libc::printf(
    //    "[%d] matmul GPU %dx%d x %dx%d: %f s\n",
    //    r,
    //    m,
    //    n,
    //    n,
    //    k,
    //    gpu_elapsed_time_s,
    //);

    //    if r > 0 {
    //        //gpu_elapsed_time_total_s += gpu_elapsed_time_s;
    //    }
    //}

    //libc::printf("[average] matmul GPU %dx%d x %dx%d on %d runs: %f s\n", m, n, n, k, reps, gpu_elapsed_time_total_s/reps);
    let end = get_time_ns();
    let duration_s = (end - start) as f64 / 1_000_000_000.0;
    libc::printf(c"gpu: %f\n".as_ptr(), duration_s);

    if check {
        // start the CPU version
        //start = get_time();
        let start = get_time_ns();

        for i in 0..reps {
            cpu_matrix_mult(
                h_a as *const i32,
                h_b as *const i32,
                h_cc,
                m as usize,
                n as usize,
                k as usize,
            );
        }
        let end = get_time_ns();
        let duration_s = (end - start) as f64 / 1_000_000_000.0;
        libc::printf(c"cpu: %f\n".as_ptr(), duration_s);

        //stop = get_time();
        //cpu_elapsed_time_s = stop - start;
        //libc::printf("matmul CPU %dx%d x %dx%d: %f s\n", m, n, n, k, cpu_elapsed_time_s);

        // validate results computed by GPU
        let mut all_ok = true;
        for i in 0..m {
            for j in 0..k {
                let val_a = *h_cc.add((i * k + j) as usize);
                let val_b = (*h_c)[(i * k + j) as usize];
                if val_a != val_b {
                    libc::printf(c"[%d] [%d] wrong!\n".as_ptr(), val_a, val_b);
                    all_ok = false;
                }
            }
        }

        if !all_ok {
            libc::printf(c"incorrect results\n".as_ptr());
            return 1;
        } else {
            libc::printf(c"correct results\n".as_ptr());
            return 1;
        }
    }

    // free memory
    free_array(h_a);
    free_array(h_b);
    free_array(h_c);
    free_array(h_cc);

    return 0;
}
#[cfg(target_os = "linux")]
unsafe fn free_array<T>(ptr: *mut T) {
    unsafe { libc::free(ptr as *mut libc::c_void) };
}

