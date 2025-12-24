#![allow(internal_features)]
#![feature(abi_gpu_kernel)]
#![feature(link_llvm_intrinsics)]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![no_std]

#[cfg(target_os = "linux")]
extern crate libc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[allow(improper_ctypes)]
unsafe extern "C" {
    #[cfg(target_arch = "nvptx64")]
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
    fn block_dim_x() -> i32;
    #[cfg(target_arch = "nvptx64")]
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    fn thread_idx_x() -> i32;
    #[cfg(target_arch = "nvptx64")]
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    fn block_idx_x() -> i32;

    #[cfg(target_arch = "amdgpu")]
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn thread_idx_x() -> i32;
    #[cfg(target_arch = "amdgpu")]
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn block_idx_x() -> i32;
    #[cfg(target_arch = "amdgpu")]
    #[link_name = "llvm.amdgcn.workgroup.size.x"]
    fn block_dim_x() -> i32;
}

#[cfg(target_os = "linux")]
#[unsafe(no_mangle)]
#[inline(never)]
unsafe fn main() {
    let mut div: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x2: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x3: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    let y1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let y2: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let y3: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let y4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    let fx1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let fx2: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let fx3: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let fx4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    let fy1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let fy2: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let fy3: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let fy4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    let real_zones: [usize; 4] = [0, 1, 2, 3];
    let half: f32 = 1.0;
    let ptiny: f32 = 0.001;
    let iend: usize = 4;

    unsafe {
        kernel(
            div.as_mut_ptr(),
            x1.as_ptr(),
            x2.as_ptr(),
            x3.as_ptr(),
            x4.as_ptr(),
            y1.as_ptr(),
            y2.as_ptr(),
            y3.as_ptr(),
            y4.as_ptr(),
            fx1.as_ptr(),
            fx2.as_ptr(),
            fx3.as_ptr(),
            fx4.as_ptr(),
            fy1.as_ptr(),
            fy2.as_ptr(),
            fy3.as_ptr(),
            fy4.as_ptr(),
            real_zones.as_ptr(),
            half,
            &ptiny,
            iend,
        )
    };
}

#[inline(never)]
unsafe fn kernel(
    div: *mut f32,
    x1: *const f32,
    x2: *const f32,
    x3: *const f32,
    x4: *const f32,
    y1: *const f32,
    y2: *const f32,
    y3: *const f32,
    y4: *const f32,
    fx1: *const f32,
    fx2: *const f32,
    fx3: *const f32,
    fx4: *const f32,
    fy1: *const f32,
    fy2: *const f32,
    fy3: *const f32,
    fy4: *const f32,
    real_zones: *const usize,
    half: f32,
    ptiny: *const f32,
    iend: usize,
) {
    core::intrinsics::offload(
        kernel_1,
        (
            div, x1, x2, x3, x4, y1, y2, y3, y4, fx1, fx2, fx3, fx4, fy1, fy2, fy3, fy4,
            real_zones, half, ptiny, iend,
        ),
    )
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn kernel_1(
        div: *mut f32,
        x1: *const f32,
        x2: *const f32,
        x3: *const f32,
        x4: *const f32,
        y1: *const f32,
        y2: *const f32,
        y3: *const f32,
        y4: *const f32,
        fx1: *const f32,
        fx2: *const f32,
        fx3: *const f32,
        fx4: *const f32,
        fy1: *const f32,
        fy2: *const f32,
        fy3: *const f32,
        fy4: *const f32,
        real_zones: *const usize,
        half: f32,
        ptiny: *const f32,
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[inline(never)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn kernel_1(
    div: *mut f32,
    x1: *const f32,
    x2: *const f32,
    x3: *const f32,
    x4: *const f32,
    y1: *const f32,
    y2: *const f32,
    y3: *const f32,
    y4: *const f32,
    fx1: *const f32,
    fx2: *const f32,
    fx3: *const f32,
    fx4: *const f32,
    fy1: *const f32,
    fy2: *const f32,
    fy3: *const f32,
    fy4: *const f32,
    real_zones: *const usize,
    half: f32,
    ptiny: *const f32,
    iend: usize,
) {
    unsafe {
        let ii = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;

        if ii < iend {
            let i = *real_zones.add(ii);

            let xi = half * (*x1.add(i) + *x2.add(i) - *x3.add(i) - *x4.add(i));
            let xj = half * (*x2.add(i) + *x3.add(i) - *x4.add(i) - *x1.add(i));
            let yi = half * (*y1.add(i) + *y2.add(i) - *y3.add(i) - *y4.add(i));
            let yj = half * (*y2.add(i) + *y3.add(i) - *y4.add(i) - *y1.add(i));
            let fxi = half * (*fx1.add(i) + *fx2.add(i) - *fx3.add(i) - *fx4.add(i));
            let fxj = half * (*fx2.add(i) + *fx3.add(i) - *fx4.add(i) - *fx1.add(i));
            let fyi = half * (*fy1.add(i) + *fy2.add(i) - *fy3.add(i) - *fy4.add(i));
            let fyj = half * (*fy2.add(i) + *fy3.add(i) - *fy4.add(i) - *fy1.add(i));

            let rarea = 1.0 / (xi * yj - xj * yi + *ptiny);
            let dfxdx = rarea * (fxi * yj - fxj * yi);
            let dfydy = rarea * (fyj * xi - fyi * xj);
            let affine = (*fy1.add(i) + *fy2.add(i) + *fy3.add(i) + *fy4.add(i))
                / (*y1.add(i) + *y2.add(i) + *y3.add(i) + *y4.add(i));
            *div.add(i) = dfxdx + dfydy + affine;
        }
    }
}
