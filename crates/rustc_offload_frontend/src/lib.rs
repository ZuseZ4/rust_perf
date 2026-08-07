#![allow(internal_features)]
#![allow(linker_messages)]
#![allow(improper_ctypes)]
#![allow(improper_gpu_kernel_arg)]
#![allow(improper_ctypes_definitions)]
#![feature(gpu_offload, core_intrinsics, gpu_intrinsics, offload)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![cfg_attr(any(target_arch = "nvptx64", target_arch = "amdgpu"), no_std)]

pub use core::offload::offload_kernel;

pub mod gpu;
pub mod partition;

#[macro_export]
macro_rules! offload {
    ( $($field:ident = $val:expr),* $(,)? ) => {
        $crate::offload!(@munch
            [ $($field = $val),* ];
            kernel = NONE;
            grid_dim = ([1, 1, 1]);
            block_dim = ([1, 1, 1]);
            dyn_cache = (0);
            args = NONE
        );
    };

    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = (SOME $val); grid_dim = $g; block_dim = $b; dyn_cache = $d; args = $a);
    };
    (@munch [grid_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = ($val); block_dim = $b; dyn_cache = $d; args = $a);
    };
    (@munch [block_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = ($val); dyn_cache = $d; args = $a);
    };
    (@munch [dyn_cache = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = $b; dyn_cache = ($val); args = $a);
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = $b; dyn_cache = $d; args = (SOME $val));
    };

    (@munch [$invalid:ident = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!(concat!("unknown field ", stringify!($invalid)));
    };

    (@munch []; kernel = NONE; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!("missing `kernel`");
    };
    (@munch []; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; dyn_cache = $d:tt; args = NONE) => {
        compile_error!("missing `args`");
    };
    (@munch []; kernel = (SOME $kernel:expr); grid_dim = ($grid_dim:expr); block_dim = ($block_dim:expr); dyn_cache = ($dyn_cache:expr); args = (SOME $args:expr)) => {
        core::intrinsics::offload::<_, _, ()>(
            $kernel,
            $grid_dim,
            $block_dim,
            $dyn_cache,
            $args,
        )
    };
}

#[cfg(target_arch = "nvptx64")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
