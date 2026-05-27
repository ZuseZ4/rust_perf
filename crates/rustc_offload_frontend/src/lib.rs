#![allow(internal_features)]
#![allow(linker_messages)]
#![allow(improper_ctypes)]
#![allow(improper_gpu_kernel_arg)]
#![allow(improper_ctypes_definitions)]
#![feature(gpu_offload)]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![cfg_attr(target_arch = "nvptx64", no_std)]

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
            args = NONE
        );
    };

    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = (SOME $val); grid_dim = $g; block_dim = $b; args = $a);
    };
    (@munch [grid_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = ($val); block_dim = $b; args = $a);
    };
    (@munch [block_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = ($val); args = $a);
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; grid_dim = $g; block_dim = $b; args = (SOME $val));
    };

    (@munch [$invalid:ident = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        compile_error!(concat!("unkown field ", stringify!($invalid)));
    };

    (@munch []; kernel = NONE; grid_dim = $g:tt; block_dim = $b:tt; args = $a:tt) => {
        compile_error!("missing `kernel`");
    };
    (@munch []; kernel = $k:tt; grid_dim = $g:tt; block_dim = $b:tt; args = NONE) => {
        compile_error!("missing `args`");
    };
    (@munch []; kernel = (SOME $kernel:expr); grid_dim = ($grid_dim:expr); block_dim = ($block_dim:expr); args = (SOME $args:expr)) => {
        core::intrinsics::offload::<_, _, ()>(
            $kernel,
            $grid_dim,
            $block_dim,
            $args,
        )
    };
}

#[cfg(target_arch = "nvptx64")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
