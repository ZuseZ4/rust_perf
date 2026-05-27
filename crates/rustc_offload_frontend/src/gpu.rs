#[derive(Clone, Copy)]
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

pub(crate) fn global_thread_dim() -> Dim3 {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        use core::arch::nvptx::*;
        Dim3 {
            x: (_block_idx_x() * _block_dim_x() + _thread_idx_x()) as usize,
            y: (_block_idx_y() * _block_dim_y() + _thread_idx_y()) as usize,
            z: (_block_idx_z() * _block_dim_z() + _thread_idx_z()) as usize,
        }
    }
    #[cfg(target_os = "linux")]
    Dim3 { x: 0, y: 0, z: 0 }
}
