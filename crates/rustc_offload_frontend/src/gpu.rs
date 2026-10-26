#[derive(Clone, Copy)]
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[cfg(target_arch = "amdgpu")]
use core::intrinsics::gpu::amdgpu_dispatch_ptr;

/// Get the packet for this dispatch.
///
/// Get a reference to the packet that was used to dispatch this kernel.
/// The dispatch packet contains information like the workgroup size and dispatch size.
///
/// # Example
///
/// ```rust
/// # #![no_std]
/// # extern crate alloc;
/// # fn main() {
/// use amdgpu_device_libs::prelude::*;
///
/// let dispatch = dispatch_ptr();
/// println!("Workgroup size {}x{}x{}", dispatch.workgroup_size_x, dispatch.workgroup_size_y, dispatch.workgroup_size_z);
/// # }
/// ```
#[cfg(target_arch = "amdgpu")]
#[inline]
pub fn dispatch_ptr() -> &'static HsaKernelDispatchPacket {
    unsafe {
        &*core::mem::transmute::<*const (), *const HsaKernelDispatchPacket>(
            core::intrinsics::gpu::amdgpu_dispatch_ptr(),
        )
    }
}
/// HSA packet to dispatch a kernel.
///
/// A pointer to the packet that was used to dispatch the currently running kernel can be obtained with [`dispatch_ptr`].
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(C)]
pub struct HsaKernelDispatchPacket {
    /// Packet header. Used to configure multiple packet parameters such as the
    /// packet type. The parameters are described by hsa_packet_header_t.
    pub header: u16,
    /// Dispatch setup parameters. Used to configure kernel dispatch parameters
    /// such as the number of dimensions in the grid. The parameters are described
    /// by hsa_kernel_dispatch_packet_setup_t.
    pub setup: u16,
    /// X dimension of work-group, in work-items. Must be greater than 0.
    pub workgroup_size_x: u16,
    /// Y dimension of work-group, in work-items. Must be greater than
    /// 0. If the grid has 1 dimension, the only valid value is 1.
    pub workgroup_size_y: u16,
    /// Z dimension of work-group, in work-items. Must be greater than
    /// 0. If the grid has 1 or 2 dimensions, the only valid value is 1.
    pub workgroup_size_z: u16,
    /// Reserved. Must be 0.
    pub reserved0: u16,
    /// X dimension of grid, in work-items. Must be greater than 0. Must
    /// not be smaller than @a workgroup_size_x.
    pub grid_size_x: u32,
    /// Y dimension of grid, in work-items. Must be greater than 0. If the grid has
    /// 1 dimension, the only valid value is 1. Must not be smaller than @a
    /// workgroup_size_y.
    pub grid_size_y: u32,
    /// Z dimension of grid, in work-items. Must be greater than 0. If the grid has
    /// 1 or 2 dimensions, the only valid value is 1. Must not be smaller than @a
    /// workgroup_size_z.
    pub grid_size_z: u32,
    /// Size in bytes of private memory allocation request (per work-item).
    pub private_segment_size: u32,
    /// Size in bytes of group memory allocation request (per work-group). Must not
    /// be less than the sum of the group memory used by the kernel (and the
    /// functions it calls directly or indirectly) and the dynamically allocated
    /// group segment variables.
    pub group_segment_size: u32,
    /// Opaque handle to a code object that includes an implementation-defined
    /// executable code for the kernel.
    pub kernel_object: u64,
    /// Pointer to the kernel arguments.
    pub kernarg_address: *mut core::ffi::c_void,
    /// Reserved. Must be 0.
    pub reserved2: u64,
    /// Signal used to indicate completion of the job. The application can use the
    /// special signal handle 0 to indicate that no signal is used.
    pub completion_signal: u64,
}
// Handle to an HSA signal.
//#[cfg(feature = "device_libs")]
//#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
//#[repr(C)]
//pub struct HsaSignal {
//    /// The internal representation of an HSA signal.
//    pub handle: u64,
//}

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
    #[cfg(target_arch = "amdgpu")]
    unsafe {
        use core::arch::amdgpu::*;
        let dispatch = dispatch_ptr();

        let x = (workgroup_id_x() * (*dispatch).workgroup_size_x as u32 + workitem_id_x()) as usize;
        let y = (workgroup_id_y() * (*dispatch).workgroup_size_y as u32 + workitem_id_y()) as usize;
        let z = (workgroup_id_z() * (*dispatch).workgroup_size_z as u32 + workitem_id_z()) as usize;
        Dim3 { x, y, z }
    }
    #[cfg(target_os = "linux")]
    Dim3 { x: 0, y: 0, z: 0 }
}

pub(crate) fn block_idx() -> Dim3 {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        use core::arch::nvptx::*;
        Dim3 {
            x: _block_idx_x() as usize,
            y: _block_idx_y() as usize,
            z: _block_idx_z() as usize,
        }
    }
    #[cfg(target_arch = "amdgpu")]
    unsafe {
        use core::arch::amdgpu::*;
        let dispatch = dispatch_ptr();

        let x = (workgroup_id_x()) as usize;
        let y = (workgroup_id_y()) as usize;
        let z = (workgroup_id_z()) as usize;
        Dim3 { x, y, z }
    }
    #[cfg(target_os = "linux")]
    Dim3 { x: 0, y: 0, z: 0 }
}

pub(crate) fn block_dim() -> Dim3 {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        use core::arch::nvptx::*;
        Dim3 {
            x: _block_dim_x() as usize,
            y: _block_dim_y() as usize,
            z: _block_dim_z() as usize,
        }
    }
    #[cfg(target_arch = "amdgpu")]
    unsafe {
        use core::arch::amdgpu::*;
        let dispatch = dispatch_ptr();

        let x = dispatch.workgroup_size_x as usize;
        let y = dispatch.workgroup_size_y as usize;
        let z = dispatch.workgroup_size_z as usize;
        Dim3 { x, y, z }
    }
    #[cfg(target_os = "linux")]
    Dim3 { x: 0, y: 0, z: 0 }
}

pub(crate) fn thread_idx() -> Dim3 {
    #[cfg(target_arch = "nvptx64")]
    unsafe {
        use core::arch::nvptx::*;
        Dim3 {
            x: _thread_idx_x() as usize,
            y: _thread_idx_y() as usize,
            z: _thread_idx_z() as usize,
        }
    }
    #[cfg(target_arch = "amdgpu")]
    unsafe {
        use core::arch::amdgpu::*;

        let x = (workitem_id_x()) as usize;
        let y = (workitem_id_y()) as usize;
        let z = (workitem_id_z()) as usize;
        Dim3 { x, y, z }
    }
    #[cfg(target_os = "linux")]
    Dim3 { x: 0, y: 0, z: 0 }
}
