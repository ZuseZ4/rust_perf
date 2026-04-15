const DEFAULT_PROBLEM_SIZE: usize = 1_000_000;
const DEFAULT_REPS: u32 = 160;

const IEND: usize = DEFAULT_PROBLEM_SIZE;
pub const COEFFLEN: usize = 16;
const THREADS_PER_BLOCK: u32 = 256;
const BLOCKS: u32 = (IEND as u32).div_ceil(THREADS_PER_BLOCK);

#[cfg(target_arch = "nvptx64")]
use core::arch::nvptx::{
    _block_dim_x as block_dim_x, _block_idx_x as block_idx_x, _thread_idx_x as thread_idx_x,
};
#[cfg(target_arch = "amdgpu")]
#[allow(improper_ctypes)]
unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn thread_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn block_idx_x() -> i32;
    #[link_name = "llvm.amdgcn.workgroup.size.x"]
    fn block_dim_x() -> i32;
}

#[cfg(target_os = "linux")]
use crate::common::data_utils::{
    alloc_and_init_data_const, alloc_and_init_data_rand_value, calc_checksum, free,
};
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::kernel_name;

#[cfg(target_os = "linux")]
use crate::common::types::{Real, to_real};

#[cfg(target_os = "linux")]
pub struct Fir {
    m_in: *mut Real,
    m_out: *mut Real,
    coeff: [Real; COEFFLEN],
}

#[cfg(target_os = "linux")]
impl Fir {
    pub const INIT: Self = Fir {
        m_in: core::ptr::null_mut(),
        m_out: core::ptr::null_mut(),
        coeff: [const { to_real(0.0) }; COEFFLEN],
    };
}

#[cfg(target_os = "linux")]
impl KernelBase for Fir {
    fn name(&self) -> &'static str {
        kernel_name!("FIR")
    }
    fn default_problem_size(&self) -> usize {
        DEFAULT_PROBLEM_SIZE
    }
    fn default_reps(&self) -> u32 {
        DEFAULT_REPS
    }

    fn setup(&mut self) {
        self.coeff = [
            to_real(3.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(3.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(3.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(-1.0),
            to_real(3.0),
        ];

        unsafe {
            self.m_in = alloc_and_init_data_rand_value(IEND + COEFFLEN - 1);
            self.m_out = alloc_and_init_data_const(IEND, to_real(0.0));
        }
    }

    fn run_kernel(&mut self) {
        unsafe {
            core::intrinsics::offload::<_, _, ()>(
                _fir,
                [BLOCKS, 1, 1],
                [THREADS_PER_BLOCK, 1, 1],
                (
                    self.m_out as *mut [Real; IEND],
                    &*(self.m_in as *const [Real; IEND + COEFFLEN]),
                    &self.coeff as &[Real; COEFFLEN],
                    IEND,
                ),
            );
        }
    }

    fn update_checksum(&self) -> f64 {
        unsafe { calc_checksum(self.m_out as *const Real, IEND) }
    }

    fn tear_down(&mut self) {
        unsafe {
            free(self.m_in);
            self.m_in = core::ptr::null_mut();
            free(self.m_out);
            self.m_out = core::ptr::null_mut();
        }
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    pub fn _fir(
        m_out: *mut [Real; IEND],
        m_in: &[Real; IEND + COEFFLEN],
        coeff: &[Real; COEFFLEN],
        iend: usize,
    );
}

#[cfg(not(target_os = "linux"))]
use crate::common::types::Real;

#[cfg(not(target_os = "linux"))]
#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn _fir(
    m_out: *mut [Real; IEND],
    m_in: &[Real; IEND + COEFFLEN],
    coeff: &[Real; COEFFLEN],
    iend: usize,
) {
    let i = unsafe { (block_idx_x() * block_dim_x() + thread_idx_x()) as usize };
    if i < iend {
        let mut sum: Real = Real::from(0.0);
        let mut j = 0;
        while j < COEFFLEN {
            unsafe {
                sum += (*coeff)[j] * (*m_in)[i + j];
            }
            j += 1;
        }
        unsafe {
            (*m_out)[i] = sum;
        }
    }
}
