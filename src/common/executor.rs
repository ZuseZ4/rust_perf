#[cfg(target_os = "linux")]
extern crate libc;

#[cfg(target_os = "linux")]
use crate::common::data_utils::reset_data_init_count;
#[cfg(target_os = "linux")]
use crate::common::kernel_base::KernelBase;
#[cfg(target_os = "linux")]
use crate::common::timer::now_ns;

#[cfg(target_os = "linux")]
pub struct KernelResult {
    pub name: &'static str,
    pub problem_size: usize,
    pub reps: u32,
    pub total_s: f64,
    pub checksum: f64,
}

#[cfg(target_os = "linux")]
pub const MAX_KERNELS: usize = 64;

#[cfg(target_os = "linux")]
pub struct Executor<'a> {
    kernels: &'a mut [&'a mut dyn KernelBase],
}

#[cfg(target_os = "linux")]
impl<'a> Executor<'a> {
    pub fn new(kernels: &'a mut [&'a mut dyn KernelBase]) -> Self {
        Self { kernels }
    }

    pub fn run_suite<'r>(
        &mut self,
        out: &'r mut [core::mem::MaybeUninit<KernelResult>; MAX_KERNELS],
    ) -> &'r [KernelResult] {
        let n = self.kernels.len().min(MAX_KERNELS);

        for (i, kernel) in self.kernels.iter_mut().enumerate().take(n) {
            let reps = kernel.default_reps();

            reset_data_init_count();
            kernel.setup();

            let t0 = now_ns();
            for _ in 0..reps {
                kernel.run_kernel();
            }
            let t1 = now_ns();

            let checksum = kernel.update_checksum();
            kernel.tear_down();

            out[i].write(KernelResult {
                name: kernel.name(),
                problem_size: kernel.default_problem_size(),
                reps,
                total_s: (t1 - t0) as f64 * 1e-9,
                checksum,
            });
        }

        unsafe { core::slice::from_raw_parts(out.as_ptr() as *const KernelResult, n) }
    }

    pub fn print_report(results: &[KernelResult]) {
        unsafe {
            libc::printf(c"\n".as_ptr());
            libc::printf(
                c"%-20s %6s %12s %16s %18s\n".as_ptr(),
                c"KernelBase".as_ptr(),
                c"Reps".as_ptr(),
                c"N".as_ptr(),
                c"Total time(s)".as_ptr(),
                c"Checksum".as_ptr(),
            );
            libc::printf(
                c"%-20s %6s %12s %16s %18s\n".as_ptr(),
                c"--------------------".as_ptr(),
                c"------".as_ptr(),
                c"------------".as_ptr(),
                c"----------------".as_ptr(),
                c"------------------".as_ptr(),
            );
            for r in results {
                libc::printf(
                    c"%-20s %6u %12zu %16.6f %18.6f\n".as_ptr(),
                    r.name.as_bytes().as_ptr(),
                    r.reps as libc::c_uint,
                    r.problem_size as libc::size_t,
                    r.total_s,
                    r.checksum,
                );
            }
            libc::printf(c"\n".as_ptr());
        }
    }

    pub fn export_csv(results: &[KernelResult], filename: *const libc::c_char) {
        unsafe {
            let fp = libc::fopen(filename, c"w".as_ptr());
            if fp.is_null() {
                return;
            }
            libc::fprintf(fp, c"Kernel,Reps,N,TotalTime,Checksum\n".as_ptr());
            for r in results {
                libc::fprintf(
                    fp,
                    c"%s,%u,%zu,%.6f,%.6f\n".as_ptr(),
                    r.name.as_bytes().as_ptr(),
                    r.reps as libc::c_uint,
                    r.problem_size as libc::size_t,
                    r.total_s,
                    r.checksum,
                );
            }
            libc::fclose(fp);
        }
    }
}
