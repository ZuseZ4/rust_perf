#[cfg(target_os = "linux")]
pub trait KernelBase {
    fn name(&self) -> &'static str;
    fn default_problem_size(&self) -> usize;
    fn default_reps(&self) -> u32;
    fn setup(&mut self);
    fn run_kernel(&mut self);
    fn update_checksum(&self) -> f64;
    fn tear_down(&mut self);
}

#[macro_export]
macro_rules! kernel_name {
    ($s:literal) => {
        unsafe { core::str::from_utf8_unchecked(concat!($s, "\0").as_bytes()) }
    };
}
