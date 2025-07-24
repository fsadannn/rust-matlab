mod daxpy;
mod dtri_maxmy;
mod scal;

pub use daxpy::daxpy_simd as daxpy;
pub use dtri_maxmy::dtri_maxmy_simd as dtri_maxmy;
pub use scal::scale_unrolled;
