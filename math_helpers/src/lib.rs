mod daxpy;
mod scal;

pub use daxpy::daxpy_simd as daxpy;
pub use scal::scale_unrolled;
