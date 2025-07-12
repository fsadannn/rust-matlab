mod daxpy;
mod identity;
mod scal;

pub use daxpy::daxpy_simd as daxpy;
pub use identity::setIdentity2_unrolled;
pub use scal::scale_unrolled;
