mod daxpy;
pub mod detec_features;
mod dtri_maxmy;
mod dxpy;
mod scal;

pub use daxpy::{daxpy, daxpy_avx, daxpy_fallback, daxpy_simd};
pub use dtri_maxmy::dtri_maxmy_simd as dtri_maxmy;
pub use dxpy::{dxpy, dxpy_avx, dxpy_fallback, dxpy_simd};
pub use scal::scale_unrolled;

pub fn frexp(x: f64) -> (f64, i32) {
    let mut y = x.to_bits();
    let ee = ((y >> 52) & 0x7ff) as i32;

    if ee == 0 {
        if x != 0.0 {
            let x1p64 = f64::from_bits(0x43f0000000000000);
            let (x, e) = frexp(x * x1p64);
            return (x, e - 64);
        }
        return (x, 0);
    } else if ee == 0x7ff {
        return (x, 0);
    }

    let e = ee - 0x3fe;
    y &= 0x800fffffffffffff;
    y |= 0x3fe0000000000000;
    (f64::from_bits(y), e)
}
