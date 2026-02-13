mod daxpy;
mod dgemm_2x2;
mod dtri_maxmy;
mod dxpy;
pub mod matrix_exp_22;
mod scal;

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_feature = "sse2")))]
#[deny(warnings)]
compile_error!("This module only supports x86 and x86_64 architectures with sse2");

#[cfg(target_arch = "x86")]
use std::arch::x86::__m128d;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__m128d;

pub use daxpy::{FnDaxpy, daxpy, daxpy_avx, daxpy_fallback, daxpy_simd};
pub use dgemm_2x2::{FnDGEM22, dgemm_2x2, dgemm_2x2_avx2, dgemm_2x2_fallback, dgemm_2x2_sse2};
pub use dtri_maxmy::{
    FnDtriMaxmy, dtri_maxmy, dtri_maxmy_avx, dtri_maxmy_fallback, dtri_maxmy_simd,
};
pub use dxpy::{FnDxpy, dxpy, dxpy_avx, dxpy_fallback, dxpy_simd};
pub use scal::{
    FnScale, scale_unrolled, scale_unrolled_avx, scale_unrolled_fallback, scale_unrolled_simd,
};

#[inline]
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

#[repr(C)]
pub union M128dAsF64s {
    pub simd: __m128d,
    pub scalars: [f64; 2],
}
