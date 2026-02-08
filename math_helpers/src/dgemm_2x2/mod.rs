pub use crate::dgemm_2x2::{
    avx2::dgemm_2x2_avx2, fallback::dgemm_2x2_fallback, sse2::dgemm_2x2_sse2,
};

mod avx2;
mod fallback;
mod sse2;

/// Type alias for a function that performs 2x2 double-precision matrix multiplication.
pub type FnDGEM22 = unsafe fn(*const f64, *const f64, *mut f64) -> ();

/// Computes the product of two 2x2 matrices (C = A * B).
///
/// This function automatically selects the best available implementation based on CPU features
/// (AVX2/FMA, SSE2, or fallback).
///
/// Matrices are assumed to be in column-major order:
/// ```text
/// [ 0  2 ]
/// [ 1  3 ]
/// ```
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to at least 4 `f64` elements (2x2 matrix).
/// - `out` must be writable.
pub unsafe fn dgemm_2x2(a: *const f64, b: *const f64, out: *mut f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dgemm_2x2_avx2(a, b, out) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { dgemm_2x2_sse2(a, b, out) };
        }
    }

    unsafe { dgemm_2x2_fallback(a, b, out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgemm_2x2_identity() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];

        unsafe {
            dgemm_2x2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }

        assert_eq!(out, b);
    }

    #[test]
    fn test_dgemm_2x2_general() {
        // A = [ 1 3 ]  (col-major: [1, 2, 3, 4])
        //     [ 2 4 ]
        // B = [ 5 7 ]  (col-major: [5, 6, 7, 8])
        //     [ 6 8 ]
        // C = A * B = [ 1*5+3*6  1*7+3*8 ] = [ 23  31 ]
        //             [ 2*5+4*6  2*7+4*8 ]   [ 34  46 ]
        // Col-major C: [23, 34, 31, 46]

        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];

        unsafe {
            dgemm_2x2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }

        let expected = [23.0, 34.0, 31.0, 46.0];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_dgemm_2x2_fallback_only() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0; 4];

        unsafe {
            fallback::dgemm_2x2_fallback(a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        }

        let expected = [23.0, 34.0, 31.0, 46.0];
        assert_eq!(out, expected);
    }
}
