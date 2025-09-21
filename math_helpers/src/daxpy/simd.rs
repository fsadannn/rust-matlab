/// This function is `unsafe` because it operates on raw pointers and uses SIMD intrinsics.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `size`
///   `f64` elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
/// - The target CPU supports the necessary x86_64 SSE2 SIMD instructions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
pub fn daxpy_simd(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128d, _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128d, _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd,
    };

    const UNROLL_FACTOR: usize = 2; // Process 2 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Broadcast the alpha factor into a 128-bit SIMD register.
    let alpha_vec: __m128d = _mm_set1_pd(alpha);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        // Load 2 f64 values from vector X
        let x_vec: __m128d = unsafe { _mm_loadu_pd(source_x.add(i)) };
        // Load 2 f64 values from vector Y
        let y_vec: __m128d = unsafe { _mm_loadu_pd(dest_y.add(i)) };

        // Perform alpha * X
        let alpha_x_vec: __m128d = _mm_mul_pd(alpha_vec, x_vec);

        // Perform (alpha * X) + Y
        let result_vec: __m128d = _mm_add_pd(alpha_x_vec, y_vec);

        // Store the result back into Y
        unsafe { _mm_storeu_pd(dest_y.add(i), result_vec) };
    }

    if remainder != 0 {
        unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_simd_exact_multiple() {
        if is_x86_feature_detected!("sse2") {
            let size = 4; // Multiple of 2
            let alpha = 2.0;
            let source_x = [1.0, 2.0, 3.0, 4.0];
            let mut dest_y = vec![5.0, 6.0, 7.0, 8.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_simd(alpha, x_ptr, y_ptr, size) };

            let expected = vec![7.0, 10.0, 13.0, 16.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_simd_remainder() {
        if is_x86_feature_detected!("sse2") {
            let size = 5; // Not a multiple of 2
            let alpha = -1.0;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_simd(alpha, x_ptr, y_ptr, size) };

            let expected = vec![9.0, 18.0, 27.0, 36.0, 45.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_simd_zero_size() {
        if is_x86_feature_detected!("sse2") {
            let size = 0;
            let alpha = 5.0;
            let source_x: Vec<f64> = vec![];
            let mut dest_y: Vec<f64> = vec![];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_simd(alpha, x_ptr, y_ptr, size) };

            let expected: Vec<f64> = vec![];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_simd_alpha_zero() {
        if is_x86_feature_detected!("sse2") {
            let size = 3;
            let alpha = 0.0;
            let source_x = [1.0, 2.0, 3.0];
            let mut dest_y = vec![10.0, 20.0, 30.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_simd(alpha, x_ptr, y_ptr, size) };

            let expected = vec![10.0, 20.0, 30.0];
            assert_eq!(dest_y, expected);
        }
    }
}
