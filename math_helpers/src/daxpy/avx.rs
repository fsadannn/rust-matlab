/// Performs the DAXPY operation: Y = alpha * X + Y, using x86_64 AVX SIMD intrinsics.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses SIMD intrinsics.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `size`
///   `f64` elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
/// - The target CPU supports the necessary x86_64 AVX SIMD instructions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub fn daxpy_avx(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    const UNROLL_FACTOR: usize = 4; // Process 4 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Broadcast the alpha factor into a 256-bit SIMD register.
    let alpha_vec: __m256d = _mm256_set1_pd(alpha);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        // Load 4 f64 values from vector X
        let x_vec: __m256d = unsafe { _mm256_loadu_pd(source_x.add(i)) };
        // Load 4 f64 values from vector Y
        let y_vec: __m256d = unsafe { _mm256_loadu_pd(dest_y.add(i)) };

        // Perform alpha * X
        let alpha_x_vec: __m256d = _mm256_mul_pd(alpha_vec, x_vec);

        // Perform (alpha * X) + Y
        let result_vec: __m256d = _mm256_add_pd(alpha_x_vec, y_vec);

        // Store the result back into Y
        unsafe { _mm256_storeu_pd(dest_y.add(i), result_vec) };
    }

    match remainder {
        1 => unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
        },
        2 => unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
            *dest_y.add(unrolled_limit + 1) =
                alpha * (*source_x.add(unrolled_limit + 1)) + (*dest_y.add(unrolled_limit + 1));
        },
        3 => unsafe {
            *dest_y.add(unrolled_limit) =
                alpha * (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
            *dest_y.add(unrolled_limit + 1) =
                alpha * (*source_x.add(unrolled_limit + 1)) + (*dest_y.add(unrolled_limit + 1));
            *dest_y.add(unrolled_limit + 2) =
                alpha * (*source_x.add(unrolled_limit + 2)) + (*dest_y.add(unrolled_limit + 2));
        },
        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_avx_exact_multiple() {
        if is_x86_feature_detected!("avx") {
            let size = 8; // Multiple of 4
            let alpha = 2.0;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_avx(alpha, x_ptr, y_ptr, size) };

            let expected = vec![12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_avx_remainder() {
        if is_x86_feature_detected!("avx") {
            let size = 7; // Not a multiple of 4
            let alpha = -1.0;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_avx(alpha, x_ptr, y_ptr, size) };

            let expected = vec![9.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_avx_zero_size() {
        if is_x86_feature_detected!("avx") {
            let size = 0;
            let alpha = 5.0;
            let source_x: Vec<f64> = vec![];
            let mut dest_y: Vec<f64> = vec![];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_avx(alpha, x_ptr, y_ptr, size) };

            let expected: Vec<f64> = vec![];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_daxpy_avx_alpha_zero() {
        if is_x86_feature_detected!("avx") {
            let size = 5;
            let alpha = 0.0;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { daxpy_avx(alpha, x_ptr, y_ptr, size) };

            let expected = vec![10.0, 20.0, 30.0, 40.0, 50.0];
            assert_eq!(dest_y, expected);
        }
    }
}
