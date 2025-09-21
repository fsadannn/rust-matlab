/// Performs the DXP-Y operation: Y = X + Y, using AVX SIMD intrinsics.
///
/// # Safety
/// This function is `unsafe` because it uses SIMD intrinsics and operates on raw pointers.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub unsafe fn dxpy_avx(source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    const UNROLL_FACTOR: usize = 4; // Process 4 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        let x_vec: __m256d = unsafe { _mm256_loadu_pd(source_x.add(i)) };
        let y_vec: __m256d = unsafe { _mm256_loadu_pd(dest_y.add(i)) };
        let result_vec: __m256d = _mm256_add_pd(x_vec, y_vec);
        unsafe { _mm256_storeu_pd(dest_y.add(i), result_vec) };
    }

    match remainder {
        1 => unsafe {
            *dest_y.add(unrolled_limit) =
                (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
        },
        2 => unsafe {
            *dest_y.add(unrolled_limit) =
                (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
            *dest_y.add(unrolled_limit + 1) =
                (*source_x.add(unrolled_limit + 1)) + (*dest_y.add(unrolled_limit + 1));
        },
        3 => unsafe {
            *dest_y.add(unrolled_limit) =
                (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
            *dest_y.add(unrolled_limit + 1) =
                (*source_x.add(unrolled_limit + 1)) + (*dest_y.add(unrolled_limit + 1));
            *dest_y.add(unrolled_limit + 2) =
                (*source_x.add(unrolled_limit + 2)) + (*dest_y.add(unrolled_limit + 2));
        },
        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dxpy_avx_exact_multiple() {
        if is_x86_feature_detected!("avx") {
            let size = 8;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { dxpy_avx(x_ptr, y_ptr, size) };

            let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dxpy_avx_remainder() {
        if is_x86_feature_detected!("avx") {
            let size = 7;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { dxpy_avx(x_ptr, y_ptr, size) };

            let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dxpy_avx_zero_size() {
        if is_x86_feature_detected!("avx") {
            let size = 0;
            let source_x: Vec<f64> = vec![];
            let mut dest_y: Vec<f64> = vec![];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { dxpy_avx(x_ptr, y_ptr, size) };

            let expected: Vec<f64> = vec![];
            assert_eq!(dest_y, expected);
        }
    }
}
