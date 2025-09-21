/// Performs the DXP-Y operation: Y = X + Y, using SSE2 SIMD intrinsics.
///
/// # Safety
/// This function is `unsafe` because it uses SIMD intrinsics and operates on raw pointers.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
pub unsafe fn dxpy_simd(source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m128d, _mm_add_pd, _mm_loadu_pd, _mm_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m128d, _mm_add_pd, _mm_loadu_pd, _mm_storeu_pd};

    const UNROLL_FACTOR: usize = 2; // Process 2 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        let x_vec: __m128d = unsafe { _mm_loadu_pd(source_x.add(i)) };
        let y_vec: __m128d = unsafe { _mm_loadu_pd(dest_y.add(i)) };
        let result_vec: __m128d = _mm_add_pd(x_vec, y_vec);
        unsafe { _mm_storeu_pd(dest_y.add(i), result_vec) };
    }

    if remainder != 0 {
        unsafe {
            *dest_y.add(unrolled_limit) =
                (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dxpy_simd_exact_multiple() {
        if is_x86_feature_detected!("sse2") {
            let size = 4;
            let source_x = [1.0, 2.0, 3.0, 4.0];
            let mut dest_y = vec![5.0, 6.0, 7.0, 8.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { dxpy_simd(x_ptr, y_ptr, size) };

            let expected = vec![6.0, 8.0, 10.0, 12.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dxpy_simd_remainder() {
        if is_x86_feature_detected!("sse2") {
            let size = 5;
            let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
            let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { dxpy_simd(x_ptr, y_ptr, size) };

            let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0];
            assert_eq!(dest_y, expected);
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_dxpy_simd_zero_size() {
        if is_x86_feature_detected!("sse2") {
            let size = 0;
            let source_x: Vec<f64> = vec![];
            let mut dest_y: Vec<f64> = vec![];

            let x_ptr = source_x.as_ptr();
            let y_ptr = dest_y.as_mut_ptr();

            unsafe { dxpy_simd(x_ptr, y_ptr, size) };

            let expected: Vec<f64> = vec![];
            assert_eq!(dest_y, expected);
        }
    }
}
