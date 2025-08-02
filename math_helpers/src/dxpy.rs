#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
fn _dxpy_simd(source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m128d, _mm_add_pd, _mm_loadu_pd, _mm_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m128d, _mm_add_pd, _mm_loadu_pd, _mm_storeu_pd};

    const UNROLL_FACTOR: usize = 2; // Process 2 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        unsafe {
            // Load 2 f64 values from vector X
            let x_vec: __m128d = _mm_loadu_pd(source_x.add(i));
            // Load 2 f64 values from vector Y
            let y_vec: __m128d = _mm_loadu_pd(dest_y.add(i));

            // Perform (alpha * X) + Y
            let result_vec: __m128d = _mm_add_pd(x_vec, y_vec);

            // Store the result back into Y
            _mm_storeu_pd(dest_y.add(i), result_vec);
        }
    }

    if remainder != 0 {
        unsafe {
            *dest_y.add(unrolled_limit) =
                (*source_x.add(unrolled_limit)) + (*dest_y.add(unrolled_limit));
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
fn _dxpy_simd256(source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_storeu_pd};

    const UNROLL_FACTOR: usize = 4; // Process 4 f64 elements per SIMD operation
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        unsafe {
            // Load 4 f64 values from vector X
            let x_vec: __m256d = _mm256_loadu_pd(source_x.add(i));
            // Load 4 f64 values from vector Y
            let y_vec: __m256d = _mm256_loadu_pd(dest_y.add(i));

            // Perform (alpha * X) + Y
            let result_vec: __m256d = _mm256_add_pd(x_vec, y_vec);

            // Store the result back into Y
            _mm256_storeu_pd(dest_y.add(i), result_vec);
        }
    }

    match remainder {
        0 => (),
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

// Fallback for non-x86_64 architectures for dxpy_simd
pub fn _dxpy_fallback(source_x: *const f64, dest_y: *mut f64, size: usize) {
    // Fallback to scalar operations if not on x86_64
    for i in 0..size {
        unsafe {
            *dest_y.add(i) = (*source_x.add(i)) + (*dest_y.add(i));
        }
    }
}

/// Performs the DAXPY operation: Y = alpha * X + Y, using x86_64 SIMD intrinsics.
///
/// # Arguments
/// * `alpha` - The scalar factor (f64).
/// * `source_x` - A raw pointer to the source vector X (f64 values).
/// * `dest_y` - A mutable raw pointer to the destination vector Y (f64 values),
///              which will be updated in-place.
/// * `size` - The total number of elements in vectors X and Y.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses SIMD intrinsics.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `size`
///   `f64` elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
/// - The target CPU supports the necessary x86_64 SIMD instructions (e.g., SSE2 for _mm_pd operations).
///   This typically means compiling with `target_feature="+sse2"` or running on a modern x86_64 CPU.
pub unsafe fn dxpy_simd(source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && size >= 4 {
            return unsafe { _dxpy_simd256(source_x, dest_y, size) };
        }
        if is_x86_feature_detected!("sse2") && size >= 2 {
            return unsafe { _dxpy_simd(source_x, dest_y, size) };
        }
    }

    _dxpy_fallback(source_x, dest_y, size);
}

// --- Example Usage and Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    // --- Tests for dxpy_simd function ---
    #[test]
    fn test_dxpy_simd_exact_multiple() {
        let size = 4;
        let source_x = [1.0, 2.0, 3.0, 4.0];
        let mut dest_y = vec![5.0, 6.0, 7.0, 8.0]; // Initial Y

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy_simd(x_ptr, y_ptr, size) };

        let expected = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_dxpy_simd_remainder() {
        let size = 5;
        let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy_simd(x_ptr, y_ptr, size) };

        let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_dxpy_simd_zero_size() {
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
