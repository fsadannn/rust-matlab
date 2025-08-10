#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
fn _simd_scale_unrolled(source: *const f64, dest: *mut f64, size: usize, scaling_factor: f64) {
    // Import x86_64 SIMD intrinsics
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m128d, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m128d, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};

    // Define the unrolling factor based on 128-bit SIMD registers for f64.
    // A __m128d register holds two f64 values.
    const UNROLL_FACTOR: usize = 2; // Process 2 f64 elements per SIMD operation

    // Calculate the limit for the unrolled loop.
    // This ensures we only process full chunks of UNROLL_FACTOR.
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Broadcast the scaling factor into a 128-bit SIMD register.
    // This creates a vector where both f64 lanes contain `scaling_factor`.
    let factor_vec: __m128d = _mm_set1_pd(scaling_factor);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        unsafe {
            // Load 2 f64 values from the source into a 128-bit register.
            // _mm_loadu_pd is for unaligned loads.
            let src_vec: __m128d = _mm_loadu_pd(source.add(i));

            // Multiply the loaded source vector by the scaling factor vector.
            let result_vec: __m128d = _mm_mul_pd(src_vec, factor_vec);

            // Store the 2 resulting f64 values into the destination.
            // _mm_storeu_pd is for unaligned stores.
            _mm_storeu_pd(dest.add(i), result_vec);
        }
    }

    // Cleanup loop: processes any remaining elements (0 to UNROLL_FACTOR - 1 elements).
    // This handles cases where 'size' is not a perfect multiple of UNROLL_FACTOR.
    if remainder != 0 {
        unsafe {
            *dest.add(unrolled_limit) = *source.add(unrolled_limit) * scaling_factor;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
fn _simd256_scale_unrolled(source: *const f64, dest: *mut f64, size: usize, scaling_factor: f64) {
    // Import x86_64 SIMD intrinsics
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256d, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256d, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    // Define the unrolling factor based on 256-bit SIMD registers for f64.
    // A __m256d register holds four f64 values.
    const UNROLL_FACTOR: usize = 4; // Process 4 f64 elements per SIMD operation

    // Calculate the limit for the unrolled loop.
    // This ensures we only process full chunks of UNROLL_FACTOR.
    let remainder = size % UNROLL_FACTOR;
    let unrolled_limit = size - remainder;

    // Broadcast the scaling factor into a 256-bit SIMD register.
    // This creates a vector where all four f64 lanes contain `scaling_factor`.
    let factor_vec: __m256d = _mm256_set1_pd(scaling_factor);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        unsafe {
            // Load 4 f64 values from the source into a 256-bit register.
            // _mm256_loadu_pd is for unaligned loads.
            let src_vec: __m256d = _mm256_loadu_pd(source.add(i));

            // Multiply the loaded source vector by the scaling factor vector.
            let result_vec: __m256d = _mm256_mul_pd(src_vec, factor_vec);

            // Store the 4 resulting f64 values into the destination.
            // _mm256_storeu_pd is for unaligned stores.
            _mm256_storeu_pd(dest.add(i), result_vec);
        }
    }

    match remainder {
        0 => (),
        1 => unsafe {
            *dest.add(unrolled_limit) = *source.add(unrolled_limit) * scaling_factor;
        },
        2 => unsafe {
            *dest.add(unrolled_limit) = *source.add(unrolled_limit) * scaling_factor;
            *dest.add(unrolled_limit + 1) = *source.add(unrolled_limit + 1) * scaling_factor;
        },
        3 => unsafe {
            *dest.add(unrolled_limit) = *source.add(unrolled_limit) * scaling_factor;
            *dest.add(unrolled_limit + 1) = *source.add(unrolled_limit + 1) * scaling_factor;
            *dest.add(unrolled_limit + 2) = *source.add(unrolled_limit + 2) * scaling_factor;
        },
        _ => (),
    }
}

fn _scale_unrolled(source: *const f64, dest: *mut f64, size: usize, scaling_factor: f64) {
    // Define the unrolling factor. We'll process 4 elements per iteration.
    const UNROLL_FACTOR: usize = 4;

    // Calculate the limit for the unrolled loop.
    // This ensures we only process full chunks of UNROLL_FACTOR.
    let unrolled_limit = size - (size % UNROLL_FACTOR);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        unsafe {
            // Process element i
            *dest.add(i) = *source.add(i) * scaling_factor;

            // Process element i + 1
            *dest.add(i + 1) = *source.add(i + 1) * scaling_factor;

            // Process element i + 2
            *dest.add(i + 2) = *source.add(i + 2) * scaling_factor;

            // Process element i + 3
            *dest.add(i + 3) = *source.add(i + 3) * scaling_factor;
        }
    }

    // Cleanup loop: processes any remaining elements (0 to UNROLL_FACTOR - 1 elements).
    // This handles cases where 'size' is not a perfect multiple of UNROLL_FACTOR.
    for i in unrolled_limit..size {
        unsafe {
            *dest.add(i) = *source.add(i) * scaling_factor;
        }
    }
}

/// Scales elements from a source array to a destination array by a given factor.
/// This function uses loop unrolling and x86_64 SIMD intrinsics (128-bit operations)
/// for potential performance improvement.
///
/// # Arguments
/// * `source` - A raw pointer to the source array of f64 values.
/// * `dest` - A mutable raw pointer to the destination array where scaled values will be stored.
/// * `size` - The total number of elements to scale.
/// * `scaling_factor` - The f64 value by which each element will be multiplied.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses SIMD intrinsics.
/// The caller must ensure that:
/// - `source` and `dest` are valid, non-null pointers.
/// - The memory regions pointed to by `source` and `dest` are valid for at least `size`
///   `f64` elements.
/// - The `source` and `dest` memory regions do not overlap if `source` is also `dest`.
///   (For scaling in-place, `source` and `dest` would be the same pointer).
/// - The target CPU supports the necessary x86_64 SIMD instructions (e.g., SSE2 for _mm_pd operations).
///   This typically means compiling with `target_feature="+sse2"` or running on a modern x86_64 CPU.
pub unsafe fn scale_unrolled(source: *const f64, dest: *mut f64, size: usize, scaling_factor: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use crate::detec_features::{HAS_AVX, HAS_SSE2};

        if HAS_AVX.load(std::sync::atomic::Ordering::Relaxed) && size >= 4 {
            return unsafe { _simd256_scale_unrolled(source, dest, size, scaling_factor) };
        }
        if HAS_SSE2.load(std::sync::atomic::Ordering::Relaxed) && size >= 2 {
            return unsafe { _simd_scale_unrolled(source, dest, size, scaling_factor) };
        }
    }

    _scale_unrolled(source, dest, size, scaling_factor);
}

// --- Example Usage and Tests ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_unrolled_exact_multiple() {
        let size = 8; // A multiple of 4
        let source_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dest_data = vec![0.0; size];
        let scaling_factor = 2.5;

        // Get raw pointers
        let source_ptr = source_data.as_ptr();
        let dest_ptr = dest_data.as_mut_ptr();

        unsafe { scale_unrolled(source_ptr, dest_ptr, size, scaling_factor) };

        // Expected results: Each element multiplied by 2.5
        let expected = vec![2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0];

        assert_eq!(dest_data, expected);
    }

    #[test]
    fn test_scale_unrolled_remainder() {
        let size = 7; // Not a multiple of 4 (remainder 3)
        let source_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mut dest_data = vec![0.0; size];
        let scaling_factor = 3.0;

        let source_ptr = source_data.as_ptr();
        let dest_ptr = dest_data.as_mut_ptr();

        unsafe { scale_unrolled(source_ptr, dest_ptr, size, scaling_factor) };

        let expected = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0];
        assert_eq!(dest_data, expected);
    }

    #[test]
    fn test_scale_unrolled_small_size() {
        let size = 2; // Smaller than unroll factor
        let source_data = [10.0, 20.0];
        let mut dest_data = vec![0.0; size];
        let scaling_factor = 0.5;

        let source_ptr = source_data.as_ptr();
        let dest_ptr = dest_data.as_mut_ptr();

        unsafe { scale_unrolled(source_ptr, dest_ptr, size, scaling_factor) };

        let expected = vec![5.0, 10.0];
        assert_eq!(dest_data, expected);
    }

    #[test]
    fn test_scale_unrolled_zero_size() {
        let size = 0;
        let source_data: Vec<f64> = vec![];
        let mut dest_data: Vec<f64> = vec![];
        let scaling_factor = 2.0;

        let source_ptr = source_data.as_ptr();
        let dest_ptr = dest_data.as_mut_ptr();

        unsafe { scale_unrolled(source_ptr, dest_ptr, size, scaling_factor) };

        let expected: Vec<f64> = vec![];
        assert_eq!(dest_data, expected);
    }

    #[test]
    fn test_scale_unrolled_in_place() {
        let size = 6;
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let scaling_factor = 10.0;

        let data_ptr = data.as_mut_ptr();

        // Call with source and dest being the same pointer for in-place scaling
        unsafe { scale_unrolled(data_ptr as *const f64, data_ptr, size, scaling_factor) };

        let expected = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        assert_eq!(data, expected);
    }
}
