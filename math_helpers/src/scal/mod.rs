mod avx;
mod fallback;
mod simd;

pub use self::avx::scale_unrolled_avx;
pub use self::fallback::scale_unrolled_fallback;
pub use self::simd::scale_unrolled_simd;

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
        if is_x86_feature_detected!("avx") && size >= 4 {
            return unsafe { scale_unrolled_avx(source, dest, size, scaling_factor) };
        }
        if is_x86_feature_detected!("sse2") && size >= 2 {
            return unsafe { scale_unrolled_simd(source, dest, size, scaling_factor) };
        }
    }

    unsafe { scale_unrolled_fallback(source, dest, size, scaling_factor) }
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
