mod avx;
mod fallback;
mod simd;

pub use self::avx::daxpy_avx;
pub use self::fallback::daxpy_fallback;
pub use self::simd::daxpy_simd;

pub type FnDaxpy = unsafe fn(f64, *const f64, *mut f64, usize) -> ();

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
pub unsafe fn daxpy(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && size >= 4 {
            return unsafe { daxpy_avx(alpha, source_x, dest_y, size) };
        }
        if is_x86_feature_detected!("sse2") && size >= 2 {
            return unsafe { daxpy_simd(alpha, source_x, dest_y, size) };
        }
    }

    unsafe { daxpy_fallback(alpha, source_x, dest_y, size) }
}

// --- Example Usage and Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    // --- Tests for daxpy_simd function ---
    #[test]
    fn test_daxpy_simd_exact_multiple() {
        let size = 4;
        let alpha = 2.0;
        let source_x = [1.0, 2.0, 3.0, 4.0];
        let mut dest_y = vec![5.0, 6.0, 7.0, 8.0]; // Initial Y

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy(alpha, x_ptr, y_ptr, size) };

        // Expected: Y = 2.0 * X + Y
        // [2*1+5, 2*2+6, 2*3+7, 2*4+8] = [7.0, 10.0, 13.0, 16.0]
        let expected = vec![7.0, 10.0, 13.0, 16.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_daxpy_simd_remainder() {
        let size = 5;
        let alpha = -1.0;
        let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy(alpha, x_ptr, y_ptr, size) };

        // Expected: Y = -1.0 * X + Y
        // [-1*1+10, -1*2+20, -1*3+30, -1*4+40, -1*5+50] = [9.0, 18.0, 27.0, 36.0, 45.0]
        let expected = vec![9.0, 18.0, 27.0, 36.0, 45.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_daxpy_simd_zero_size() {
        let size = 0;
        let alpha = 5.0;
        let source_x: Vec<f64> = vec![];
        let mut dest_y: Vec<f64> = vec![];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy(alpha, x_ptr, y_ptr, size) };

        let expected: Vec<f64> = vec![];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_daxpy_simd_alpha_zero() {
        let size = 3;
        let alpha = 0.0;
        let source_x = [1.0, 2.0, 3.0];
        let mut dest_y = vec![10.0, 20.0, 30.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy(alpha, x_ptr, y_ptr, size) };

        // Expected: Y = 0.0 * X + Y = Y
        let expected = vec![10.0, 20.0, 30.0];
        assert_eq!(dest_y, expected);
    }
}
