mod avx;
mod fallback;
mod simd;

pub use self::avx::dxpy_avx;
pub use self::fallback::dxpy_fallback;
pub use self::simd::dxpy_simd;

/// Performs the DXP-Y operation: Y = X + Y, using x86_64 SIMD intrinsics if available.
///
/// This operation is equivalent to DAXPY with alpha = 1.0.
///
/// # Arguments
/// * `source_x` - A raw pointer to the source vector X (f64 values).
/// * `dest_y` - A mutable raw pointer to the destination vector Y (f64 values),
///              which will be updated in-place.
/// * `size` - The total number of elements in vectors X and Y.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers. The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions they point to are valid for at least `size` `f64` elements.
/// - The memory regions do not overlap in a way that would cause data races.
pub unsafe fn dxpy(source_x: *const f64, dest_y: *mut f64, size: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && size >= 4 {
            return unsafe { dxpy_avx(source_x, dest_y, size) };
        }
        if is_x86_feature_detected!("sse2") && size >= 2 {
            return unsafe { dxpy_simd(source_x, dest_y, size) };
        }
    }

    // Fallback for non-x86 or if SIMD is not supported
    unsafe { dxpy_fallback(source_x, dest_y, size) }
}

// --- Example Usage and Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    // --- Tests for dxpy function ---
    #[test]
    fn test_dxpy_exact_multiple() {
        let size = 4;
        let source_x = [1.0, 2.0, 3.0, 4.0];
        let mut dest_y = vec![5.0, 6.0, 7.0, 8.0]; // Initial Y

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy(x_ptr, y_ptr, size) };

        let expected = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_dxpy_remainder() {
        let size = 5;
        let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy(x_ptr, y_ptr, size) };

        let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_dxpy_zero_size() {
        let size = 0;
        let source_x: Vec<f64> = vec![];
        let mut dest_y: Vec<f64> = vec![];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy(x_ptr, y_ptr, size) };

        let expected: Vec<f64> = vec![];
        assert_eq!(dest_y, expected);
    }
}
