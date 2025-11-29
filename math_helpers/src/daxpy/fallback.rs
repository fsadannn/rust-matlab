/// Performs the DAXPY operation: Y = alpha * X + Y, using scalar operations.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses scalar operations.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `size`
///   `f64` elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
#[inline]
pub unsafe fn daxpy_fallback(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    // Fallback to scalar operations if not on x86_64
    for i in 0..size {
        unsafe {
            *dest_y.add(i) += alpha * (*source_x.add(i));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daxpy_fallback_basic() {
        let size = 5;
        let alpha = 2.0;
        let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy_fallback(alpha, x_ptr, y_ptr, size) };

        // Expected: Y = 2.0 * X + Y
        // [2*1+10, 2*2+20, 2*3+30, 2*4+40, 2*5+50] = [12.0, 24.0, 36.0, 48.0, 60.0]
        let expected = vec![12.0, 24.0, 36.0, 48.0, 60.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_daxpy_fallback_zero_size() {
        let size = 0;
        let alpha = 5.0;
        let source_x: Vec<f64> = vec![];
        let mut dest_y: Vec<f64> = vec![];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy_fallback(alpha, x_ptr, y_ptr, size) };

        let expected: Vec<f64> = vec![];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_daxpy_fallback_alpha_zero() {
        let size = 3;
        let alpha = 0.0;
        let source_x = [1.0, 2.0, 3.0];
        let mut dest_y = vec![10.0, 20.0, 30.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy_fallback(alpha, x_ptr, y_ptr, size) };

        // Expected: Y = 0.0 * X + Y = Y
        let expected = vec![10.0, 20.0, 30.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_daxpy_fallback_negative_alpha() {
        let size = 4;
        let alpha = -1.0;
        let source_x = [1.0, 2.0, 3.0, 4.0];
        let mut dest_y = vec![5.0, 6.0, 7.0, 8.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { daxpy_fallback(alpha, x_ptr, y_ptr, size) };

        // Expected: Y = -1.0 * X + Y
        // [-1*1+5, -1*2+6, -1*3+7, -1*4+8] = [4.0, 4.0, 4.0, 4.0]
        let expected = vec![4.0, 4.0, 4.0, 4.0];
        assert_eq!(dest_y, expected);
    }
}
