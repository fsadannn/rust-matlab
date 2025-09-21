/// Performs the DXP-Y operation: Y = X + Y, using scalar operations.
///
/// # Safety
/// This function is `unsafe` because it operates on raw pointers.
#[inline]
pub unsafe fn dxpy_fallback(source_x: *const f64, dest_y: *mut f64, size: usize) {
    for i in 0..size {
        unsafe {
            *dest_y.add(i) = (*source_x.add(i)) + (*dest_y.add(i));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dxpy_fallback_basic() {
        let size = 5;
        let source_x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dest_y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy_fallback(x_ptr, y_ptr, size) };

        let expected = vec![11.0, 22.0, 33.0, 44.0, 55.0];
        assert_eq!(dest_y, expected);
    }

    #[test]
    fn test_dxpy_fallback_zero_size() {
        let size = 0;
        let source_x: Vec<f64> = vec![];
        let mut dest_y: Vec<f64> = vec![];

        let x_ptr = source_x.as_ptr();
        let y_ptr = dest_y.as_mut_ptr();

        unsafe { dxpy_fallback(x_ptr, y_ptr, size) };

        let expected: Vec<f64> = vec![];
        assert_eq!(dest_y, expected);
    }
}
