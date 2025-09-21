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
pub fn daxpy_fallback(alpha: f64, source_x: *const f64, dest_y: *mut f64, size: usize) {
    // Fallback to scalar operations if not on x86_64
    for i in 0..size {
        unsafe {
            *dest_y.add(i) = alpha * (*source_x.add(i)) + (*dest_y.add(i));
        }
    }
}
