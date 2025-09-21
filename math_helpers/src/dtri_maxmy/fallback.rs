/// # Safety
/// This function is `unsafe` because it operates on raw pointers and uses scalar operations.
/// The caller must ensure that:
/// - `source_x` and `dest_y` are valid, non-null pointers.
/// - The memory regions pointed to by `source_x` and `dest_y` are valid for at least `n`
///   elements.
/// - The `source_x` and `dest_y` memory regions do not overlap unless `source_x` is also `dest_y`
///   (which would be an in-place operation on Y, not a standard DAXPY).
#[inline]
pub unsafe fn dtri_maxmy_fallback(alpha: f64, source_x: *const f64, dest_y: *mut f64, n: usize) {
    // Fallback to scalar operations if not on x86_64
    unsafe {
        *dest_y.add(0) = alpha * (*source_x.add(0)) + (*dest_y.add(0));
    }

    for j in 1..n {
        let gap = j * n;
        for i in 0..(j + 1) {
            unsafe {
                *dest_y.add(gap + i) = alpha * (*source_x.add(gap + i)) + (*dest_y.add(gap + i));
            }
        }
    }
}
