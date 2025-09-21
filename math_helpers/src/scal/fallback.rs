/// # Safety
/// This function is `unsafe` because it operates on raw pointers and does not provide any memory safety guarantees.
/// The caller must ensure that:
/// - `source` and `dest` are valid, non-null pointers.
/// - The memory regions pointed to by `source` and `dest` are valid for at least `size` `f64` elements.
/// - The `source` and `dest` memory regions do not overlap unless `source` is also `dest` (which would be an in-place operation).
pub unsafe fn scale_unrolled_fallback(
    source: *const f64,
    dest: *mut f64,
    size: usize,
    scaling_factor: f64,
) {
    // Define the unrolling factor. We'll process 4 elements per iteration.
    const UNROLL_FACTOR: usize = 4;

    // Calculate the limit for the unrolled loop.
    // This ensures we only process full chunks of UNROLL_FACTOR.
    let unrolled_limit = size - (size % UNROLL_FACTOR);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        // Process element i
        unsafe { *dest.add(i) = *source.add(i) * scaling_factor };

        // Process element i + 1
        unsafe { *dest.add(i + 1) = *source.add(i + 1) * scaling_factor };

        // Process element i + 2
        unsafe { *dest.add(i + 2) = *source.add(i + 2) * scaling_factor };

        // Process element i + 3
        unsafe { *dest.add(i + 3) = *source.add(i + 3) * scaling_factor };
    }

    // Cleanup loop: processes any remaining elements (0 to UNROLL_FACTOR - 1 elements).
    // This handles cases where 'size' is not a perfect multiple of UNROLL_FACTOR.
    for i in unrolled_limit..size {
        unsafe {
            *dest.add(i) = *source.add(i) * scaling_factor;
        }
    }
}
