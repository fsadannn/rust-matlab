/// # Safety
///
/// This function is `unsafe` because it uses raw pointers and relies on the caller
/// to ensure that:
/// - `source` and `dest` are valid, non-null pointers.
/// - The memory regions they point to are valid for at least `size` `f64` elements.
/// - The memory regions do not overlap in a way that would cause data races.
/// - The target CPU supports the necessary x86_64 SSE2 SIMD instructions.
///   This typically means compiling with `target_feature="+sse2"` or running on a modern x86_64 CPU.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn scale_unrolled_simd(
    source: *const f64,
    dest: *mut f64,
    size: usize,
    scaling_factor: f64,
) {
    // Import x86_64 SIMD intrinsics
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m128d, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m128d, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};

    const UNROLL_FACTOR: usize = 2; // Process 2 f64 elements per SIMD operation

    let remainder = size & 1;
    let unrolled_limit = size - remainder;

    // Broadcast the scaling factor into a 128-bit SIMD register.
    let factor_vec: __m128d = _mm_set1_pd(scaling_factor);

    // Main unrolled loop: processes UNROLL_FACTOR elements at a time using SIMD.
    for i in (0..unrolled_limit).step_by(UNROLL_FACTOR) {
        unsafe {
            // Load 2 f64 values from the source into a 128-bit register.
            let src_vec: __m128d = _mm_loadu_pd(source.add(i));

            // Multiply the loaded source vector by the scaling factor vector.
            let result_vec: __m128d = _mm_mul_pd(src_vec, factor_vec);

            // Store the 2 resulting f64 values into the destination.
            _mm_storeu_pd(dest.add(i), result_vec);
        }
    }

    // Cleanup loop: processes any remaining elements (0 to UNROLL_FACTOR - 1 elements).
    // This handles cases where 'size' is not a perfect multiple of UNROLL_FACTOR.
    if size & 1 == 0 {
        unsafe {
            *dest.add(unrolled_limit) = *source.add(unrolled_limit) * scaling_factor;
        }
    }
}
