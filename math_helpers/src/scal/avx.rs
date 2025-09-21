/// Scales elements from a source array to a destination array by a given factor.
/// This function uses the AVX instruction set and is only available on x86 and x86_64
/// architectures.
///
/// # Safety
/// This function is `unsafe` because it uses raw pointers and relies on the caller
/// to ensure that:
/// - `source` and `dest` are valid, non-null pointers.
/// - The memory regions pointed to by `source` and `dest` are valid for at least `size`
///   `f64` elements.
/// - The `source` and `dest` memory regions do not overlap if `source` is also `dest`.
///   (For scaling in-place, `source` and `dest` would be the same pointer).
/// - The target CPU supports the necessary x86_64 SIMD instructions (e.g., AVX for _mm256_ operations).
///   This typically means compiling with `target_feature="+avx"` or running on a modern x86_64 CPU.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
pub unsafe fn scale_unrolled_avx(
    source: *const f64,
    dest: *mut f64,
    size: usize,
    scaling_factor: f64,
) {
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
        // Load 4 f64 values from the source into a 256-bit register.
        // _mm256_loadu_pd is for unaligned loads.
        let src_vec: __m256d = unsafe { _mm256_loadu_pd(source.add(i)) };

        // Multiply the loaded source vector by the scaling factor vector.
        let result_vec: __m256d = _mm256_mul_pd(src_vec, factor_vec);

        // Store the 4 resulting f64 values into the destination.
        // _mm256_storeu_pd is for unaligned stores.
        unsafe { _mm256_storeu_pd(dest.add(i), result_vec) };
    }

    match remainder {
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
