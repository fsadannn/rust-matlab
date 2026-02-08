#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
/// Computes the product of two 2x2 matrices using AVX2 and FMA instructions.
///
/// Matrices are assumed to be in column-major order.
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to at least 4 `f64` elements (2x2 matrix).
/// - `out` must be writable.
/// - The CPU must support AVX2 and FMA instructions.
pub unsafe fn dgemm_2x2_avx2(a: *const f64, b: *const f64, out: *mut f64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute4x64_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute4x64_pd, _mm256_storeu_pd,
    };
    let va = unsafe { _mm256_loadu_pd(a) };
    let vb = unsafe { _mm256_loadu_pd(b) };

    // Shuffle logic matches previous AVX implementation
    let vb_shuf1 = _mm256_permute4x64_pd(vb, 0xA0); // [b0, b0, b2, b2]
    let va_shuf1 = _mm256_permute4x64_pd(va, 0x44); // [a0, a1, a0, a1]
    let term1 = _mm256_mul_pd(vb_shuf1, va_shuf1);

    let vb_shuf2 = _mm256_permute4x64_pd(vb, 0xF5); // [b1, b1, b3, b3]
    let va_shuf2 = _mm256_permute4x64_pd(va, 0xEE); // [a2, a3, a2, a3]

    let result = _mm256_fmadd_pd(vb_shuf2, va_shuf2, term1);

    unsafe { _mm256_storeu_pd(out, result) };
}
