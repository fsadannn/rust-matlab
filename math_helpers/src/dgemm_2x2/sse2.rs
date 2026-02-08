#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
pub unsafe fn dgemm_2x2_sse2(a: *const f64, b: *const f64, out: *mut f64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_add_pd, _mm_load1_pd, _mm_loadu_pd, _mm_mul_pd, _mm_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_add_pd, _mm_load1_pd, _mm_loadu_pd, _mm_mul_pd, _mm_storeu_pd};
    let a_col0 = unsafe { _mm_loadu_pd(a) };
    let a_col1 = unsafe { _mm_loadu_pd(a.add(2)) };

    // Load elements of B and broadcast them (duplicate to both halves of register)
    // _mm_load1_pd loads one double and splats it: | x | -> | x |
    //                                                       | x |
    let b0 = unsafe { _mm_load1_pd(b) }; // splat b0
    let b1 = unsafe { _mm_load1_pd(b.add(1)) }; // splat b1
    let b2 = unsafe { _mm_load1_pd(b.add(2)) }; // splat b2
    let b3 = unsafe { _mm_load1_pd(b.add(3)) }; // splat b3

    // Calculate First Column of C
    // c_col0 = (a_col0 * b0) + (a_col1 * b1)
    let c_col0 = _mm_add_pd(_mm_mul_pd(a_col0, b0), _mm_mul_pd(a_col1, b1));

    // Calculate Second Column of C
    // c_col1 = (a_col0 * b2) + (a_col1 * b3)
    let c_col1 = _mm_add_pd(_mm_mul_pd(a_col0, b2), _mm_mul_pd(a_col1, b3));

    // Store results
    unsafe { _mm_storeu_pd(out, c_col0) }; // Store c0, c1
    unsafe { _mm_storeu_pd(out.add(2), c_col1) }; // Store c2, c3
}
