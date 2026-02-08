use crate::{FnDGEM22, dgemm_2x2_fallback, frexp};

#[inline(always)]
pub fn solve_2x2(a: &[f64; 4], b: &mut [f64; 4]) {
    const DET_EPSILON: f64 = 2e-15;
    // 1. Unpack A into mutable variables (so we can swap rows)
    // A = | a00 a01 |  stored as [a00, a10, a01, a11]
    //     | a10 a11 |
    let mut a00 = a[0];
    let mut a10 = a[1];
    let mut a01 = a[2];
    let mut a11 = a[3];

    // Unpack B (we perform the same row operations on B)
    let mut b00 = b[0];
    let mut b10 = b[1];
    let mut b01 = b[2];
    let mut b11 = b[3];

    // 2. Partial Pivoting
    // We want the largest absolute value in the first column to be at a00.
    if a10.abs() > a00.abs() {
        // Swap Row 0 and Row 1
        std::mem::swap(&mut a00, &mut a10);
        std::mem::swap(&mut a01, &mut a11);

        // Apply same swap to B
        std::mem::swap(&mut b00, &mut b10);
        std::mem::swap(&mut b01, &mut b11);
    }

    // 3. Singularity Check
    // If the pivot is essentially zero, the matrix is singular.
    if a00.abs() < DET_EPSILON {
        panic!("near singular system");
    }

    // 4. Forward Elimination
    // We want to eliminate a10 (make it 0).
    // Factor f = a10 / a00
    let f = a10 / a00;

    // Row1 = Row1 - f * Row0
    // a10 becomes 0 (we don't need to store it)
    a11 -= f * a01;

    // Apply to B
    b10 -= f * b00;
    b11 -= f * b01;

    // 5. Check second pivot
    if a11.abs() < DET_EPSILON {
        panic!("near singular system");
    }

    // 6. Back Substitution
    // Solve for x1 (bottom row of result) first
    // x10 * a11 = b10  => x10 = b10 / a11
    let x10 = b10 / a11;
    let x11 = b11 / a11;

    // Solve for x0 (top row of result)
    // x00 * a00 + x10 * a01 = b00 => x00 = (b00 - x10 * a01) / a00
    let x00 = (b00 - x10 * a01) / a00;
    let x01 = (b01 - x11 * a01) / a00;

    // Return in Column-Major format
    b[0] = x00;
    b[1] = x10;
    b[2] = x01;
    b[3] = x11;
}

fn scale_fallback(a: &[f64; 4], out: &mut [f64; 4], factor: f64) {
    out[0] = a[0] * factor;
    out[1] = a[1] * factor;
    out[2] = a[2] * factor;
    out[3] = a[3] * factor;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
fn scale_avx(a: &[f64; 4], out: &mut [f64; 4], factor: f64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256d, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256d, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    let factor_vec: __m256d = _mm256_set1_pd(factor);
    let src_vec: __m256d = unsafe { _mm256_loadu_pd(a.as_ptr()) };
    let result_vec: __m256d = _mm256_mul_pd(src_vec, factor_vec);
    unsafe { _mm256_storeu_pd(out.as_mut_ptr(), result_vec) };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
fn scale_sse(a: &[f64; 4], out: &mut [f64; 4], factor: f64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{__m128d, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{__m128d, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd};

    let factor_vec: __m128d = _mm_set1_pd(factor);

    let mut src_vec: __m128d = unsafe { _mm_loadu_pd(a.as_ptr()) };
    let mut result_vec: __m128d = _mm_mul_pd(src_vec, factor_vec);
    unsafe { _mm_storeu_pd(out.as_mut_ptr(), result_vec) };

    src_vec = unsafe { _mm_loadu_pd(a.as_ptr().add(2)) };
    result_vec = _mm_mul_pd(src_vec, factor_vec);
    unsafe { _mm_storeu_pd(out.as_mut_ptr().add(2), result_vec) };
}

fn daxpy_fallback(alpha: f64, a: &[f64; 4], out: &mut [f64; 4]) {
    out[0] += a[0] * alpha;
    out[1] += a[1] * alpha;
    out[2] += a[2] * alpha;
    out[3] += a[3] * alpha;
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
fn daxpy_avx(alpha: f64, a: &[f64; 4], out: &mut [f64; 4]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m256d, _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd,
    };

    let factor_vec: __m256d = _mm256_set1_pd(alpha);
    let src_vec: __m256d = unsafe { _mm256_loadu_pd(a.as_ptr()) };
    let mut result_vec: __m256d = _mm256_mul_pd(src_vec, factor_vec);
    result_vec = _mm256_add_pd(unsafe { _mm256_loadu_pd(out.as_ptr()) }, result_vec);
    unsafe { _mm256_storeu_pd(out.as_mut_ptr(), result_vec) };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
fn daxpy_sse(alpha: f64, a: &[f64; 4], out: &mut [f64; 4]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{
        __m128d, _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd,
    };
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{
        __m128d, _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd,
    };

    let factor_vec: __m128d = _mm_set1_pd(alpha);

    let mut src_vec: __m128d = unsafe { _mm_loadu_pd(a.as_ptr()) };
    let mut result_vec: __m128d = _mm_mul_pd(src_vec, factor_vec);
    result_vec = _mm_add_pd(unsafe { _mm_loadu_pd(out.as_ptr()) }, result_vec);
    unsafe { _mm_storeu_pd(out.as_mut_ptr(), result_vec) };

    src_vec = unsafe { _mm_loadu_pd(a.as_ptr().add(2)) };
    result_vec = _mm_mul_pd(src_vec, factor_vec);
    result_vec = _mm_add_pd(unsafe { _mm_loadu_pd(out.as_ptr().add(2)) }, result_vec);
    unsafe { _mm_storeu_pd(out.as_mut_ptr().add(2), result_vec) };
}

type FnScale = unsafe fn(&[f64; 4], &mut [f64; 4], f64) -> ();
type FnDaxpy = unsafe fn(f64, &[f64; 4], &mut [f64; 4]) -> ();

/// Computes the matrix exponential of a 2x2 f64 matrix in column-major order.
/// Input: `[a, b, c, d]` represents the matrix [[a, c], [b, d]]
#[allow(non_snake_case)]
pub fn matrix_exp_2x2(A: &[f64; 4], p: u32) -> [f64; 4] {
    let mut daxpy: FnDaxpy = daxpy_fallback;
    let mut scale: FnScale = scale_fallback;
    let mut dgemm: FnDGEM22 = dgemm_2x2_fallback;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            use crate::dgemm_2x2_avx2;

            dgemm = dgemm_2x2_avx2
        }
        if is_x86_feature_detected!("avx") {
            daxpy = daxpy_avx;
            scale = scale_avx;
        } else if is_x86_feature_detected!("sse2") {
            use crate::dgemm_2x2_sse2;

            daxpy = daxpy_sse;
            scale = scale_sse;
            dgemm = dgemm_2x2_sse2;
        }
    }

    let normA = (A[0].abs() + A[2].abs()).max(A[1].abs() + A[3].abs());
    let (_, e) = frexp(normA);
    let s: u32 = (e + 1).max(0) as u32;

    let mut P = [0.0; 4];
    P[0] = 1.0;
    P[3] = 1.0;

    let mut Q = [0.0; 4];
    Q[0] = 1.0;
    Q[3] = 1.0;

    let ps: f64 = 2.0f64.powf(s as f64);
    let is: f64 = 1.0f64 / ps;
    let mut Ak: [f64; 4] = [0.0; 4];
    unsafe { scale(A, &mut Ak, is) };
    let copyofA = Ak;
    let mut Aux = [0.0; 4];

    unsafe { daxpy(0.5, &Ak, &mut P) };
    unsafe { daxpy(-0.5, &Ak, &mut Q) };

    match p {
        1 => {}
        2 => {
            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C: f64 = 0.083333333333333_f64;
            unsafe { daxpy(C, &Aux, &mut P) };
            unsafe { daxpy(C, &Aux, &mut Q) };
        }
        3 => {
            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C1: f64 = 0.100000000000000_f64;
            unsafe { daxpy(C1, &Aux, &mut P) };
            unsafe { daxpy(C1, &Aux, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Aux.as_ptr(), Ak.as_mut_ptr()) };
            const C2: f64 = 0.008333333333333_f64;
            const MC2: f64 = -C2;
            unsafe { daxpy(C2, &Ak, &mut P) };
            unsafe { daxpy(MC2, &Ak, &mut Q) };
            println!("P: {P:?}");
            println!("Q: {Q:?}");
        }
        4 => {
            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C1: f64 = 0.107142857142857_f64;
            unsafe { daxpy(C1, &Aux, &mut P) };
            unsafe { daxpy(C1, &Aux, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Aux.as_ptr(), Ak.as_mut_ptr()) };
            const C2: f64 = 0.011904761904762_f64;
            const MC2: f64 = -C2;
            unsafe { daxpy(C2, &Ak, &mut P) };
            unsafe { daxpy(MC2, &Ak, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C3: f64 = 5.952_380_952_380_952e-4_f64;
            unsafe { daxpy(C3, &Aux, &mut P) };
            unsafe { daxpy(C3, &Aux, &mut Q) };
        }
        5 => {
            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C1: f64 = 0.111111111111111_f64;
            unsafe { daxpy(C1, &Aux, &mut P) };
            unsafe { daxpy(C1, &Aux, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Aux.as_ptr(), Ak.as_mut_ptr()) };
            const C2: f64 = 0.013888888888889_f64;
            const MC2: f64 = -C2;
            unsafe { daxpy(C2, &Ak, &mut P) };
            unsafe { daxpy(MC2, &Ak, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C3: f64 = 9.920_634_920_634_92e-4_f64;
            unsafe { daxpy(C3, &Aux, &mut P) };
            unsafe { daxpy(C3, &Aux, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Aux.as_ptr(), Ak.as_mut_ptr()) };
            const C4: f64 = 3.306_878_306_878_306e-5_f64;
            const MC4: f64 = -C4;
            unsafe { daxpy(C4, &Ak, &mut P) };
            unsafe { daxpy(MC4, &Ak, &mut Q) };
        }
        6 => {
            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C1: f64 = 0.113636363636364_f64;
            unsafe { daxpy(C1, &Aux, &mut P) };
            unsafe { daxpy(C1, &Aux, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Aux.as_ptr(), Ak.as_mut_ptr()) };
            const C2: f64 = 0.015151515151515_f64;
            const MC2: f64 = -C2;
            unsafe { daxpy(C2, &Ak, &mut P) };
            unsafe { daxpy(MC2, &Ak, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C3: f64 = 0.001262626262626_f64;
            unsafe { daxpy(C3, &Aux, &mut P) };
            unsafe { daxpy(C3, &Aux, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Aux.as_ptr(), Ak.as_mut_ptr()) };
            const C4: f64 = 6.313_131_313_131_313e-5_f64;
            const MC4: f64 = -C4;
            unsafe { daxpy(C4, &Ak, &mut P) };
            unsafe { daxpy(MC4, &Ak, &mut Q) };

            unsafe { dgemm(copyofA.as_ptr(), Ak.as_ptr(), Aux.as_mut_ptr()) };
            const C5: f64 = 1.503_126_503_126_503e-6_f64;
            unsafe { daxpy(C5, &Aux, &mut P) };
            unsafe { daxpy(C5, &Aux, &mut Q) };
        }
        _ => {
            panic!("poldegree must be between 2 and 7.");
        }
    }

    solve_2x2(&Q, &mut P);
    let poldegree = (s / 2) as i32;

    for _k in 0..poldegree {
        unsafe { dgemm(P.as_ptr(), P.as_ptr(), Aux.as_mut_ptr()) };
        unsafe { dgemm(Aux.as_ptr(), Aux.as_ptr(), P.as_mut_ptr()) };
    }

    if (s & 1) != 0 {
        unsafe { dgemm(P.as_ptr(), P.as_ptr(), Aux.as_mut_ptr()) };
        return Aux;
    }

    P
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        // exp(I) = e * I
        let e = std::f64::consts::E;
        let m = [1.0, 0.0, 0.0, 1.0];
        let res = matrix_exp_2x2(&m, 6);
        assert!((res[0] - e).abs() < 1e-15);
        assert!((res[1] - 0.0).abs() < 1e-15);
        assert!((res[2] - 0.0).abs() < 1e-15);
        assert!((res[3] - e).abs() < 1e-15);
    }

    #[test]
    fn test_rotation() {
        // Exponential of a skew-symmetric matrix yields a rotation matrix
        // A = [[0, -pi], [pi, 0]]
        let pi = std::f64::consts::PI;
        let m = [0.0, pi, -pi, 0.0];
        let res = matrix_exp_2x2(&m, 6);
        // Should yield [[cos(pi), -sin(pi)], [sin(pi), cos(pi)]] = [[-1, 0], [0, -1]]
        assert!((res[0] - -1.0).abs() < 1e-15);
        assert!((res[1] - 0.0).abs() < 1e-15);
        assert!((res[2] - 0.0).abs() < 1e-15);
        assert!((res[3] - -1.0).abs() < 1e-15);
    }

    #[test]
    fn test_rand_matrix() {
        // Exponential of a skew-symmetric matrix yields a rotation matrix
        // A = [[0, -pi], [pi, 0]]
        let m = [-2.0, 998.0, 1.0, -999.0];
        let res = matrix_exp_2x2(&m, 6);

        assert!((res[0] - 0.367511193482465).abs() < 1e-15);
        assert!((res[1] - 0.367511193482466).abs() < 1e-15);
        assert!((res[2] - 0.000368247688860).abs() < 1e-15);
        assert!((res[3] - 0.000368247688860).abs() < 1e-15);
    }
}
