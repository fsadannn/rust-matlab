use crate::identity::set_identity2;
use math_helpers::{
    FnDaxpy, FnScale, daxpy_avx, daxpy_fallback, daxpy_simd, scale_unrolled_avx,
    scale_unrolled_fallback, scale_unrolled_simd,
};
use matlab_blas_wrapper::blas::dgemm;
use matlab_lapack_wrapper::lapack::dgesv;
use std::ops::Rem;

/// This function computes the Padé  approximation of the matrix exponential of `A` to the power `p` and stores it in `P`.
///
/// # Safety
/// This function is marked as `unsafe` because it calls several functions from the BLAS and LAPACK libraries that are not
/// safe to call.
///
/// # Panics
/// This function will panic if the LAPACK functions `dgesv` or BLAS functions `dgemm` returns an error.
#[allow(non_snake_case)]
pub unsafe fn pade(
    P: *mut f64,
    A: *mut f64,
    p: i32,
    s: f64,
    nrows: usize,
    ncols: usize,
) -> Result<(), String> {
    let total_size = nrows * ncols;
    let rows: *const usize = &nrows;
    const CHN: *const u8 = "N\0".as_ptr();
    const ONE: *const f64 = &(1f64);
    const ZERO: *const f64 = &(0f64);

    let mut daxpy: FnDaxpy = daxpy_fallback;
    let mut scale_unrolled: FnScale = scale_unrolled_fallback;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") && total_size >= 4 {
            daxpy = daxpy_avx;
            scale_unrolled = scale_unrolled_avx;
        } else if is_x86_feature_detected!("sse2") && total_size >= 2 {
            daxpy = daxpy_simd;
            scale_unrolled = scale_unrolled_simd;
        }
    }

    /* P and Q will store the matrix polynomials, are initialized
     * to identity */
    let mut Q_rust: Vec<f64> = vec![0.0; total_size];
    let Q = Q_rust.as_mut_ptr();

    /* initialize to identity P, Q */
    unsafe { set_identity2(P, Q, nrows, ncols) };

    /* s = 2^s; */
    let ps: f64 = 2.0f64.powf(s);
    let is: f64 = 1.0f64 / ps;

    let mut Ak_M: Vec<f64> = Vec::with_capacity(total_size);
    let Ak = Ak_M.as_mut_ptr();
    /* Ak = A*(1/s) */
    unsafe { scale_unrolled(A, Ak, total_size, is) };
    // since Ak_M has `uninitialized` elements, after we filled it, we need to set its length
    unsafe { Ak_M.set_len(total_size) };
    // unsafe {
    //     mexPrintf(format!("{:?}\n\0", Ak_M).as_ptr() as *const u8);
    // }

    let mut copyofA_M: Vec<f64> = Ak_M.clone();
    let copyofA = copyofA_M.as_mut_ptr();

    let mut Aux_M: Vec<f64> = vec![0.0; total_size];
    let Aux: *mut f64 = Aux_M.as_mut_ptr();

    let mut c: f64 = 0.5f64;
    unsafe { daxpy(c, Ak, P, total_size) };
    let mut mc: f64 = -c;
    unsafe { daxpy(mc, Ak, Q, total_size) };

    match p {
        1 => {}
        2 => {
            c = 0.083333333333333_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };
        }
        3 => {
            c = 0.100000000000000_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 0.008333333333333_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };
        }
        4 => {
            c = 0.107142857142857_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 0.011904761904762_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };

            c = 5.952_380_952_380_952e-4_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };
        }
        5 => {
            c = 0.111111111111111_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 0.013888888888889_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };

            c = 9.920_634_920_634_92e-4_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 3.306_878_306_878_306e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };
        }
        6 => {
            c = 0.113636363636364_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 0.015151515151515_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };

            c = 0.001262626262626_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 6.313_131_313_131_313e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };

            c = 1.503_126_503_126_503e-6_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };
        }
        7 => {
            c = 0.11538461538461539_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 0.016025641025641024_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };

            c = 0.001456876456876457_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 8.741_258_741_258_741e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };

            c = 3.237_503_237_503_237_6e-6_f64;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Ak, rows, ZERO, Aux, rows,
                )
            };
            unsafe { daxpy(c, Aux, P, total_size) };
            unsafe { daxpy(c, Aux, Q, total_size) };

            c = 5.781_255_781_255_781e-8_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dgemm(
                    CHN, CHN, rows, rows, rows, ONE, copyofA, rows, Aux, rows, ZERO, Ak, rows,
                )
            };
            unsafe { daxpy(c, Ak, P, total_size) };
            unsafe { daxpy(mc, Ak, Q, total_size) };
        }
        _ => {
            return Err(String::from("poldegree must be between 2 and 7."));
        }
    }

    let mut iPivot_r: Vec<isize> = vec![0; nrows];
    let iPivot: *mut isize = iPivot_r.as_mut_ptr();

    let mut info: isize = 0;
    unsafe { dgesv(rows, rows, Q, rows, iPivot, P, rows, &mut info) };

    let poldegree = (s / 2f64).floor() as i32;
    for _k in 0..poldegree {
        unsafe {
            dgemm(
                CHN, CHN, rows, rows, rows, ONE, P, rows, P, rows, ZERO, Aux, rows,
            );
            dgemm(
                CHN, CHN, rows, rows, rows, ONE, Aux, rows, Aux, rows, ZERO, P, rows,
            );
        };
    }

    if (s as i32).rem(2) != 0 {
        unsafe {
            dgemm(
                CHN, CHN, rows, rows, rows, ONE, P, rows, P, rows, ZERO, Aux, rows,
            )
        };
        unsafe { std::ptr::copy_nonoverlapping(Aux, P, total_size) };
    }

    Ok(())
}
