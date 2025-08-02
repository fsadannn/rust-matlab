use crate::identity::set_identity2_unrolled as set_identity2;
use math_helpers::{dtri_maxmy, scale_unrolled};
use matlab_blas_wrapper::blas::{dtrmm, dtrsm};

/// This function computes the Padé  approximation of the matrix exponential of `A` to the power `p` and stores it in `P`.
///
/// # Safety
/// This function is marked as `unsafe` because it calls several functions from the BLAS and LAPACK libraries that are not
/// safe to call.
///
/// # Panics
/// This function will panic if the LAPACK functions `dgesv` or BLAS functions `dgemm` returns an error.
#[allow(non_snake_case)]
pub unsafe fn pade_triangular(
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
    const SIDE: *const u8 = "L\0".as_ptr();
    const UPLO: *const u8 = "U\0".as_ptr();
    const ONE: *const f64 = &(1f64);
    let mut Q_rust: Vec<f64> = vec![0.0; total_size];
    let Q = Q_rust.as_mut_ptr();

    /* initialize to identity P, Q */
    set_identity2(P, Q, nrows, ncols);

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

    let mut c: f64 = 0.5f64;
    unsafe { dtri_maxmy(c, Ak, P, nrows) };
    let mut mc: f64 = -c;
    unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

    match p {
        2 => {
            c = 0.083333333333333_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };
        }
        3 => {
            c = 0.100000000000000_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.008333333333333_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };
        }
        4 => {
            c = 0.107142857142857_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.011904761904762_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 5.952_380_952_380_952e-4_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };
        }
        5 => {
            c = 0.111111111111111_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.013888888888889_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 9.920_634_920_634_92e-4_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 3.306_878_306_878_306e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };
        }
        6 => {
            c = 0.113636363636364_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.015151515151515_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 0.001262626262626_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 6.313_131_313_131_313e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 1.503_126_503_126_503e-6_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };
        }
        7 => {
            c = 0.11538461538461539_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.016025641025641024_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 0.001456876456876457_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 8.741_258_741_258_741e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 3.237_503_237_503_237_6e-6_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 5.781_255_781_255_781e-8_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    SIDE, UPLO, CHN, CHN, rows, rows, ONE, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };
        }
        _ => {
            return Err(String::from("poldegree must be between 2 and 7."));
        }
    }

    unsafe { dtrsm(SIDE, UPLO, CHN, CHN, rows, rows, ONE, Q, rows, P, rows) };

    for _k in 0..(s as isize) {
        unsafe {
            std::ptr::copy_nonoverlapping(P, Q, total_size);
            dtrmm(SIDE, UPLO, CHN, CHN, rows, rows, ONE, Q, rows, P, rows);
        };
    }

    Ok(())
}
