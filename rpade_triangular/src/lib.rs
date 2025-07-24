#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod identity;

use std::os::raw::c_int;

use math_helpers::{dtri_maxmy, scale_unrolled};
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};
use matlab_blas_wrapper::blas::{dtrmm, dtrsm};

use crate::identity::setIdentity2_unrolled;

#[allow(unused_variables)]
#[unsafe(no_mangle)]
pub extern "C" fn mexFunction(
    nlhs: c_int,
    plhs: *mut *mut mxArray,
    nrhs: c_int,
    prhs: *mut *mut mxArray,
) {
    let rhslice: Rhs =
        unsafe { ::std::slice::from_raw_parts(prhs as *const &mxArray, nrhs as usize) };
    let lhslice = unsafe {
        ::std::slice::from_raw_parts_mut(plhs as *mut Option<&mut mxArray>, nlhs as usize)
    };

    if nrhs != 3 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("expm64v4: two input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("expm64v4: too many output arguments.\n\0".as_ptr());
        }
    }

    let Amx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"expm64v4: first argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let qmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"expm64v4: second argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let smx = match rhslice.get(2) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"expm64v4: third argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let dimensions = Amx.dimensions();
    if dimensions.len() != 2
        || *dimensions.first().unwrap_or(&0) != *dimensions.get(1).unwrap_or(&1)
    {
        unsafe { mexErrMsgTxt("expm64: Input matrix must be square!\n\0".as_ptr()) };
    }
    if !Amx.is_double() {
        unsafe { mexErrMsgTxt("expm64:  Input matrix must be real!.\n\0".as_ptr()) };
    }

    if !qmx.is_double() || !qmx.is_scalar() {
        unsafe { mexErrMsgTxt("expm64: Second argument must be a scalar.\n\0".as_ptr()) };
    }
    if !smx.is_scalar() {
        unsafe { mexErrMsgTxt("expm64: SecoThirdnd argument must be a scalar.\n\0".as_ptr()) };
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("expm64v4: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("expm64v4: too few output arguments.\n\0".as_ptr());
        }
    }

    let nrows: usize = *dimensions.first().unwrap();

    let ncols: usize = *dimensions.get(1).unwrap();

    let total_size = nrows * ncols;
    let A: *mut f64 = Amx.get_ptr();
    let p = qmx.get_scalar() as i32;
    let s = smx.get_scalar().ceil();

    let chn: *const u8 = "N\0".as_ptr();
    let side = "L\0".as_ptr();
    let uplo = "U\0".as_ptr();
    let rows: *const usize = &nrows;
    let one: *const f64 = &(1f64);
    let zero: *const f64 = &(0f64);

    let ans_matrix: *mut mxArray =
        unsafe { mxCreateDoubleMatrix(nrows, ncols, mxComplexity_mxREAL) };

    unsafe { *plhs.add(0) = ans_matrix };

    /* P and Q will store the matrix polynomials, are initialized
     * to identity */
    let P = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    let mut Q_rust: Vec<f64> = vec![0.0; total_size];
    let Q = Q_rust.as_mut_ptr();

    /* initialize to identity P, Q */
    setIdentity2_unrolled(P, Q, nrows, ncols);

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
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
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
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.008333333333333_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
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
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.011904761904762_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 5.952_380_952_380_952e-4_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
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
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.013888888888889_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 9.920_634_920_634_92e-4_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 3.306_878_306_878_306e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
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
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.015151515151515_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 0.001262626262626_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 6.313_131_313_131_313e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 1.503_126_503_126_503e-6_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
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
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 0.016025641025641024_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 0.001456876456876457_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 8.741_258_741_258_741e-5_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };

            c = 3.237_503_237_503_237_6e-6_f64;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(c, Ak, Q, nrows) };

            c = 5.781_255_781_255_781e-8_f64;
            mc = -c;
            /* Ak = A*Ak; */
            unsafe {
                dtrmm(
                    side, uplo, chn, chn, rows, rows, one, copyofA, rows, Ak, rows,
                );
            }
            unsafe { dtri_maxmy(c, Ak, P, nrows) };
            unsafe { dtri_maxmy(mc, Ak, Q, nrows) };
        }
        _ => {
            unsafe { mexErrMsgTxt("poldegree must be between 2 and 7.\n\0".as_ptr()) };
        }
    }

    unsafe { dtrsm(side, uplo, chn, chn, rows, rows, one, Q, rows, P, rows) };

    for k in 0..(s as isize) {
        unsafe {
            std::ptr::copy_nonoverlapping(P, Q, total_size);
            dtrmm(side, uplo, chn, chn, rows, rows, one, Q, rows, P, rows);
        };
    }
}
