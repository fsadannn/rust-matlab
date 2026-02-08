#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::{mem::swap, os::raw::c_int};

use math_helpers::{FnDGEM22, dgemm_2x2_fallback, matrix_exp_22::matrix_exp_2x2};
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};

#[allow(unused_variables, clippy::not_unsafe_ptr_arg_deref)]
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

    if nrhs != 5 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lambert_3_15: 5 input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lambert_3_15: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lambert_3_15: too few output arguments.\n\0".as_ptr());
        }
    }

    let sigma_1 = match rhslice.first() {
        Some(a) => {
            if !a.is_scalar() {
                unsafe {
                    mexErrMsgTxt(b"lambert_3_15: first argument must be a number.\n\0".as_ptr());
                }
            }
            a.get_scalar()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: first argument must be a number.\n\0".as_ptr());
            }
            return;
        }
    };
    let x0 = match rhslice.get(1) {
        Some(a) => {
            let max_dim = *a.dimensions().iter().max().unwrap_or(&0);
            if max_dim != 2 {
                unsafe {
                    mexErrMsgTxt(
                        b"lambert_3_15: second argument must be a vector of 2 components.\n\0"
                            .as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: second argument must be a vector.\n\0".as_ptr());
            }
            return;
        }
    };

    let (t, n) = match rhslice.get(2) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_3_15: 3rd argument must be a vector.\n\0".as_ptr());
                }
            }
            let n: usize = *a.dimensions().iter().max().unwrap();
            (a.get_slice(), n)
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: 3rd argument must be a vector.\n\0".as_ptr());
            }
            return;
        }
    };

    let dW = match rhslice.get(3) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_3_15: 4rd argument must be a mxArray.\n\0".as_ptr());
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"lambert_3_15: 3rd and 4rd argument must be the same size.\n\0".as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: 4th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let dZ = match rhslice.get(4) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_3_15: 5rd argument must be a mxArray.\n\0".as_ptr());
                }
            }
            let nn: usize = *a.dimensions().iter().max().unwrap();
            if nn != n {
                unsafe {
                    mexErrMsgTxt(
                        b"lambert_3_15: 3rd and 5rd argument must be the same size.\n\0".as_ptr(),
                    );
                }
            }
            a.get_slice()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: 5th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let mut dgemm: FnDGEM22 = dgemm_2x2_fallback;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            use math_helpers::dgemm_2x2_avx2;

            dgemm = dgemm_2x2_avx2
        } else if is_x86_feature_detected!("sse2") {
            use math_helpers::dgemm_2x2_sse2;

            dgemm = dgemm_2x2_sse2;
        }
    }

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(2, n, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };
    unsafe {
        *res.add(0) = x0[0];
    };
    unsafe {
        *res.add(1) = x0[1];
    };
    let mut yn = [0.0; 2];
    yn[0] = x0[0];
    yn[1] = x0[1];

    let mut a = [0.0; 2];
    let mut b = [0.0; 2];

    // assume the evenly spaced partition
    let h = unsafe { *t.get_unchecked(1) - *t.get_unchecked(0) };
    let h2_2 = h.powf(2.0) / 2.0;
    const A: [f64; 4] = [-2.0, 998.0, 1.0, -999.0];
    let Ah: [f64; 4] = [A[0] * h, A[1] * h, A[2] * h, A[3] * h];
    let expAh = matrix_exp_2x2(&Ah, 6);
    let mut expAt = [1.0, 0.0, 0.0, 1.0];
    let mut aux = [0.0; 4];

    for i in 1..n {
        let t_n = unsafe { *t.get_unchecked(i - 1) };
        // let h = unsafe { *t.get_unchecked(i) - *t.get_unchecked(i - 1) };
        let I_1 = unsafe { *dW.get_unchecked(i) };
        let I_10 = unsafe { *dZ.get_unchecked(i) };
        a[0] = A[0] * yn[0] + A[2] * yn[1] + 2.0 * t_n.sin();
        a[1] = A[1] * yn[0] + A[3] * yn[1] + 998.0 * (t_n.cos() - t_n.sin());
        // expAt = matrix_exp_2x2(&[-2.0 * t_n, 998.0 * t_n, 1.0 * t_n, -999.0 * t_n], 6);
        b[0] = sigma_1 * (expAt[0] + expAt[2]);
        b[1] = sigma_1 * (expAt[1] + expAt[3]);
        unsafe {
            dgemm(expAt.as_ptr(), expAh.as_ptr(), aux.as_mut_ptr());
        }
        swap(&mut expAt, &mut aux);
        yn[0] += a[0] * h
            + b[0] * I_1
            + (A[0] * a[0] + A[2] * a[1]) * h2_2
            + (A[0] * b[0] + A[2] * b[1]) * I_10;
        yn[1] += a[1] * h
            + b[1] * I_1
            + (A[1] * a[0] + A[3] * a[1]) * h2_2
            + (A[1] * b[0] + A[3] * b[1]) * I_10;
        unsafe {
            *res.add(2 * i) = yn[0];
            *res.add(2 * i + 1) = yn[1];
        }
    }
}
