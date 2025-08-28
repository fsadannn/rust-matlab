#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

use math_helpers::daxpy;
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};
use matlab_blas_wrapper::blas::dgemv;

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

    if nrhs != 7 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lin_euler_maruyama_multi: 7 input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lin_euler_maruyama_multi: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("lin_euler_maruyama_multi: too few output arguments.\n\0".as_ptr());
        }
    }

    let Amx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: first argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let amx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let Bmx = match rhslice.get(2) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let bmx = match rhslice.get(3) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let x0mx = match rhslice.get(4) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let tmx = match rhslice.get(5) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let dWmx = match rhslice.get(6) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"lin_euler_maruyama_multi: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

    let mut dimensions = Amx.dimensions();
    let d = *dimensions.first().unwrap_or(&0);
    if dimensions.len() != 2 || d != *dimensions.get(1).unwrap_or(&0) {
        unsafe {
            mexErrMsgTxt(
                b"lin_euler_maruyama_multi: 1st argument must be a square matrix.\n\0".as_ptr(),
            );
        }
        return;
    }

    if *x0mx.dimensions().iter().max().unwrap_or(&1) != d {
        unsafe {
            mexErrMsgTxt(
                b"lin_euler_maruyama_multi: 5th argument must be a square matrix.\n\0".as_ptr(),
            );
        }
        return;
    }

    dimensions = Bmx.dimensions();
    if dimensions.len() > 3 || d != *dimensions.get(1).unwrap_or(&0) {
        unsafe {
            mexErrMsgTxt(
                b"lin_euler_maruyama_multi: 3rd argument must be a square matrix or 3d matrix with square pages.\n\0".as_ptr(),
            );
        }
        return;
    }
    let mut _m: usize = 1;
    if dimensions.len() == 3 {
        _m = *dimensions.get(2).unwrap_or(&1);
    }
    let m: usize = _m;

    let n: usize = *tmx.dimensions().iter().max().unwrap();

    let A: *mut f64 = Amx.get_ptr();
    let a: *mut f64 = amx.get_ptr();
    let B: *mut f64 = Bmx.get_ptr();
    let b: *mut f64 = bmx.get_ptr();
    let t: *mut f64 = tmx.get_ptr();
    let dW: *mut f64 = dWmx.get_ptr();

    const CHN: *const u8 = "N\0".as_ptr();
    const ONE: *const f64 = &(1f64);
    const ONEI: *const usize = &(1usize);

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(d, n, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };

    unsafe { std::ptr::copy_nonoverlapping(x0mx.get_ptr(), res, d) };

    if m == 1 {
        unsafe {
            mexErrMsgTxt(b"lin_euler_maruyama_multi: only multiple noises for now.\n\0".as_ptr());
        }
    }

    let mut yn: *mut f64 = unsafe { res.add(0) };
    let rows: *const usize = &d;
    let mut auxr: Vec<f64> = vec![0.0; d];
    let aux: *mut f64 = auxr.as_mut_ptr();
    #[allow(unused_assignments)]
    let mut II: f64 = 0f64;

    for i in 1..n {
        let I_1: *mut f64 = unsafe { dW.add(m * i) };
        let yn1: *mut f64 = unsafe { res.add(d * i) };
        let h: f64 = unsafe { *t.add(i) } - unsafe { *t.add(i - 1) };
        // yn1 = yn;
        unsafe { std::ptr::copy_nonoverlapping(yn, yn1, d) }
        // aux = a
        unsafe { std::ptr::copy_nonoverlapping(a, aux, d) }
        // aux = A*y_n + a
        unsafe { dgemv(CHN, rows, rows, ONE, A, rows, yn, ONEI, ONE, aux, ONEI) }
        // yn1 = yn1 + aux*h = yn1 + (A*y_n + a)*h;
        unsafe { daxpy(h, aux, yn1, d) };
        // yn1 = yn1 + A*aux*h^2/5 = A*(A*y_n + a)*h^2/2;
        unsafe {
            dgemv(
                CHN,
                rows,
                rows,
                &(h * h / 2f64),
                A,
                rows,
                aux,
                ONEI,
                ONE,
                yn1,
                ONEI,
            )
        }
        // // yn1 = yn1 + a * h;
        // unsafe { daxpy(h, a, yn1, d) };
        // // yn1 = yn1 + A * yn * h;
        // unsafe { dgemv(CHN, rows, rows, &h, A, rows, yn, ONEI, ONE, yn1, ONEI) }
        for j in 0..m {
            // aux = b(:,j)
            unsafe { std::ptr::copy_nonoverlapping(b.add(d * j), aux, d) }
            // aux = aux + B(:,j) * yn = b(:,j) + B(:,j) * yn
            unsafe {
                dgemv(
                    CHN,
                    rows,
                    rows,
                    ONE,
                    B.add(d * d * j),
                    rows,
                    yn,
                    ONEI,
                    ONE,
                    aux,
                    ONEI,
                )
            }
            // yn1 = yn1 + aux * I_1(j) = yn1 + (b(:,j) + B(:,j) * yn) * I_1(j)
            unsafe { daxpy(*I_1.add(j), aux, yn1, d) };
            for k in 0..m {
                II = unsafe { *I_1.add(j) } * unsafe { *I_1.add(k) };
                if j == k {
                    II -= h;
                }
                II /= 2.0f64;
                // yn1 = yn1 + B(:,k) * aux * I_{(j,k)} = yn1 + B(:,k) * (b(:,j) + B(:,j) * yn) * I_{(j,k)}
                unsafe {
                    dgemv(
                        CHN,
                        rows,
                        rows,
                        &II,
                        B.add(d * d * k),
                        rows,
                        aux,
                        ONEI,
                        ONE,
                        yn1,
                        ONEI,
                    )
                }
            }
        }
        yn = yn1;
    }
}
