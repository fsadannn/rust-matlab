#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod taylor;

use std::os::raw::c_int;

use math_helpers::{daxpy, dxpy, scale_unrolled};
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};
use matlab_blas_wrapper::blas::{dgemm, dgemv};

use crate::taylor::exp_matrix_action;

const CHN: *const u8 = "N\0".as_ptr();
const ONE: *const f64 = &(1f64);
const ONEI: *const usize = &(1usize);

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

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(d, n, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };

    unsafe { std::ptr::copy_nonoverlapping(x0mx.get_ptr(), res, d) };

    if m == 1 {
        unsafe {
            mexErrMsgTxt(b"lin_euler_maruyama_multi: only multiple noises for now.\n\0".as_ptr());
        }
    }

    let d2 = d * d;
    let mut yn: *mut f64 = unsafe { res.add(0) };
    let rows: *const usize = &d;
    let bcols: *const usize = &m;
    let mut auxr: Vec<f64> = vec![0.0; d];
    let aux: *mut f64 = auxr.as_mut_ptr();
    let mut auxr2: Vec<f64> = vec![0.0; d];
    let aux2: *mut f64 = auxr2.as_mut_ptr();
    let mut Cr: Vec<f64> = vec![0.0; d2];
    let C: *mut f64 = Cr.as_mut_ptr();
    let mut exr: Vec<f64> = vec![0.0; d];
    let ex: *mut f64 = exr.as_mut_ptr();

    for i in 1..n {
        let I_1: *mut f64 = unsafe { dW.add(m * i) };
        let yn1: *mut f64 = unsafe { res.add(d * i) };
        let h: f64 = unsafe { *t.add(i) } - unsafe { *t.add(i - 1) };
        // yn1 = yn1 + a * h;
        unsafe { daxpy(h, a, yn1, d) };
        // yn1 = yn1 + b * I_1;
        unsafe { dgemv(CHN, rows, bcols, ONE, b, rows, I_1, ONEI, ONE, yn1, ONEI) };
        // C = A*h
        unsafe { scale_unrolled(A, C, d2, h) };
        for j in 0..m {
            // C = C + -h/2*B(:,:,j)^2 = A*h -h/2* B(:,:,j)^2
            unsafe {
                dgemm(
                    CHN,
                    CHN,
                    rows,
                    rows,
                    rows,
                    &(-h / 2f64),
                    B.add(d2 * j),
                    rows,
                    B.add(d2 * j),
                    rows,
                    ONE,
                    C,
                    rows,
                )
            };
            // C = C + B(:,:,j) * I_(j)
            unsafe { daxpy(*I_1.add(j), B.add(d2 * j), C, d2) };

            // yn1 = yn1 + B(:,:,j) * b(:,j) * (I_(j)^2 - h)/2
            unsafe {
                dgemv(
                    CHN,
                    rows,
                    rows,
                    &((*I_1.add(j) * *I_1.add(j) - h) / 2f64),
                    B.add(d2 * j),
                    rows,
                    b.add(d * j),
                    ONEI,
                    ONE,
                    yn1,
                    ONEI,
                )
            }

            for k in 0..m {
                if k == j {
                    continue;
                }

                // yn1 = yn1 + B(:,:,j) * b(:,k) * I_(j)*I_(k)/2
                unsafe {
                    dgemv(
                        CHN,
                        rows,
                        rows,
                        &((*I_1.add(j) * *I_1.add(k)) / 2f64),
                        B.add(d * d * j),
                        rows,
                        b.add(d * k),
                        ONEI,
                        ONE,
                        yn1,
                        ONEI,
                    )
                }
            }
        }
        // ex = exp(C);
        exp_matrix_action(C, yn, ex, aux, aux2, d);
        unsafe { dxpy(ex, yn1, d) };
        yn = yn1;
    }
}
