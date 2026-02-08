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
            a.get_ptr()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: second argument must be a vector.\n\0".as_ptr());
            }
            return;
        }
    };

    let t = match rhslice.get(2) {
        Some(a) => {
            if a.dimensions().len() > 2 {
                unsafe {
                    mexErrMsgTxt(b"lambert_3_15: 3rd argument must be a vector.\n\0".as_ptr());
                }
            }
            a.get_ptr()
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
            a.get_ptr()
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
            a.get_ptr()
        }
        None => {
            unsafe {
                mexErrMsgTxt(b"lambert_3_15: 5th argument must be mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
}
