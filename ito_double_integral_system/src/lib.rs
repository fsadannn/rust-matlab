#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod ito_integral;

use std::os::raw::c_int;

use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};

use crate::ito_integral::ito_double_integral;

#[allow(clippy::not_unsafe_ptr_arg_deref)]
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

    if nrhs != 2 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("ito_double_integral_system: 2 input argument required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("ito_double_integral: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("ito_double_integral_system: too few output arguments.\n\0".as_ptr());
        }
    }
    let dWmx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"ito_double_integral_system: first argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };
    let dimensions = dWmx.dimensions();

    if dimensions.len() > 2
        || (dimensions.len() == 2
            && *dimensions.first().unwrap_or(&1) == 1
            && *dimensions.get(1).unwrap_or(&1) == 1)
    {
        unsafe {
            mexErrMsgTxt(
                b"ito_double_integral_system: first argument must be a 2d array.\n\0".as_ptr(),
            );
        }
        return;
    }

    let hmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(
                    b"ito_double_integral_system: second argument must be a mxArray.\n\0".as_ptr(),
                );
            }
            return;
        }
    };

    if !hmx.is_scalar() {
        unsafe {
            mexErrMsgTxt(
                b"ito_double_integral_system: second argument must be a scalar.\n\0".as_ptr(),
            );
        }
        return;
    }

    let h: f64 = hmx.get_scalar();

    let dW_vec: *const f64 = dWmx.get_ptr();

    let m: usize = *dimensions.first().unwrap_or(&1);
    let n: usize = *dimensions.get(1).unwrap_or(&1);

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(m, m, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };

    ito_double_integral(n, m, dW_vec, h, res);
}
