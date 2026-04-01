#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

use math_helpers::matrix_exp_22::matrix_exp_2x2;
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};

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
            mexErrMsgTxt("rpade_no_norm_2x2: two input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("rpade_no_norm_2x2: too many output arguments.\n\0".as_ptr());
        }
    }

    let Amx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"rpade_no_norm_2x2: first argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let qmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"rpade_no_norm_2x2: second argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let dimensions = Amx.dimensions();
    if dimensions.len() != 2
        || *dimensions.first().unwrap_or(&1) != 2
        || *dimensions.get(1).unwrap_or(&1) != 2
    {
        unsafe { mexErrMsgTxt("rpade_no_norm_2x2: Input matrix must be 2x2!\n\0".as_ptr()) };
    }
    if !Amx.is_double() {
        unsafe { mexErrMsgTxt("rpade_no_norm_2x2:  Input matrix must be real!.\n\0".as_ptr()) };
    }

    if !qmx.is_double() || !qmx.is_scalar() {
        unsafe {
            mexErrMsgTxt("rpade_no_norm_2x2: Second argument must be a scalar.\n\0".as_ptr())
        };
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("rpade_no_norm_2x2: too many output arguments.\n\0".as_ptr());
        }
    }

    let Apt = Amx.get_ptr();
    let p: u32 = qmx.get_scalar() as u32;
    let A = unsafe { [*Apt.add(0), *Apt.add(1), *Apt.add(2), *Apt.add(3)] };

    let ans_matrix: *mut mxArray = unsafe { mxCreateDoubleMatrix(2, 2, mxComplexity_mxREAL) };

    unsafe { *plhs.add(0) = ans_matrix };
    let res = matrix_exp_2x2(&A, p);
    unsafe {
        std::ptr::copy_nonoverlapping(res.as_ptr(), ans_matrix.as_mut().unwrap().get_ptr(), 4)
    };
}
