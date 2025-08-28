#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

use math_helpers::frexp;
use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};
use matlab_lapack_wrapper::helpers::norm_inf;
use rpade_shared::pade;

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

    let A = Amx.get_ptr();
    let p: i32 = qmx.get_scalar() as i32;

    let normA = unsafe { norm_inf(nrows, A, nrows) };
    let (_, e) = frexp(normA);
    let s: u32 = std::cmp::max(0, e + 1).try_into().unwrap();

    let ans_matrix: *mut mxArray =
        unsafe { mxCreateDoubleMatrix(nrows, ncols, mxComplexity_mxREAL) };

    unsafe { *plhs.add(0) = ans_matrix };

    let P = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };

    let res = unsafe { pade(P, A, p, s as f64, nrows, ncols) };
    match res {
        Ok(()) => (),
        Err(e) => unsafe { mexErrMsgTxt(e.as_ptr()) },
    }
}
