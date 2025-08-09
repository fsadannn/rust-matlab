#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::os::raw::c_int;

use matlab_base_wrapper::{
    mex::mexErrMsgTxt,
    mx::mxCreateDoubleMatrix,
    raw::{Rhs, mxArray, mxComplexity_mxREAL},
};
use rpade_shared::pade_triangular;

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
            mexErrMsgTxt("expm64v4: three input arguments required.\n\0".as_ptr());
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

    let A: *mut f64 = Amx.get_ptr();
    let p = qmx.get_scalar() as i32;
    let s = smx.get_scalar().ceil();

    let ans_matrix: *mut mxArray =
        unsafe { mxCreateDoubleMatrix(nrows, ncols, mxComplexity_mxREAL) };

    unsafe { *plhs.add(0) = ans_matrix };

    /* P and Q will store the matrix polynomials, are initialized
     * to identity */
    let P = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    let res = unsafe { pade_triangular(P, A, p, s, nrows, ncols) };
    match res {
        Ok(()) => (),
        Err(e) => unsafe { mexErrMsgTxt(e.as_ptr()) },
    }
}
