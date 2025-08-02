#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::c_int;

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
    let xmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"expm64v4: second argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let ymx = match rhslice.get(2) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"expm64v4: third argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    let dimensions = Amx.dimensions();
    let nrows: usize = *dimensions.first().unwrap();
    let ncols: usize = *dimensions.get(1).unwrap();
    let mut pages: usize = 1;
    if dimensions.len() > 2 {
        pages = *dimensions.get(2).unwrap();
    }

    let xdimensions = xmx.dimensions().first().unwrap_or(&0);
    let ydimensions = ymx.dimensions();
    if dimensions.len() < 2 || nrows != ncols || ncols != *xmx.dimensions().first().unwrap_or(&1) {
        unsafe { mexErrMsgTxt("expm64: Input matrix must be square!\n\0".as_ptr()) };
    }
    if !Amx.is_double() {
        unsafe { mexErrMsgTxt("expm64:  Input matrix must be real!.\n\0".as_ptr()) };
    }

    if !xmx.is_double() {
        unsafe { mexErrMsgTxt("expm64: Second argument must be a vector.\n\0".as_ptr()) };
    }
    if !ymx.is_double() || *ymx.dimensions().get(1).unwrap_or(&0) != pages {
        unsafe {
            mexErrMsgTxt("expm64: Third argument must be a matrix with the same number of columns as the pages of the first argument.\n\0".as_ptr())
        };
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

    let rows: *const usize = &nrows;

    let A: *mut f64 = Amx.get_ptr();
    let x: *mut f64 = xmx.get_ptr();
    let y: *mut f64 = ymx.get_ptr();

    const CHN: *const u8 = "N\0".as_ptr();
    const ONE: *const f64 = &(1f64);
    const ONEI: *const usize = &(1usize);

    let ans_matrix: *mut mxArray =
        unsafe { mxCreateDoubleMatrix(nrows, pages, mxComplexity_mxREAL) };
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
    unsafe { *plhs.add(0) = ans_matrix };

    unsafe { std::ptr::copy_nonoverlapping(y, res, nrows * pages) };
    unsafe { dgemv(CHN, rows, rows, ONE, A, rows, x, ONEI, ONE, res, ONEI) };

    if pages == 1 {
        return;
    }

    for i in 1..pages {
        unsafe {
            dgemv(
                CHN,
                rows,
                rows,
                ONE,
                A.add(nrows * ncols * i),
                rows,
                x,
                ONEI,
                ONE,
                res.add(nrows * i),
                ONEI,
            )
        };
    }
}
