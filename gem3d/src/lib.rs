#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;
mod dgem3d;

use matlab_base_wrapper::{
    mex::{mexErrMsgTxt, mexPrintf},
    mx::mxCreateNumericArray,
    raw::{Rhs, mxArray, mxClassID_mxDOUBLE_CLASS, mxComplexity_mxREAL},
};

use crate::dgem3d::dgem3d;

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
            mexErrMsgTxt("gem3d: three input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("gem3d: too many output arguments.\n\0".as_ptr());
        }
    }

    let Amx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"gem3d: first argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let Bmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"gem3d: second argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let ymx = match rhslice.get(2) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"gem3d: third argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    if !Amx.is_double() || !Bmx.is_double() || !ymx.is_double() {
        unsafe { mexErrMsgTxt("gem3d:  Input matrices must be real!.\n\0".as_ptr()) };
    }

    let a_dimensions = Amx.dimensions();
    let a_dim: [usize; 3] = [
        *a_dimensions.first().unwrap_or(&1),
        *a_dimensions.get(1).unwrap_or(&1),
        *a_dimensions.get(2).unwrap_or(&1),
    ];

    let B_dimensions = Bmx.dimensions();
    let b_dim: [usize; 3] = [
        *B_dimensions.first().unwrap_or(&1),
        *B_dimensions.get(1).unwrap_or(&1),
        *B_dimensions.get(2).unwrap_or(&1),
    ];
    let y_dimensions = ymx.dimensions();
    let y_dim: [usize; 3] = [
        *y_dimensions.first().unwrap_or(&1),
        *y_dimensions.get(1).unwrap_or(&1),
        *y_dimensions.get(2).unwrap_or(&1),
    ];

    if (a_dim[1] == b_dim[0] && a_dim[2] != b_dim[2] && (a_dim[2] != b_dim[1] || b_dim[2] != 1))
        || (a_dim[1] != b_dim[0] && (b_dim[0] != 1 || b_dim[1] != a_dim[2] || b_dim[2] != 1))
    {
        unsafe {
            mexPrintf(
                "gem3d: Dimensions of A are [%zu %zu %zu], of B are [%zu %zu %zu] and of y are [%zu %zu %zu].\n\0".as_ptr(),
                a_dim[0], a_dim[1], a_dim[2],
                b_dim[0], b_dim[1], b_dim[2],
                y_dim[0], y_dim[1], y_dim[2],
            );
        }
        unsafe { mexErrMsgTxt("gem3d: Dimensions mismatch!\n\0".as_ptr()) };
    }

    let cross_page_dims = [
        a_dim[0],
        if a_dim[1] == b_dim[0] {
            if b_dim[2] == 1 { 1 } else { b_dim[1] }
        } else {
            a_dim[1]
        },
        a_dim[2],
    ];

    let ans_matrix: *mut mxArray;
    if cross_page_dims[1] == 1 && cross_page_dims[2] == 1 {
        ans_matrix = unsafe {
            mxCreateNumericArray(
                2,
                cross_page_dims[..2].as_ptr(),
                mxClassID_mxDOUBLE_CLASS,
                mxComplexity_mxREAL,
            )
        };
    } else if cross_page_dims[1] == 1 && cross_page_dims[2] != 1 {
        let dimm = [cross_page_dims[0], cross_page_dims[2]];
        ans_matrix = unsafe {
            mxCreateNumericArray(
                2,
                dimm.as_ptr(),
                mxClassID_mxDOUBLE_CLASS,
                mxComplexity_mxREAL,
            )
        };
    } else {
        ans_matrix = unsafe {
            mxCreateNumericArray(
                3,
                cross_page_dims.as_ptr(),
                mxClassID_mxDOUBLE_CLASS,
                mxComplexity_mxREAL,
            )
        };
    }
    let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };

    match dgem3d(
        Amx.get_ptr(),
        &a_dim,
        Bmx.get_ptr(),
        &b_dim,
        ymx.get_ptr(),
        &y_dim,
        res,
    ) {
        Ok(_) => {}
        Err(_) => {
            unsafe {
                mexPrintf(
                "gem3d: Dimensions of A are [%zu %zu %zu], of B are [%zu %zu %zu] and of y are [%zu %zu %zu].\n\0".as_ptr(),
                a_dim[0], a_dim[1], a_dim[2],
                b_dim[0], b_dim[1], b_dim[2],
                y_dim[0], y_dim[1], y_dim[2],
            );
            }
            unsafe { mexErrMsgTxt("gem3d: Dimensions mismatch!\n\0".as_ptr()) };
        }
    }
}
