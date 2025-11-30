#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use std::os::raw::c_int;

use math_helpers::{
    FnDxpy, FnScale, dxpy_avx, dxpy_fallback, dxpy_simd, scale_unrolled_avx,
    scale_unrolled_fallback, scale_unrolled_simd,
};
use matlab_base_wrapper::{
    mex::{mexErrMsgTxt, mexPrintf},
    mx::{mxCreateDoubleMatrix, mxCreateNumericArray},
    raw::{Rhs, mxArray, mxClassID_mxDOUBLE_CLASS, mxComplexity_mxREAL},
};
use matlab_blas_wrapper::blas::{dgemm, dgemv};

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
            mexErrMsgTxt("gemv3d: three input arguments required.\n\0".as_ptr());
        }
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("gemv3d: too many output arguments.\n\0".as_ptr());
        }
    }

    let Amx = match rhslice.first() {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"gemv3d: first argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let xmx = match rhslice.get(1) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"gemv3d: second argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };
    let ymx = match rhslice.get(2) {
        Some(a) => a,
        None => {
            unsafe {
                mexErrMsgTxt(b"gemv3d: third argument must be a mxArray.\n\0".as_ptr());
            }
            return;
        }
    };

    if !Amx.is_double() || !xmx.is_double() || !ymx.is_double() {
        unsafe { mexErrMsgTxt("gemv3d:  Input matrices must be real!.\n\0".as_ptr()) };
    }

    let dimensions = Amx.dimensions();
    let nrows: usize = *dimensions.first().unwrap();
    let ncols: usize = *dimensions.get(1).unwrap();
    let mut pages: usize = 1;
    if dimensions.len() > 2 {
        pages = *dimensions.get(2).unwrap();
    }

    let xdimensions = xmx.dimensions();
    let ydimensions = ymx.dimensions();
    if dimensions.len() < 2
        || (ncols != *xdimensions.first().unwrap_or(&1) && *xdimensions.first().unwrap_or(&1) != 1)
        || (xdimensions.len() == 3 && *xdimensions.get(2).unwrap_or(&1) != pages)
        || (xdimensions.len() < 3
            && *xdimensions.get(1).unwrap_or(&1) != pages
            && *xdimensions.get(1).unwrap_or(&1) != 1)
        || (ydimensions.len() == 3 && *ydimensions.get(2).unwrap_or(&1) != pages)
        || (ydimensions.len() < 3
            && *ydimensions.get(1).unwrap_or(&1) != pages
            && *ydimensions.get(1).unwrap_or(&1) != 1
            && *ydimensions.get(1).unwrap_or(&1) != (ncols).max(*xdimensions.get(1).unwrap_or(&1)))
    {
        unsafe {
            mexPrintf(
                "gemv3d: Dimensions of A are [%zu %zu %zu], of x are [%zu %zu %zu] and of y are [%zu %zu %zu].\n\0".as_ptr(),
                dimensions[0], dimensions[1], pages, xdimensions[0], xdimensions[1], *xdimensions.get(2).unwrap_or(&1),
                ydimensions[0], ydimensions[1], *ydimensions.get(2).unwrap_or(&1)
            );
        }
        unsafe { mexErrMsgTxt("gemv3d: Dimensions mismatch!\n\0".as_ptr()) };
    }

    if !ymx.is_double() || (*ymx.dimensions().get(1).unwrap_or(&0) != pages && !ymx.is_scalar()) {
        unsafe {
            mexErrMsgTxt("gemv3d: Third argument must be a matrix with the same number of columns as the pages of the first argument.\n\0".as_ptr())
        };
    }

    if nlhs > 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("gemv3d: too many output arguments.\n\0".as_ptr());
        }
    }

    if nlhs < 1 {
        // Letting the standard library do the work of making Rusts strings C-compatible
        unsafe {
            mexErrMsgTxt("gemv3d: too few output arguments.\n\0".as_ptr());
        }
    }

    let mut dxpy: FnDxpy = dxpy_fallback;
    let mut scale_unrolled: FnScale = scale_unrolled_fallback;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            dxpy = dxpy_avx;
            scale_unrolled = scale_unrolled_avx;
        } else if is_x86_feature_detected!("sse2") {
            dxpy = dxpy_simd;
            scale_unrolled = scale_unrolled_simd;
        }
    }

    let rows: *const usize = &nrows;
    let cols: *const usize = &ncols;

    let A: *mut f64 = Amx.get_ptr();
    let x: *mut f64 = xmx.get_ptr();
    let y: *mut f64 = ymx.get_ptr();

    const CHN: *const u8 = "N\0".as_ptr();
    const ONE: *const f64 = &(1f64);
    const ONEI: *const usize = &(1usize);

    // A is a 3d matrix of N x M * pages
    // case: 3d * (1 x ?)
    // broadcast the value value times each 2d matrices of pages
    if *xdimensions.first().unwrap_or(&1) == 1 {
        if ncols != *ydimensions.get(1).unwrap_or(&1) {
            unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size 2.\n\0".as_ptr()) };
        }

        let mut dims: [usize; 3] = [0, 0, 0];
        dims[0] = nrows;
        dims[1] = ncols;
        dims[2] = pages;
        let ans_matrix: *mut mxArray = unsafe {
            mxCreateNumericArray(
                3,
                dims.as_ptr(),
                mxClassID_mxDOUBLE_CLASS,
                mxComplexity_mxREAL,
            )
        };
        let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
        unsafe { *plhs.add(0) = ans_matrix };

        let mut should_scale = true;
        if *xdimensions.get(1).unwrap_or(&1) == 1 {
            unsafe { scale_unrolled(A, res, nrows * ncols * pages, *x.add(0)) };
            should_scale = false;
        } else {
            unsafe { std::ptr::copy_nonoverlapping(A, res, nrows * ncols * pages) };
        }

        match ydimensions.len() {
            2 => {
                let page_size = nrows * ncols;
                for i in 0..pages {
                    if should_scale {
                        unsafe {
                            scale_unrolled(
                                res.add(page_size * i),
                                res.add(page_size * i),
                                page_size,
                                *x.add(i),
                            )
                        };
                    }

                    unsafe { dxpy(y, res.add(page_size * i), page_size) };
                }
            }
            3 => {
                let page_size = nrows * ncols;
                for i in 0..pages {
                    if should_scale {
                        unsafe {
                            scale_unrolled(
                                res.add(page_size * i),
                                res.add(page_size * i),
                                page_size,
                                *x.add(i),
                            )
                        };
                    }

                    unsafe { dxpy(y.add(page_size * i), res.add(page_size * i), page_size) };
                }
            }
            1 => {
                if !ymx.is_scalar() || ymx.get_scalar() != 0f64 {
                    unsafe {
                        mexPrintf(
                            "xdism: %d - %d - %d\n\0".as_ptr(),
                            ydimensions[0],
                            ydimensions[1],
                            *ydimensions.get(2).unwrap_or(&1),
                        )
                    };
                    unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size (other).\n\0".as_ptr()) };
                }
            }
            _ => {
                unsafe {
                    mexPrintf(
                        "xdism: %d - %d - %d\n\0".as_ptr(),
                        ydimensions[0],
                        ydimensions[1],
                        *ydimensions.get(2).unwrap_or(&1),
                    )
                };
                unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size (other).\n\0".as_ptr()) };
            }
        }

        return;
    }

    // case: 3d * vec
    // multiply across the pages axis each 2d matrix times the same vector resulting in a 2d matrix of N x pages
    if xdimensions.len() < 3 && ydimensions.len() < 3 && *xdimensions.get(1).unwrap_or(&1) == 1 {
        let ans_matrix: *mut mxArray =
            unsafe { mxCreateDoubleMatrix(nrows, pages, mxComplexity_mxREAL) };
        let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
        unsafe { *plhs.add(0) = ans_matrix };

        // 2d ( N x pages )
        // sum to the result each vector across the pages
        if *ydimensions.get(1).unwrap_or(&1) != 1 {
            unsafe { std::ptr::copy_nonoverlapping(y, res, nrows * pages) };
            unsafe { dgemv(CHN, rows, rows, ONE, A, rows, x, ONEI, ONE, res, ONEI) };

            for i in 1..pages {
                unsafe {
                    dgemv(
                        CHN,
                        rows,
                        cols,
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
        } else {
            // + vec
            // sum to the result the same vector across the pages
            unsafe { std::ptr::copy_nonoverlapping(y, res, nrows) };
            unsafe { dgemv(CHN, rows, rows, ONE, A, rows, x, ONEI, ONE, res, ONEI) };

            for i in 1..pages {
                unsafe { std::ptr::copy_nonoverlapping(y, res.add(nrows * i), nrows) };
                unsafe {
                    dgemv(
                        CHN,
                        rows,
                        cols,
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

        return;
    }

    // case: 3d * 2d
    // 2d matrix is N x pages
    // multiply across the pages axis each 2d matrix times the vector resulting in a 2d matrix of N x pages
    if xdimensions.len() < 3 && ydimensions.len() < 3 && *xdimensions.get(1).unwrap_or(&1) != 1 {
        let ans_matrix: *mut mxArray =
            unsafe { mxCreateDoubleMatrix(nrows, pages, mxComplexity_mxREAL) };
        let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
        unsafe { *plhs.add(0) = ans_matrix };

        // 2d ( N x pages )
        // sum to the result each vector across the pages
        if *ydimensions.get(1).unwrap_or(&1) != 1 {
            unsafe { std::ptr::copy_nonoverlapping(y, res, nrows * pages) };
            unsafe { dgemv(CHN, rows, rows, ONE, A, rows, x, ONEI, ONE, res, ONEI) };

            for i in 1..pages {
                unsafe {
                    dgemv(
                        CHN,
                        rows,
                        cols,
                        ONE,
                        A.add(nrows * ncols * i),
                        rows,
                        x.add(ncols * i),
                        ONEI,
                        ONE,
                        res.add(nrows * i),
                        ONEI,
                    )
                };
            }
        } else {
            // + vec
            // sum to the result the same vector across the pages
            unsafe { std::ptr::copy_nonoverlapping(y, res, nrows) };
            unsafe { dgemv(CHN, rows, rows, ONE, A, rows, x, ONEI, ONE, res, ONEI) };

            for i in 1..pages {
                unsafe { std::ptr::copy_nonoverlapping(y, res.add(nrows * i), nrows) };
                unsafe {
                    dgemv(
                        CHN,
                        rows,
                        cols,
                        ONE,
                        A.add(nrows * ncols * i),
                        rows,
                        x.add(ncols * i),
                        ONEI,
                        ONE,
                        res.add(nrows * i),
                        ONEI,
                    )
                };
            }
        }

        return;
    }

    // case: 3d * 3d
    // multiply across the pages axis each 2d matrices resulting in a 3d matrix of N * N * x pages
    if xdimensions.len() == 3 {
        let mut dims: [usize; 3] = [0, 0, 0];
        dims[0] = nrows;
        dims[1] = *xdimensions.get(1).unwrap_or(&1);
        dims[2] = pages;
        let ans_matrix: *mut mxArray = unsafe {
            mxCreateNumericArray(
                3,
                dims.as_ptr(),
                mxClassID_mxDOUBLE_CLASS,
                mxComplexity_mxREAL,
            )
        };
        let res = unsafe { ans_matrix.as_mut().unwrap().get_ptr() };
        unsafe { *plhs.add(0) = ans_matrix };

        match ydimensions.len() {
            2 => {
                let colres = xdimensions.get(1).unwrap_or(&1);
                let ncolres = *colres;
                if *colres != *ydimensions.get(1).unwrap_or(&1) {
                    unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size 2.\n\0".as_ptr()) };
                }

                unsafe { std::ptr::copy_nonoverlapping(y, res, nrows * ncolres) };
                unsafe {
                    dgemm(
                        CHN, CHN, rows, colres, cols, ONE, A, rows, x, cols, ONE, res, rows,
                    )
                };

                for i in 1..pages {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            y,
                            res.add(nrows * colres * i),
                            nrows * colres,
                        )
                    };
                    unsafe {
                        dgemm(
                            CHN,
                            CHN,
                            rows,
                            colres,
                            cols,
                            ONE,
                            A.add(nrows * ncols * i),
                            rows,
                            x.add(ncols * ncolres * i),
                            cols,
                            ONE,
                            res.add(nrows * ncols * i),
                            rows,
                        )
                    };
                }
            }
            3 => {
                let colres = xdimensions.get(1).unwrap_or(&1);
                let ncolres = *colres;
                if ncolres != *ydimensions.get(1).unwrap_or(&1) {
                    unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size 3.\n\0".as_ptr()) };
                }

                unsafe { std::ptr::copy_nonoverlapping(y, res, nrows * ncolres) };
                unsafe {
                    dgemm(
                        CHN, CHN, rows, colres, cols, ONE, A, rows, x, cols, ONE, res, rows,
                    )
                };

                for i in 1..pages {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            y.add(nrows * colres * i),
                            res.add(nrows * colres * i),
                            nrows * colres,
                        )
                    };
                    unsafe {
                        dgemm(
                            CHN,
                            CHN,
                            rows,
                            colres,
                            cols,
                            ONE,
                            A.add(nrows * ncols * i),
                            rows,
                            x.add(ncols * ncolres * i),
                            cols,
                            ONE,
                            res.add(nrows * ncols * i),
                            rows,
                        )
                    };
                }
            }
            1 => {
                if !ymx.is_scalar() || ymx.get_scalar() != 0f64 {
                    unsafe {
                        mexPrintf(
                            "xdism: %d - %d - %d\n\0".as_ptr(),
                            ydimensions[0],
                            ydimensions[1],
                            *ydimensions.get(2).unwrap_or(&1),
                        )
                    };
                    unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size 1.\n\0".as_ptr()) };
                }
            }
            _ => {
                unsafe {
                    mexPrintf(
                        "xdism: %d - %d - %d\n\0".as_ptr(),
                        ydimensions[0],
                        ydimensions[1],
                        *ydimensions.get(2).unwrap_or(&1),
                    )
                };
                unsafe { mexErrMsgTxt("gemv3d: invalid 3rd term size (other).\n\0".as_ptr()) };
            }
        }

        return;
    }

    unsafe {
        mexErrMsgTxt("gemv3d: can not reach this point in code.\n\0".as_ptr());
    }
}
