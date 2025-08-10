/*!
 * Implementations on the raw `mxArray` pointer.
 */
#![allow(bad_style)]
pub type mxComplexity = ::std::os::raw::c_uint;
pub type mxClassID = ::std::os::raw::c_uint;
// NOTE: Bindgen made these signed types, but Matlab's tmwtypes header says in a comment
// they should be unsigned (which makes more sense tbh)
pub type mwSize = usize;

pub const mxComplexity_mxREAL: mxComplexity = 0;
#[allow(unused)]
pub const mxComplexity_mxCOMPLEX: mxComplexity = 1;

pub const mxClassID_mxDOUBLE_CLASS: mxClassID = 6;

/**
 * The main matlab opaque array type, returned and accepted as argument by various mex
 * functions.
 */
#[allow(non_camel_case_types)]
pub type mxArray = mxArray_tag;

// NOTE: Bindgen derived Copy and Clone for this type, while it definitely is not
// straightforwardly Copy or Clone.
#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct mxArray_tag {
    _unused: [u8; 0],
}

use super::mx::{
    mxGetDimensions, mxGetNumberOfDimensions, mxGetNumberOfElements, mxGetPr, mxGetScalar,
    mxIsComplex, mxIsDouble, mxIsSparse,
};

pub type Rhs<'mex, 'matlab> = &'mex [&'matlab mxArray];

impl mxArray {
    /// Return the sizes of the constituent dimensions of the mxArray
    pub fn dimensions(&self) -> &[usize] {
        unsafe { std::slice::from_raw_parts(mxGetDimensions(self), mxGetNumberOfDimensions(self)) }
    }

    /// Return the number of elements contained in this array.
    pub fn numel(&self) -> usize {
        unsafe { mxGetNumberOfElements(self) }
    }

    pub fn get_ptr(&self) -> *mut f64 {
        unsafe { mxGetPr(self) }
    }

    pub fn get_scalar(&self) -> f64 {
        unsafe { mxGetScalar(self) }
    }

    /// Check whether the backing array is complex. Since the only arrays which can
    /// be complex are numeric arrays, this also implies that.
    pub fn is_complex(&self) -> bool {
        unsafe { mxIsComplex(self) }
    }

    pub fn is_double(&self) -> bool {
        unsafe { mxIsDouble(self) }
    }

    /// Check whether the backing array is a sparse matrix
    pub fn is_sparse(&self) -> bool {
        unsafe { mxIsSparse(self) }
    }

    /// Check whether the backing array is empty
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Check whether the backing array only holds one element
    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }
}
