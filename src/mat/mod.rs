mod mat2_wide;
mod mat3_ext;
mod mat3_wide;
mod mat_ext;
mod symmetric_mat2;
mod symmetric_mat3;
mod symmetric_mat6;

pub use mat_ext::MatExt;
#[cfg(feature = "f64")]
pub use mat2_wide::{DMat2x2, DMat2x4};
pub use mat2_wide::{Mat2x4, Mat2x8};
pub use mat3_ext::{DMat3Ext, Mat3AExt, Mat3Ext};
#[cfg(feature = "f64")]
pub use mat3_wide::{DMat3x2, DMat3x4};
pub use mat3_wide::{Mat3x4, Mat3x8};
#[cfg(feature = "f64")]
pub use symmetric_mat2::{DSymmetricMat2, DSymmetricMat2x2, DSymmetricMat2x4};
pub use symmetric_mat2::{SymmetricMat2, SymmetricMat2x4, SymmetricMat2x8};
#[cfg(feature = "f64")]
pub use symmetric_mat3::{DSymmetricMat3, DSymmetricMat3x2, DSymmetricMat3x4};
pub use symmetric_mat3::{SymmetricMat3, SymmetricMat3x4, SymmetricMat3x8};
#[cfg(feature = "f64")]
pub use symmetric_mat6::{DSymmetricMat6, DSymmetricMat6x2, DSymmetricMat6x4};
pub use symmetric_mat6::{SymmetricMat6, SymmetricMat6x4, SymmetricMat6x8};

/// An error that can occur when converting matrices to other representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatConversionError {
    /// Tried to convert a matrix to a symmetric matrix type, but the matrix is not symmetric.
    Asymmetric,
}

impl core::fmt::Display for MatConversionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MatConversionError::Asymmetric => write!(f, "Matrix is not symmetric"),
        }
    }
}
