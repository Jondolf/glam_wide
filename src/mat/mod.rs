mod mat2;
#[cfg(feature = "glam_matrix_extensions")]
mod mat23;
mod mat3;
#[cfg(feature = "glam_matrix_extensions")]
mod mat32;
#[cfg(feature = "glam_matrix_extensions")]
mod symmetric_mat2;
#[cfg(feature = "glam_matrix_extensions")]
mod symmetric_mat3;
#[cfg(feature = "glam_matrix_extensions")]
mod symmetric_mat6;

#[cfg(feature = "f64")]
pub use mat2::{DMat2x2, DMat2x4};
#[cfg(feature = "f32")]
pub use mat2::{Mat2x4, Mat2x8};
#[cfg(feature = "f64")]
pub use mat3::{DMat3x2, DMat3x4};
#[cfg(feature = "f32")]
pub use mat3::{Mat3x4, Mat3x8};
#[cfg(all(feature = "f64", feature = "glam_matrix_extensions"))]
pub use mat23::{DMat23x2, DMat23x4};
#[cfg(all(feature = "f32", feature = "glam_matrix_extensions"))]
pub use mat23::{Mat23x4, Mat23x8};
#[cfg(all(feature = "f64", feature = "glam_matrix_extensions"))]
pub use mat32::{DMat32x2, DMat32x4};
#[cfg(all(feature = "f32", feature = "glam_matrix_extensions"))]
pub use mat32::{Mat32x4, Mat32x8};
#[cfg(all(feature = "f64", feature = "glam_matrix_extensions"))]
pub use symmetric_mat2::{DSymmetricMat2x2, DSymmetricMat2x4};
#[cfg(all(feature = "f32", feature = "glam_matrix_extensions"))]
pub use symmetric_mat2::{SymmetricMat2x4, SymmetricMat2x8};
#[cfg(all(feature = "f64", feature = "glam_matrix_extensions"))]
pub use symmetric_mat3::{DSymmetricMat3x2, DSymmetricMat3x4};
#[cfg(all(feature = "f32", feature = "glam_matrix_extensions"))]
pub use symmetric_mat3::{SymmetricMat3x4, SymmetricMat3x8};
#[cfg(all(feature = "f64", feature = "glam_matrix_extensions"))]
pub use symmetric_mat6::{DSymmetricMat6x2, DSymmetricMat6x4};
#[cfg(all(feature = "f32", feature = "glam_matrix_extensions"))]
pub use symmetric_mat6::{SymmetricMat6x4, SymmetricMat6x8};

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
