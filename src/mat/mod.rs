mod mat2_wide;
mod mat3_wide;
mod symmetric_mat2;
mod symmetric_mat3;

#[cfg(feature = "f64")]
pub use mat2_wide::{DMat2x2, DMat2x4};
pub use mat2_wide::{Mat2x4, Mat2x8};
#[cfg(feature = "f64")]
pub use mat3_wide::{DMat3x2, DMat3x4};
pub use mat3_wide::{Mat3x4, Mat3x8};
#[cfg(feature = "f64")]
pub use symmetric_mat2::{DSymmetricMat2, DSymmetricMat2x2, DSymmetricMat2x4};
pub use symmetric_mat2::{SymmetricMat2, SymmetricMat2x4, SymmetricMat2x8};
#[cfg(feature = "f64")]
pub use symmetric_mat3::{DSymmetricMat3, DSymmetricMat3x2, DSymmetricMat3x4};
pub use symmetric_mat3::{SymmetricMat3, SymmetricMat3x4, SymmetricMat3x8};
