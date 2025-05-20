mod mat2_wide;
mod mat3_wide;

#[cfg(feature = "f64")]
pub use mat2_wide::{DMat2x2, DMat2x4};
pub use mat2_wide::{Mat2x4, Mat2x8};
#[cfg(feature = "f64")]
pub use mat3_wide::{DMat3x2, DMat3x4};
pub use mat3_wide::{Mat3x4, Mat3x8};
