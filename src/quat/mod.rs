mod quat_wide;

#[cfg(feature = "f64")]
pub use quat_wide::{DQuatx2, DQuatx4};
pub use quat_wide::{Quatx4, Quatx8};
