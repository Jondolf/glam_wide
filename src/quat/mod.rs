mod quat;

#[cfg(feature = "f64")]
pub use quat::{DQuatx2, DQuatx4};
#[cfg(feature = "f32")]
pub use quat::{Quatx4, Quatx8};
