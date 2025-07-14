#[cfg(feature = "f64")]
mod drot2_scalar;
mod rot2;

#[cfg(feature = "f64")]
pub use drot2_scalar::DRot2;
#[cfg(feature = "f64")]
pub use rot2::{DRot2x2, DRot2x4};
#[cfg(feature = "f32")]
pub use rot2::{Rot2x4, Rot2x8};
