#![cfg(feature = "f64")]
mod drot2;
mod rot2_wide;

#[cfg(feature = "f64")]
pub use drot2::DRot2;
#[cfg(feature = "f64")]
pub use rot2_wide::{DRot2x2, DRot2x4};
pub use rot2_wide::{Rot2x4, Rot2x8};
