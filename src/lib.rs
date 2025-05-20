//! Wide SIMD types for `glam` and `bevy_math`.

#![warn(missing_docs)]

mod quat;
mod vec2;
mod vec3;

#[cfg(feature = "f64")]
pub use quat::{DQuatx2, DQuatx4};
pub use quat::{Quatx4, Quatx8};
#[cfg(feature = "f64")]
pub use vec2::{DVec2x2, DVec2x4};
pub use vec2::{Vec2x4, Vec2x8};
#[cfg(feature = "f64")]
pub use vec3::{DVec3x2, DVec3x4};
pub use vec3::{Vec3x4, Vec3x8};

pub use wide::*;
