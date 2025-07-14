mod vec2;
mod vec3;

#[cfg(feature = "f64")]
pub use vec2::{DVec2x2, DVec2x4};
#[cfg(feature = "f32")]
pub use vec2::{Vec2x4, Vec2x8};
#[cfg(feature = "f64")]
pub use vec3::{DVec3x2, DVec3x4};
#[cfg(feature = "f32")]
pub use vec3::{Vec3x4, Vec3x8};
