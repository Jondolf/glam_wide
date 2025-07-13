mod vec2_wide;
mod vec3_wide;

#[cfg(feature = "f64")]
pub use vec2_wide::{DVec2x2, DVec2x4};
#[cfg(feature = "f32")]
pub use vec2_wide::{Vec2x4, Vec2x8};
#[cfg(feature = "f64")]
pub use vec3_wide::{DVec3x2, DVec3x4};
#[cfg(feature = "f32")]
pub use vec3_wide::{Vec3x4, Vec3x8};
