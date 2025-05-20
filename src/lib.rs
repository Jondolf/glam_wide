//! Wide SIMD types for `glam` and `bevy_math`.

#![warn(missing_docs)]

mod quat;
mod rot2;
mod vec;

pub use quat::*;
pub use rot2::*;
pub use vec::*;

pub use wide::*;

pub(crate) trait SimdLaneCount {
    /// The number of lanes in the SIMD type.
    const LANES: usize;
}

macro_rules! impl_simd_lane_count {
    ($(($ty:ty, $lanes:expr),)+) => {
        $(impl SimdLaneCount for $ty {
            const LANES: usize = $lanes;
        })+
    };
}

// Number types
impl_simd_lane_count! {
    (f32, 1),
    (f64, 1),
    (i8, 1),
    (i16, 1),
    (i32, 1),
    (i64, 1),
    (u8, 1),
    (u16, 1),
    (u32, 1),
    (u64, 1),
    (f32x4, 4),
    (f32x8, 8),
    (f64x2, 2),
    (f64x4, 4),
    (i8x16, 16),
    (i8x32, 32),
    (i16x8, 8),
    (i16x16, 16),
    (i32x4, 4),
    (i32x8, 8),
    (i64x2, 2),
    (i64x4, 4),
    (u8x16, 16),
    (u16x8, 8),
    (u16x16, 16),
    (u32x4, 4),
    (u32x8, 8),
    (u64x2, 2),
    (u64x4, 4),
}

// Vector types
impl_simd_lane_count! {
    (Vec2x4, 4),
    (Vec2x8, 8),
    (DVec2x2, 2),
    (DVec2x4, 4),
    (Vec3x4, 4),
    (Vec3x8, 8),
    (DVec3x2, 2),
    (DVec3x4, 4),
}

// 2D rotation types
impl_simd_lane_count! {
    (Rot2x4, 4),
    (Rot2x8, 8),
    (DRot2x2, 2),
    (DRot2x4, 4),
}

// Quaternion types
impl_simd_lane_count! {
    (Quatx4, 4),
    (Quatx8, 8),
    (DQuatx2, 2),
    (DQuatx4, 4),
}
