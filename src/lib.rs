//! Wide SIMD types for the [`glam`] ecosystem.

#![warn(missing_docs)]

mod bool_wide;
mod bvec;
mod mat;
mod quat;
#[cfg(feature = "bevy_math")]
mod rot2;
mod swizzles;
mod vec;

pub use bool_wide::*;
pub use bvec::*;
pub use mat::*;
pub use quat::*;
#[cfg(feature = "bevy_math")]
pub use rot2::*;
pub use vec::*;

pub use wide::*;

/// An extension trait for wide floating-point SIMD types.
pub trait SimdFloatExt {
    /// Not a Number (NaN).
    const NAN: Self;
}

macro_rules! impl_simd_const_ext {
    ($(($t:ident, $n:ident, $lanes:expr),)+) => {
        $(impl SimdFloatExt for $n {
            const NAN: Self = Self::new([$t::NAN; $lanes]);
        })+
    };
}

impl_simd_const_ext! {
    (f32, f32x4, 4),
    (f32, f32x8, 8),
    (f64, f64x2, 2),
    (f64, f64x4, 4),
}

pub(crate) trait SimdLaneCount {
    /// The number of lanes in the SIMD type.
    const LANES: usize;
}

macro_rules! impl_simd_lane_count_scalar {
    ($($n:ident,)+) => {
        $(impl SimdLaneCount for $n {
            const LANES: usize = 1;
        })+
    };
}

macro_rules! impl_simd_lane_count {
    ($(($n:ident, $lanes:expr),)+) => {
        $(impl SimdLaneCount for $n {
            const LANES: usize = $lanes;
        })+
    };
}

// Scalar types
impl_simd_lane_count_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,);

// Number types
impl_simd_lane_count! {
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

// Mask types
impl_simd_lane_count! {
    (boolf32x4, 4),
    (boolf32x8, 8),
    (boolf64x2, 2),
    (boolf64x4, 4),
}

// Vector types
#[cfg(feature = "f32")]
impl_simd_lane_count! {
    (Vec2x4, 4),
    (Vec2x8, 8),
    (Vec3x4, 4),
    (Vec3x8, 8),
    (BVec2x4, 4),
    (BVec2x8, 8),
    (BVec3x4, 4),
    (BVec3x8, 8),
    (BVec4x4, 4),
    (BVec4x8, 8),
}
#[cfg(feature = "f64")]
impl_simd_lane_count! {
    (DVec2x2, 2),
    (DVec2x4, 4),
    (DVec3x2, 2),
    (DVec3x4, 4),
    (BDVec2x2, 2),
    (BDVec2x4, 4),
    (BDVec3x2, 2),
    (BDVec3x4, 4),
    (BDVec4x2, 2),
    (BDVec4x4, 4),
}

// 2D rotation types
#[cfg(all(feature = "f32", feature = "bevy_math"))]
impl_simd_lane_count! {
    (Rot2x4, 4),
    (Rot2x8, 8),
}
#[cfg(all(feature = "f64", feature = "bevy_math"))]
impl_simd_lane_count! {
    (DRot2x2, 2),
    (DRot2x4, 4),
}

// Quaternion types
#[cfg(feature = "f32")]
impl_simd_lane_count! {
    (Quatx4, 4),
    (Quatx8, 8),
}
#[cfg(feature = "f64")]
impl_simd_lane_count! {
    (DQuatx2, 2),
    (DQuatx4, 4),
}
