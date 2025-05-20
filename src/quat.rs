#[cfg(feature = "f64")]
use bevy_math::DQuat;
use bevy_math::Quat;
use core::iter::{Product, Sum};
use core::ops::*;
use wide::{f32x4, f32x8, f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::vec3::{DVec3x2, DVec3x4};
use crate::{Vec3x4, Vec3x8};

macro_rules! quats {
    ($(($nonwiden:ident, $n:ident, $vt:ident) => ($nonwidet:ident, $t:ident)),+) => {
        $(
        /// A wide quaternion representing an orientation.
        ///
        /// This quaternion is intended to be of unit length but may denormalize due to
        /// floating point "error creep" which can occur when successive quaternion
        /// operations are applied.
        #[derive(Clone, Copy, Debug)]
        #[repr(C)]
        pub struct $n {
            /// The X component of the quaternion.
            pub x: $t,
            /// The Y component of the quaternion.
            pub y: $t,
            /// The Z component of the quaternion.
            pub z: $t,
            /// The W component of the quaternion.
            pub w: $t,
        }

        impl $n {
            /// All zeros.
            pub const ZERO: Self = Self::from_xyzw($t::ZERO, $t::ZERO, $t::ZERO, $t::ZERO);

            /// The identity quaternion. Corresponds to no rotation.
            pub const IDENTITY: Self = Self::from_xyzw($t::ZERO, $t::ZERO, $t::ZERO, $t::ONE);

            /// Creates a new rotation quaternion.
            #[inline(always)]
            #[must_use]
            pub const fn from_xyzw(x: $t, y: $t, z: $t, w: $t) -> Self {
                Self { x, y, z, w }
            }

            /// Creates a new quaternion with all lanes set to the same `x`, `y`, and `z` values.
            #[inline(always)]
            #[must_use]
            pub fn from_xyzw_splat(x: $nonwidet, y: $nonwidet, z: $nonwidet, w: $nonwidet) -> Self {
                Self {
                    x: $t::splat(x),
                    y: $t::splat(y),
                    z: $t::splat(z),
                    w: $t::splat(w),
                }
            }

            /// Creates a new quaternion with all lanes set to `q`.
            #[inline(always)]
            #[must_use]
            pub fn splat(q: $nonwiden) -> Self {
                Self {
                    x: $t::splat(q.x),
                    y: $t::splat(q.y),
                    z: $t::splat(q.z),
                    w: $t::splat(q.w),
                }
            }

            /// Creates a new rotation quaternion from an array.
            #[inline(always)]
            #[must_use]
            pub const fn from_array(arr: [$t; 4]) -> Self {
                Self::from_xyzw(arr[0], arr[1], arr[2], arr[3])
            }

            /// Returns the rotation quaternion as an array.
            #[inline(always)]
            #[must_use]
            pub const fn to_array(self) -> [$t; 4] {
                [self.x, self.y, self.z, self.w]
            }

            /// Creates a rotation quaternion from the first 4 values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline(always)]
            #[must_use]
            pub const fn from_slice(slice: &[$t]) -> Self {
                assert!(slice.len() == 4);
                Self::from_xyzw(slice[0], slice[1], slice[2], slice[3])
            }

            /// Writes the elements of `self` to the first 4 elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline(always)]
            pub fn write_to_slice(self, slice: &mut [$t]) {
                slice[..4].copy_from_slice(&self.to_array());
            }

            /// Creates a quaternion for a normalized rotation `axis` and `angle` (in radians).
            ///
            /// The axis must be a unit vector.
            #[inline(always)]
            #[must_use]
            pub fn from_axis_angle(axis: $vt, angle: $t) -> Self {
                let half_angle = angle * $t::splat(0.5);
                let (sin, cos) = half_angle.sin_cos();
                let v = axis * sin;
                Self::from_xyzw(v.x, v.y, v.z, cos)
            }

            /// Creates a quaternion that rotates `v.length()` radians around `v.normalize()`.
            ///
            /// `v` must not be zero.
            #[inline(always)]
            #[must_use]
            pub fn from_scaled_axis(v: $vt) -> Self {
                let angle = v.length();
                Self::from_axis_angle(v/angle, angle)
            }

            /// Creates a quaternion from the `angle` (in radians) around the X axis.
            #[inline(always)]
            #[must_use]
            pub fn from_rotation_x(angle: $t) -> Self {
                let half_angle = angle * $t::splat(0.5);
                let (sin, cos) = half_angle.sin_cos();
                Self::from_xyzw(sin, $t::ZERO, $t::ZERO, cos)
            }

            /// Creates a quaternion from the `angle` (in radians) around the Y axis.
            #[inline(always)]
            #[must_use]
            pub fn from_rotation_y(angle: $t) -> Self {
                let half_angle = angle * $t::splat(0.5);
                let (sin, cos) = half_angle.sin_cos();
                Self::from_xyzw($t::ZERO, sin, $t::ZERO, cos)
            }

            /// Creates a quaternion from the `angle` (in radians) around the Z axis.
            #[inline(always)]
            #[must_use]
            pub fn from_rotation_z(angle: $t) -> Self {
                let half_angle = angle * $t::splat(0.5);
                let (sin, cos) = half_angle.sin_cos();
                Self::from_xyzw($t::ZERO, $t::ZERO, sin, cos)
            }

            /// Returns the quaternion conjugate of `self`. For a unit quaternion
            /// the conjugate is also the inverse.
            #[inline(always)]
            #[must_use]
            pub fn conjugate(self) -> Self {
                Self::from_xyzw(-self.x, -self.y, -self.z, self.w)
            }

            /// Returns the inverse of a normalized quaternion.
            ///
            /// Typically quaternion inverse returns the conjugate of a normalized quaternion.
            /// Because `self` is assumed to already be unit length this method *does not* normalize
            /// before returning the conjugate.
            #[inline(always)]
            #[must_use]
            pub fn inverse(self) -> Self {
                self.conjugate()
            }

            /// Computes the length of `self`.
            #[inline(always)]
            #[must_use]
            pub fn length(self) -> $t {
                self.length_squared().sqrt()
            }

            /// Computes the squared length of `self`.
            ///
            /// This is faster than `length()` as it avoids a square root operation.
            #[inline(always)]
            #[must_use]
            pub fn length_squared(self) -> $t {
                self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
            }

            /// Computes `1.0 / length()`.
            ///
            /// For valid results, `self` must _not_ be of length zero.
            #[inline(always)]
            #[must_use]
            pub fn length_recip(self) -> $t {
                $t::ONE / self.length()
            }

            /// Computes the Euclidean distance between two points in space.
            #[inline(always)]
            #[must_use]
            pub fn distance(self, rhs: Self) -> $t {
                (self - rhs).length()
            }

            /// Compute the squared euclidean distance between two points in space.
            #[inline(always)]
            #[must_use]
            pub fn distance_squared(self, rhs: Self) -> $t {
                (self - rhs).length_squared()
            }

            /// Returns `self` normalized to length 1.0.
            ///
            /// For valid results, `self` must be finite and _not_ of length zero, nor very close to zero.
            #[inline(always)]
            #[must_use]
            pub fn normalize(self) -> Self {
                let length = self.length();
                Self::from_xyzw(
                    self.x / length,
                    self.y / length,
                    self.z / length,
                    self.w / length,
                )
            }

            // TODO: A method for rotating multiple vectors at once, reusing intermediate results.
            /// Multiplies a quaternion and a 3D vector, returning the rotated vector.
            #[inline(always)]
            #[must_use]
            pub fn mul_vec3(self, rhs: $vt) -> $vt {
                // TODO: Can we make this faster?
                let b = $vt::new(self.x, self.y, self.z);
                let b2 = b.length_squared();
                rhs.mul(self.w * self.w - b2)
                    .add(b.mul(rhs.dot(b) * $t::splat(2.0)))
                    .add(b.cross(rhs).mul(self.w * $t::splat(2.0)))
            }

            /// Multiplies two quaternions. If they each represent a rotation, the result will
            /// represent the combined rotation.
            ///
            /// Note that due to floating point rounding the result may not be perfectly normalized.
            #[inline(always)]
            #[must_use]
            pub fn mul_quat(self, rhs: Self) -> Self {
                let (x0, y0, z0, w0) = self.into();
                let (x1, y1, z1, w1) = rhs.into();
                Self::from_xyzw(
                    w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                    w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                    w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
                    w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                )
            }
        }

        impl Default for $n {
            #[inline]
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl From<$n> for [$t; 4] {
            #[inline]
            fn from(v: $n) -> Self {
                [v.x, v.y, v.z, v.w]
            }
        }

        impl From<[$t; 4]> for $n {
            #[inline]
            fn from(comps: [$t; 4]) -> Self {
                Self::from_xyzw(comps[0], comps[1], comps[2], comps[3])
            }
        }

        impl From<&[$t; 4]> for $n {
            #[inline]
            fn from(comps: &[$t; 4]) -> Self {
                Self::from(*comps)
            }
        }

        impl From<&mut [$t; 4]> for $n {
            #[inline]
            fn from(comps: &mut [$t; 4]) -> Self {
                Self::from(*comps)
            }
        }

        impl From<($t, $t, $t, $t)> for $n {
            #[inline]
            fn from(comps: ($t, $t, $t, $t)) -> Self {
                Self::from_xyzw(comps.0, comps.1, comps.2, comps.3)
            }
        }

        impl From<&($t, $t, $t, $t)> for $n {
            #[inline]
            fn from(comps: &($t, $t, $t, $t)) -> Self {
                Self::from(*comps)
            }
        }

        impl From<$n> for ($t, $t, $t, $t) {
            #[inline]
            fn from(v: $n) -> Self {
                (v.x, v.y, v.z, v.w)
            }
        }

        impl Add for $n {
            type Output = Self;

            /// Adds two quaternions.
            ///
            /// The sum is not guaranteed to be normalized.
            ///
            /// Note that addition is not the same as combining the rotations represented by the
            /// two quaternions! That corresponds to multiplication.
            #[inline]
            fn add(self, rhs: $n) -> Self {
                $n::from_xyzw(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
            }
        }

        impl Sub for $n {
            type Output = Self;

            /// Subtracts the `rhs` quaternion from `self`.
            ///
            /// The difference is not guaranteed to be normalized.
            ///
            /// Note that subtraction is not the same as combining the rotations represented by the
            /// two quaternions! That corresponds to multiplication.
            #[inline]
            fn sub(self, rhs: $n) -> Self {
                $n::from_xyzw(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
            }
        }

        impl Mul<$t> for $n {
            type Output = $n;

            /// Multiplies a quaternion by a scalar value.
            ///
            /// The product is not guaranteed to be normalized.
            #[inline]
            fn mul(self, rhs: $t) -> $n {
                $n::from_xyzw(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
            }
        }

        impl Div<$t> for $n {
            type Output = $n;

            /// Divides a quaternion by a scalar value.
            /// The quotient is not guaranteed to be normalized.
            #[inline]
            fn div(self, rhs: $t) -> $n {
                $n::from_xyzw(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
            }
        }

        impl Mul for $n {
            type Output = Self;

            /// Multiplies two quaternions. If they each represent a rotation, the result will
            /// represent the combined rotation.
            ///
            /// Note that due to floating point rounding the result may not be perfectly
            /// normalized.
            #[inline]
            fn mul(self, rhs: $n) -> Self {
                self.mul_quat(rhs)
            }
        }

        impl MulAssign for $n {
            /// Multiplies two quaternions. If they each represent a rotation, the result will
            /// represent the combined rotation.
            ///
            /// Note that due to floating point rounding the result may not be perfectly
            /// normalized.
            #[inline]
            fn mul_assign(&mut self, rhs: $n) {
                *self = self.mul_quat(rhs);
            }
        }

        impl Mul<$vt> for $n {
            type Output = $vt;

            /// Multiplies a quaternion and a 3D vector, returning the rotated vector.
            #[inline]
            fn mul(self, rhs: $vt) -> $vt {
                self.mul_vec3(rhs)
            }
        }

        impl Neg for $n {
            type Output = $n;
            #[inline]
            fn neg(self) -> $n {
                self * $t::splat(-1.0)
            }
        }

        impl Sum for $n {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, Self::add)
            }
        }

        impl<'a> Sum<&'a $n> for $n {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
            }
        }

        impl Product for $n {
            fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::IDENTITY, Self::mul)
            }
        }

        impl<'a> Product<&'a $n> for $n {
            fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self::IDENTITY, |a, &b| Self::mul(a, b))
            }
        }

        impl Index<usize> for $n {
            type Output = $t;

            fn index(&self, index: usize) -> &Self::Output {
                match index {
                    0 => &self.x,
                    1 => &self.y,
                    2 => &self.z,
                    3 => &self.w,
                    i => panic!("Invalid index {i} for vector of type: {}", std::any::type_name::<$n>()),
                }
            }
        }

        impl IndexMut<usize> for $n {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                match index {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    2 => &mut self.z,
                    3 => &mut self.w,
                    i => panic!("Invalid index {i} for vector of type: {}", std::any::type_name::<$n>()),
                }
            }
        }

        impl From<$nonwiden> for $n {
            #[inline]
            fn from(vec: $nonwiden) -> Self {
                Self::splat(vec)
            }
        }
        )+
    };
}

impl From<Quatx4> for [Quat; 4] {
    #[inline]
    fn from(v: Quatx4) -> Self {
        let xs: [f32; 4] = v.x.into();
        let ys: [f32; 4] = v.y.into();
        let zs: [f32; 4] = v.z.into();
        let ws: [f32; 4] = v.w.into();
        [
            Quat::from_xyzw(xs[0], ys[0], zs[0], ws[0]),
            Quat::from_xyzw(xs[1], ys[1], zs[1], ws[1]),
            Quat::from_xyzw(xs[2], ys[2], zs[2], ws[2]),
            Quat::from_xyzw(xs[3], ys[3], zs[3], ws[3]),
        ]
    }
}

impl From<[Quat; 4]> for Quatx4 {
    #[inline]
    fn from(vecs: [Quat; 4]) -> Self {
        Self {
            x: f32x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f32x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
            z: f32x4::from([vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z]),
            w: f32x4::from([vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w]),
        }
    }
}

impl From<Quatx8> for [Quat; 8] {
    #[inline]
    fn from(v: Quatx8) -> Self {
        let xs: [f32; 8] = v.x.into();
        let ys: [f32; 8] = v.y.into();
        let zs: [f32; 8] = v.z.into();
        let ws: [f32; 8] = v.w.into();
        [
            Quat::from_xyzw(xs[0], ys[0], zs[0], ws[0]),
            Quat::from_xyzw(xs[1], ys[1], zs[1], ws[1]),
            Quat::from_xyzw(xs[2], ys[2], zs[2], ws[2]),
            Quat::from_xyzw(xs[3], ys[3], zs[3], ws[3]),
            Quat::from_xyzw(xs[4], ys[4], zs[4], ws[4]),
            Quat::from_xyzw(xs[5], ys[5], zs[5], ws[5]),
            Quat::from_xyzw(xs[6], ys[6], zs[6], ws[6]),
            Quat::from_xyzw(xs[7], ys[7], zs[7], ws[7]),
        ]
    }
}

impl From<[Quat; 8]> for Quatx8 {
    #[inline]
    fn from(vecs: [Quat; 8]) -> Self {
        Self {
            x: f32x8::from([
                vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x, vecs[4].x, vecs[5].x, vecs[6].x,
                vecs[7].x,
            ]),
            y: f32x8::from([
                vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y, vecs[4].y, vecs[5].y, vecs[6].y,
                vecs[7].y,
            ]),
            z: f32x8::from([
                vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z, vecs[4].z, vecs[5].z, vecs[6].z,
                vecs[7].z,
            ]),
            w: f32x8::from([
                vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w, vecs[4].w, vecs[5].w, vecs[6].w,
                vecs[7].w,
            ]),
        }
    }
}

#[cfg(feature = "f64")]
impl From<DQuatx2> for [DQuat; 2] {
    #[inline]
    fn from(v: DQuatx2) -> Self {
        let xs: [f64; 2] = v.x.into();
        let ys: [f64; 2] = v.y.into();
        let zs: [f64; 2] = v.z.into();
        let ws: [f64; 2] = v.w.into();
        [
            DQuat::from_xyzw(xs[0], ys[0], zs[0], ws[0]),
            DQuat::from_xyzw(xs[1], ys[1], zs[1], ws[1]),
        ]
    }
}

#[cfg(feature = "f64")]
impl From<[DQuat; 2]> for DQuatx2 {
    #[inline]
    fn from(vecs: [DQuat; 2]) -> Self {
        Self {
            x: f64x2::from([vecs[0].x, vecs[1].x]),
            y: f64x2::from([vecs[0].y, vecs[1].y]),
            z: f64x2::from([vecs[0].z, vecs[1].z]),
            w: f64x2::from([vecs[0].w, vecs[1].w]),
        }
    }
}

#[cfg(feature = "f64")]
impl From<DQuatx4> for [DQuat; 4] {
    #[inline]
    fn from(v: DQuatx4) -> Self {
        let xs: [f64; 4] = v.x.into();
        let ys: [f64; 4] = v.y.into();
        let zs: [f64; 4] = v.z.into();
        let ws: [f64; 4] = v.w.into();
        [
            DQuat::from_xyzw(xs[0], ys[0], zs[0], ws[0]),
            DQuat::from_xyzw(xs[1], ys[1], zs[1], ws[1]),
            DQuat::from_xyzw(xs[2], ys[2], zs[2], ws[2]),
            DQuat::from_xyzw(xs[3], ys[3], zs[3], ws[3]),
        ]
    }
}

#[cfg(feature = "f64")]
impl From<[DQuat; 4]> for DQuatx4 {
    #[inline]
    fn from(vecs: [DQuat; 4]) -> Self {
        Self {
            x: f64x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f64x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
            z: f64x4::from([vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z]),
            w: f64x4::from([vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w]),
        }
    }
}

quats!(
    (Quat, Quatx4, Vec3x4) => (f32, f32x4),
    (Quat, Quatx8, Vec3x8) => (f32, f32x8)
);

#[cfg(feature = "f64")]
quats!(
    (DQuat, DQuatx2, DVec3x2) => (f64, f64x2),
    (DQuat, DQuatx4, DVec3x4) => (f64, f64x4)
);
