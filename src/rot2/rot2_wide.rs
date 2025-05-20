use bevy_math::Rot2;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DRot2, DVec2x2, DVec2x4};
use crate::{SimdLaneCount, Vec2x4, Vec2x8};

macro_rules! wide_rot2s {
    ($(($nonwiden:ident, $n:ident, $vt:ident) => ($nonwidet:ident, $t:ident)),+) => {
        $(
        /// A wide counterclockwise 2D rotation.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $n {
            /// The cosine of the rotation angle in radians.
            ///
            /// This is the real part of the unit complex number representing the rotation.
            pub cos: $t,
            /// The sine of the rotation angle in radians.
            ///
            /// This is the imaginary part of the unit complex number representing the rotation.
            pub sin: $t,
        }

        impl Default for $n {
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl $n {
            /// No rotation.
            pub const IDENTITY: Self = Self { cos: $t::ONE, sin: $t::ZERO };

            /// A rotation of π radians.
            pub const PI: Self = Self {
                cos: $t::new([-1.0; $t::LANES]),
                sin: $t::ZERO,
            };

            /// A counterclockwise rotation of π/2 radians.
            pub const FRAC_PI_2: Self = Self { cos: $t::ZERO, sin: $t::ONE };

            /// A counterclockwise rotation of π/3 radians.
            pub const FRAC_PI_3: Self = Self {
                cos: $t::HALF,
                sin: $t::new([0.866_025_4; $t::LANES]),
            };

            /// A counterclockwise rotation of π/4 radians.
            pub const FRAC_PI_4: Self = Self {
                cos: $t::FRAC_1_SQRT_2,
                sin: $t::FRAC_1_SQRT_2,
            };

            /// A counterclockwise rotation of π/6 radians.
            pub const FRAC_PI_6: Self = Self {
                cos: $t::new([0.866_025_4; $t::LANES]),
                sin: $t::HALF,
            };

            /// A counterclockwise rotation of π/8 radians.
            pub const FRAC_PI_8: Self = Self {
                cos: $t::new([0.923_879_5; $t::LANES]),
                sin: $t::new([0.382_683_43; $t::LANES]),
            };

            /// Creates a 2D rotation from a counterclockwise angle in radians.
            ///
            /// # Note
            ///
            /// The input rotation will always be clamped to the range `(-π, π]` by design.
            #[inline]
            pub fn radians(radians: $t) -> Self {
                let (sin, cos) = radians.sin_cos();
                Self::from_sin_cos(sin, cos)
            }

            /// Creates a 2D rotation from a counterclockwise angle in degrees.
            ///
            /// # Note
            ///
            /// The input rotation will always be clamped to the range `(-180°, 180°]` by design.
            #[inline]
            pub fn degrees(degrees: $t) -> Self {
                Self::radians(degrees.to_radians())
            }

            /// Creates a 2D rotation from a counterclockwise fraction of a full turn of 360 degrees.
            ///
            /// # Note
            ///
            /// The input rotation will always be clamped to the range `(-50%, 50%]` by design.
            #[inline]
            pub fn turn_fraction(fraction: $t) -> Self {
                Self::radians($t::TAU * fraction)
            }

            /// Creates a 2D rotation from the sine and cosine of an angle in radians.
            ///
            /// The rotation is only valid if `sin * sin + cos * cos == 1.0`.
            #[inline]
            pub fn from_sin_cos(sin: $t, cos: $t) -> Self {
                Self { sin, cos }
            }

            /// Creates a 2D rotation with all lanes set to the same sine and cosine values of an angle in radians.
            ///
            /// The rotation is only valid if `sin * sin + cos * cos == 1.0`.
            #[inline]
            pub fn from_sin_cos_splat(sin: $nonwidet, cos: $nonwidet) -> Self {
                Self {
                    sin: $t::new([sin; $t::LANES]),
                    cos: $t::new([cos; $t::LANES]),
                }
            }

            /// Creates a 2D rotation with all lanes set to `rot`.
            #[inline]
            pub const fn splat(rot: $nonwiden) -> Self {
                Self {
                    sin: $t::new([rot.sin; $t::LANES]),
                    cos: $t::new([rot.cos; $t::LANES]),
                }
            }

            /// Returns the rotation in radians in the `(-pi, pi]` range.
            #[inline]
            pub fn as_radians(self) -> $t {
                self.sin.atan2(self.cos)
            }

            /// Returns the rotation in degrees in the `(-180, 180]` range.
            #[inline]
            pub fn as_degrees(self) -> $t {
                self.as_radians().to_degrees()
            }

            /// Returns the rotation as a fraction of a full 360 degree turn.
            #[inline]
            pub fn as_turn_fraction(self) -> $t {
                self.as_radians() / $t::TAU
            }

            /// Returns the sine and cosine of the rotation angle in radians.
            #[inline]
            pub const fn sin_cos(self) -> ($t, $t) {
                (self.sin, self.cos)
            }

            /// Computes the length or norm of the complex number used to represent the rotation.
            ///
            /// The length is typically expected to be `1.0`. Unexpectedly denormalized rotations
            /// can be a result of incorrect construction or floating point error caused by
            /// successive operations.
            #[inline]
            #[doc(alias = "norm")]
            pub fn length(self) -> $t {
                $vt::new(self.sin, self.cos).length()
            }

            /// Computes the squared length or norm of the complex number used to represent the rotation.
            ///
            /// This is generally faster than [`$n::length()`], as it avoids a square
            /// root operation.
            ///
            /// The length is typically expected to be `1.0`. Unexpectedly denormalized rotations
            /// can be a result of incorrect construction or floating point error caused by
            /// successive operations.
            #[inline]
            #[doc(alias = "norm2")]
            pub fn length_squared(self) -> $t {
                $vt::new(self.sin, self.cos).length_squared()
            }

            /// Computes `1.0 / self.length()`.
            ///
            /// For valid results, `self` must _not_ have a length of zero.
            #[inline]
            pub fn length_recip(self) -> $t {
                $vt::new(self.sin, self.cos).length_recip()
            }

            /// Returns `self` with a length of `1.0`.
            ///
            /// Note that 2D rotation should typically already be normalized by design.
            /// Manual normalization is only needed when successive operations result in
            /// accumulated floating point error, or if the rotation was constructed
            /// with invalid values.
            #[inline]
            pub fn normalize(self) -> Self {
                let length_recip = self.length_recip();
                Self::from_sin_cos(self.sin * length_recip, self.cos * length_recip)
            }

            /// Returns `self` after an approximate normalization, assuming the value is already nearly normalized.
            /// Useful for preventing numerical error accumulation.
            #[inline]
            pub fn fast_renormalize(self) -> Self {
                let length_squared = self.length_squared();
                // Based on a Taylor approximation of the inverse square root
                let length_recip_approx = 0.5 * (3.0 - length_squared);
                $n {
                    sin: self.sin * length_recip_approx,
                    cos: self.cos * length_recip_approx,
                }
            }

            /// Returns the angle in radians needed to make `self` and `other` coincide.
            #[inline]
            pub fn angle_to(self, other: Self) -> $t {
                (other * self.inverse()).as_radians()
            }

            /// Returns the inverse of the rotation. This is also the conjugate
            /// of the unit complex number representing the rotation.
            #[inline]
            #[must_use]
            #[doc(alias = "conjugate")]
            pub fn inverse(self) -> Self {
                Self {
                    cos: self.cos,
                    sin: -self.sin,
                }
            }

            /// Performs a spherical linear interpolation between `self` and `end`
            /// based on the value `s`.
            ///
            /// This corresponds to interpolating between the two angles at a constant angular velocity.
            ///
            /// When `s == 0.0`, the result will be equal to `self`.
            /// When `s == 1.0`, the result will be equal to `rhs`.
            #[inline]
            pub fn slerp(self, end: Self, s: $t) -> Self {
                self * Self::radians(self.angle_to(end) * s)
            }
        }

        impl From<$t> for $n {
            /// Creates a 2D rotation from a counterclockwise angle in radians.
            fn from(rotation: $t) -> Self {
                Self::radians(rotation)
            }
        }

        impl core::ops::Mul for $n {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self {
                    cos: self.cos * rhs.cos - self.sin * rhs.sin,
                    sin: self.sin * rhs.cos + self.cos * rhs.sin,
                }
            }
        }

        impl core::ops::MulAssign for $n {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl core::ops::Mul<$vt> for $n {
            type Output = $vt;

            /// Rotates a [`$vt`] by a 2D rotation.
            fn mul(self, rhs: $vt) -> Self::Output {
                $vt::new(
                    rhs.x * self.cos - rhs.y * self.sin,
                    rhs.x * self.sin + rhs.y * self.cos,
                )
            }
        }
        )+
    }
}

wide_rot2s!(
    (Rot2, Rot2x4, Vec2x4) => (f32, f32x4),
    (Rot2, Rot2x8, Vec2x8) => (f32, f32x8)
);

#[cfg(feature = "f64")]
wide_rot2s!(
    (DRot2, DRot2x2, DVec2x2) => (f64, f64x2),
    (DRot2, DRot2x4, DVec2x4) => (f64, f64x4)
);
