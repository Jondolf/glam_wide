// This is just a direct copy of `Rot2`, but with `f64` types.

use core::f64::consts::TAU;

use bevy_math::{DMat2, DVec2, FloatExt};

#[cfg(feature = "bevy_reflect")]
use bevy_reflect::{Reflect, std_traits::ReflectDefault};
#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

// TODO: Optionally use libm ops for cross-platform determinism and `no_std` compatibility.
/// A counterclockwise 2D rotation.
///
/// # Example
///
/// ```
/// # use approx::assert_relative_eq;
/// # use bevy_math::{DRot2, DVec2};
/// use std::f64::consts::PI;
///
/// // Create rotations from radians or degrees
/// let rotation1 = DRot2::radians(PI / 2.0);
/// let rotation2 = DRot2::degrees(45.0);
///
/// // Get the angle back as radians or degrees
/// assert_eq!(rotation1.as_degrees(), 90.0);
/// assert_eq!(rotation2.as_radians(), PI / 4.0);
///
/// // "Add" rotations together using `*`
/// #[cfg(feature = "approx")]
/// assert_relative_eq!(rotation1 * rotation2, DRot2::degrees(135.0));
///
/// // Rotate vectors
/// #[cfg(feature = "approx")]
/// assert_relative_eq!(rotation1 * DVec2::X, DVec2::Y);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Default, Clone)
)]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
#[doc(alias = "rotation", alias = "rotation2d", alias = "rotation_2d")]
pub struct DRot2 {
    /// The cosine of the rotation angle in radians.
    ///
    /// This is the real part of the unit complex number representing the rotation.
    pub cos: f64,
    /// The sine of the rotation angle in radians.
    ///
    /// This is the imaginary part of the unit complex number representing the rotation.
    pub sin: f64,
}

impl Default for DRot2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl DRot2 {
    /// No rotation.
    pub const IDENTITY: Self = Self { cos: 1.0, sin: 0.0 };

    /// A rotation of π radians.
    pub const PI: Self = Self {
        cos: -1.0,
        sin: 0.0,
    };

    /// A counterclockwise rotation of π/2 radians.
    pub const FRAC_PI_2: Self = Self { cos: 0.0, sin: 1.0 };

    /// A counterclockwise rotation of π/3 radians.
    pub const FRAC_PI_3: Self = Self {
        cos: 0.5,
        sin: 0.866_025_4,
    };

    /// A counterclockwise rotation of π/4 radians.
    pub const FRAC_PI_4: Self = Self {
        cos: core::f64::consts::FRAC_1_SQRT_2,
        sin: core::f64::consts::FRAC_1_SQRT_2,
    };

    /// A counterclockwise rotation of π/6 radians.
    pub const FRAC_PI_6: Self = Self {
        cos: 0.866_025_4,
        sin: 0.5,
    };

    /// A counterclockwise rotation of π/8 radians.
    pub const FRAC_PI_8: Self = Self {
        cos: 0.923_879_5,
        sin: 0.382_683_43,
    };

    /// Creates a [`DRot2`] from a counterclockwise angle in radians.
    ///
    /// # Note
    ///
    /// The input rotation will always be clamped to the range `(-π, π]` by design.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::DRot2;
    /// # use approx::assert_relative_eq;
    /// # use std::f64::consts::{FRAC_PI_2, PI};
    ///
    /// let rot1 = DRot2::radians(3.0 * FRAC_PI_2);
    /// let rot2 = DRot2::radians(-FRAC_PI_2);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(rot1, rot2);
    ///
    /// let rot3 = DRot2::radians(PI);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(rot1 * rot1, rot3);
    /// ```
    #[inline]
    pub fn radians(radians: f64) -> Self {
        let (sin, cos) = radians.sin_cos();
        Self::from_sin_cos(sin, cos)
    }

    /// Creates a [`DRot2`] from a counterclockwise angle in degrees.
    ///
    /// # Note
    ///
    /// The input rotation will always be clamped to the range `(-180°, 180°]` by design.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::DRot2;
    /// # use approx::assert_relative_eq;
    ///
    /// let rot1 = DRot2::degrees(270.0);
    /// let rot2 = DRot2::degrees(-90.0);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(rot1, rot2);
    ///
    /// let rot3 = DRot2::degrees(180.0);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(rot1 * rot1, rot3);
    /// ```
    #[inline]
    pub fn degrees(degrees: f64) -> Self {
        Self::radians(degrees.to_radians())
    }

    /// Creates a [`DRot2`] from a counterclockwise fraction of a full turn of 360 degrees.
    ///
    /// # Note
    ///
    /// The input rotation will always be clamped to the range `(-50%, 50%]` by design.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::DRot2;
    /// # use approx::assert_relative_eq;
    ///
    /// let rot1 = DRot2::turn_fraction(0.75);
    /// let rot2 = DRot2::turn_fraction(-0.25);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(rot1, rot2);
    ///
    /// let rot3 = DRot2::turn_fraction(0.5);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(rot1 * rot1, rot3);
    /// ```
    #[inline]
    pub fn turn_fraction(fraction: f64) -> Self {
        Self::radians(TAU * fraction)
    }

    /// Creates a [`DRot2`] from the sine and cosine of an angle in radians.
    ///
    /// The rotation is only valid if `sin * sin + cos * cos == 1.0`.
    ///
    /// # Panics
    ///
    /// Panics if `sin * sin + cos * cos != 1.0` when the `glam_assert` feature is enabled.
    #[inline]
    pub fn from_sin_cos(sin: f64, cos: f64) -> Self {
        let rotation = Self { sin, cos };
        debug_assert!(
            rotation.is_normalized(),
            "the given sine and cosine produce an invalid rotation"
        );
        rotation
    }

    /// Returns the rotation in radians in the `(-pi, pi]` range.
    #[inline]
    pub fn as_radians(self) -> f64 {
        self.sin.atan2(self.cos)
    }

    /// Returns the rotation in degrees in the `(-180, 180]` range.
    #[inline]
    pub fn as_degrees(self) -> f64 {
        self.as_radians().to_degrees()
    }

    /// Returns the rotation as a fraction of a full 360 degree turn.
    #[inline]
    pub fn as_turn_fraction(self) -> f64 {
        self.as_radians() / TAU
    }

    /// Returns the sine and cosine of the rotation angle in radians.
    #[inline]
    pub const fn sin_cos(self) -> (f64, f64) {
        (self.sin, self.cos)
    }

    /// Computes the length or norm of the complex number used to represent the rotation.
    ///
    /// The length is typically expected to be `1.0`. Unexpectedly denormalized rotations
    /// can be a result of incorrect construction or floating point error caused by
    /// successive operations.
    #[inline]
    #[doc(alias = "norm")]
    pub fn length(self) -> f64 {
        DVec2::new(self.sin, self.cos).length()
    }

    /// Computes the squared length or norm of the complex number used to represent the rotation.
    ///
    /// This is generally faster than [`DRot2::length()`], as it avoids a square
    /// root operation.
    ///
    /// The length is typically expected to be `1.0`. Unexpectedly denormalized rotations
    /// can be a result of incorrect construction or floating point error caused by
    /// successive operations.
    #[inline]
    #[doc(alias = "norm2")]
    pub fn length_squared(self) -> f64 {
        DVec2::new(self.sin, self.cos).length_squared()
    }

    /// Computes `1.0 / self.length()`.
    ///
    /// For valid results, `self` must _not_ have a length of zero.
    #[inline]
    pub fn length_recip(self) -> f64 {
        DVec2::new(self.sin, self.cos).length_recip()
    }

    /// Returns `self` with a length of `1.0` if possible, and `None` otherwise.
    ///
    /// `None` will be returned if the sine and cosine of `self` are both zero (or very close to zero),
    /// or if either of them is NaN or infinite.
    ///
    /// Note that [`DRot2`] should typically already be normalized by design.
    /// Manual normalization is only needed when successive operations result in
    /// accumulated floating point error, or if the rotation was constructed
    /// with invalid values.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let recip = self.length_recip();
        if recip.is_finite() && recip > 0.0 {
            Some(Self::from_sin_cos(self.sin * recip, self.cos * recip))
        } else {
            None
        }
    }

    /// Returns `self` with a length of `1.0`.
    ///
    /// Note that [`DRot2`] should typically already be normalized by design.
    /// Manual normalization is only needed when successive operations result in
    /// accumulated floating point error, or if the rotation was constructed
    /// with invalid values.
    ///
    /// # Panics
    ///
    /// Panics if `self` has a length of zero, NaN, or infinity when debug assertions are enabled.
    #[inline]
    pub fn normalize(self) -> Self {
        let length_recip = self.length_recip();
        Self::from_sin_cos(self.sin * length_recip, self.cos * length_recip)
    }

    /// Returns `self` after an approximate normalization, assuming the value is already nearly normalized.
    /// Useful for preventing numerical error accumulation.
    /// See [`Dir3::fast_renormalize`](crate::Dir3::fast_renormalize) for an example of when such error accumulation might occur.
    #[inline]
    pub fn fast_renormalize(self) -> Self {
        let length_squared = self.length_squared();
        // Based on a Taylor approximation of the inverse square root, see [`Dir3::fast_renormalize`](crate::Dir3::fast_renormalize) for more details.
        let length_recip_approx = 0.5 * (3.0 - length_squared);
        DRot2 {
            sin: self.sin * length_recip_approx,
            cos: self.cos * length_recip_approx,
        }
    }

    /// Returns `true` if the rotation is neither infinite nor NaN.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.sin.is_finite() && self.cos.is_finite()
    }

    /// Returns `true` if the rotation is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.sin.is_nan() || self.cos.is_nan()
    }

    /// Returns whether `self` has a length of `1.0` or not.
    ///
    /// Uses a precision threshold of approximately `1e-4`.
    #[inline]
    pub fn is_normalized(self) -> bool {
        // The allowed length is 1 +/- 1e-4, so the largest allowed
        // squared length is (1 + 1e-4)^2 = 1.00020001, which makes
        // the threshold for the squared length approximately 2e-4.
        (self.length_squared() - 1.0).abs() <= 2e-4
    }

    /// Returns `true` if the rotation is near [`DRot2::IDENTITY`].
    #[inline]
    pub fn is_near_identity(self) -> bool {
        // Same as `Quat::is_near_identity`, but using sine and cosine
        let threshold_angle_sin = 0.000_049_692_047; // let threshold_angle = 0.002_847_144_6;
        self.cos > 0.0 && self.sin.abs() < threshold_angle_sin
    }

    /// Returns the angle in radians needed to make `self` and `other` coincide.
    #[inline]
    pub fn angle_to(self, other: Self) -> f64 {
        (other * self.inverse()).as_radians()
    }

    /// Returns the inverse of the rotation. This is also the conjugate
    /// of the unit complex number representing the rotation.
    #[inline]
    #[must_use]
    #[doc(alias = "conjugate")]
    pub const fn inverse(self) -> Self {
        Self {
            cos: self.cos,
            sin: -self.sin,
        }
    }

    /// Performs a linear interpolation between `self` and `rhs` based on
    /// the value `s`, and normalizes the rotation afterwards.
    ///
    /// When `s == 0.0`, the result will be equal to `self`.
    /// When `s == 1.0`, the result will be equal to `rhs`.
    ///
    /// This is slightly more efficient than [`slerp`](Self::slerp), and produces a similar result
    /// when the difference between the two rotations is small. At larger differences,
    /// the result resembles a kind of ease-in-out effect.
    ///
    /// If you would like the angular velocity to remain constant, consider using [`slerp`](Self::slerp) instead.
    ///
    /// # Details
    ///
    /// `nlerp` corresponds to computing an angle for a point at position `s` on a line drawn
    /// between the endpoints of the arc formed by `self` and `rhs` on a unit circle,
    /// and normalizing the result afterwards.
    ///
    /// Note that if the angles are opposite like 0 and π, the line will pass through the origin,
    /// and the resulting angle will always be either `self` or `rhs` depending on `s`.
    /// If `s` happens to be `0.5` in this case, a valid rotation cannot be computed, and `self`
    /// will be returned as a fallback.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::DRot2;
    /// #
    /// let rot1 = DRot2::IDENTITY;
    /// let rot2 = DRot2::degrees(135.0);
    ///
    /// let result1 = rot1.nlerp(rot2, 1.0 / 3.0);
    /// assert_eq!(result1.as_degrees(), 28.675055);
    ///
    /// let result2 = rot1.nlerp(rot2, 0.5);
    /// assert_eq!(result2.as_degrees(), 67.5);
    /// ```
    #[inline]
    pub fn nlerp(self, end: Self, s: f64) -> Self {
        Self {
            sin: self.sin.lerp(end.sin, s),
            cos: self.cos.lerp(end.cos, s),
        }
        .try_normalize()
        // Fall back to the start rotation.
        // This can happen when `self` and `end` are opposite angles and `s == 0.5`,
        // because the resulting rotation would be zero, which cannot be normalized.
        .unwrap_or(self)
    }

    /// Performs a spherical linear interpolation between `self` and `end`
    /// based on the value `s`.
    ///
    /// This corresponds to interpolating between the two angles at a constant angular velocity.
    ///
    /// When `s == 0.0`, the result will be equal to `self`.
    /// When `s == 1.0`, the result will be equal to `rhs`.
    ///
    /// If you would like the rotation to have a kind of ease-in-out effect, consider
    /// using the slightly more efficient [`nlerp`](Self::nlerp) instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::DRot2;
    /// #
    /// let rot1 = DRot2::IDENTITY;
    /// let rot2 = DRot2::degrees(135.0);
    ///
    /// let result1 = rot1.slerp(rot2, 1.0 / 3.0);
    /// assert_eq!(result1.as_degrees(), 45.0);
    ///
    /// let result2 = rot1.slerp(rot2, 0.5);
    /// assert_eq!(result2.as_degrees(), 67.5);
    /// ```
    #[inline]
    pub fn slerp(self, end: Self, s: f64) -> Self {
        self * Self::radians(self.angle_to(end) * s)
    }
}

impl From<f64> for DRot2 {
    /// Creates a [`DRot2`] from a counterclockwise angle in radians.
    fn from(rotation: f64) -> Self {
        Self::radians(rotation)
    }
}

impl From<DRot2> for DMat2 {
    /// Creates a [`DMat2`] rotation matrix from a [`DRot2`].
    fn from(rot: DRot2) -> Self {
        DMat2::from_cols_array(&[rot.cos, -rot.sin, rot.sin, rot.cos])
    }
}

impl core::ops::Mul for DRot2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            cos: self.cos * rhs.cos - self.sin * rhs.sin,
            sin: self.sin * rhs.cos + self.cos * rhs.sin,
        }
    }
}

impl core::ops::MulAssign for DRot2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::Mul<DVec2> for DRot2 {
    type Output = DVec2;

    /// Rotates a [`DVec2`] by a [`DRot2`].
    fn mul(self, rhs: DVec2) -> Self::Output {
        DVec2::new(
            rhs.x * self.cos - rhs.y * self.sin,
            rhs.x * self.sin + rhs.y * self.cos,
        )
    }
}

#[cfg(any(feature = "approx", test))]
impl approx::AbsDiffEq for DRot2 {
    type Epsilon = f64;
    fn default_epsilon() -> f64 {
        f64::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        self.cos.abs_diff_eq(&other.cos, epsilon) && self.sin.abs_diff_eq(&other.sin, epsilon)
    }
}

#[cfg(any(feature = "approx", test))]
impl approx::RelativeEq for DRot2 {
    fn default_max_relative() -> f64 {
        f64::EPSILON
    }
    fn relative_eq(&self, other: &Self, epsilon: f64, max_relative: f64) -> bool {
        self.cos.relative_eq(&other.cos, epsilon, max_relative)
            && self.sin.relative_eq(&other.sin, epsilon, max_relative)
    }
}

#[cfg(any(feature = "approx", test))]
impl approx::UlpsEq for DRot2 {
    fn default_max_ulps() -> u32 {
        4
    }
    fn ulps_eq(&self, other: &Self, epsilon: f64, max_ulps: u32) -> bool {
        self.cos.ulps_eq(&other.cos, epsilon, max_ulps)
            && self.sin.ulps_eq(&other.sin, epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use core::f64::consts::FRAC_PI_2;

    use approx::assert_relative_eq;
    use bevy_math::DVec2;

    use crate::DRot2;

    #[test]
    fn creation() {
        let rotation1 = DRot2::radians(FRAC_PI_2);
        let rotation2 = DRot2::degrees(90.0);
        let rotation3 = DRot2::from_sin_cos(1.0, 0.0);
        let rotation4 = DRot2::turn_fraction(0.25);

        // All three rotations should be equal
        assert_relative_eq!(rotation1.sin, rotation2.sin);
        assert_relative_eq!(rotation1.cos, rotation2.cos);
        assert_relative_eq!(rotation1.sin, rotation3.sin);
        assert_relative_eq!(rotation1.cos, rotation3.cos);
        assert_relative_eq!(rotation1.sin, rotation4.sin);
        assert_relative_eq!(rotation1.cos, rotation4.cos);

        // The rotation should be 90 degrees
        assert_relative_eq!(rotation1.as_radians(), FRAC_PI_2);
        assert_relative_eq!(rotation1.as_degrees(), 90.0);
        assert_relative_eq!(rotation1.as_turn_fraction(), 0.25);
    }

    #[test]
    fn rotate() {
        let rotation = DRot2::degrees(90.0);

        assert_relative_eq!(rotation * DVec2::X, DVec2::Y);
    }

    #[test]
    fn rotation_range() {
        // the rotation range is `(-180, 180]` and the constructors
        // normalize the rotations to that range
        assert_relative_eq!(DRot2::radians(3.0 * FRAC_PI_2), DRot2::radians(-FRAC_PI_2));
        assert_relative_eq!(DRot2::degrees(270.0), DRot2::degrees(-90.0));
        assert_relative_eq!(DRot2::turn_fraction(0.75), DRot2::turn_fraction(-0.25));
    }

    #[test]
    fn add() {
        let rotation1 = DRot2::degrees(90.0);
        let rotation2 = DRot2::degrees(180.0);

        // 90 deg + 180 deg becomes -90 deg after it wraps around to be within the `(-180, 180]` range
        assert_eq!((rotation1 * rotation2).as_degrees(), -90.0);
    }

    #[test]
    fn subtract() {
        let rotation1 = DRot2::degrees(90.0);
        let rotation2 = DRot2::degrees(45.0);

        assert_relative_eq!((rotation1 * rotation2.inverse()).as_degrees(), 45.0);

        // This should be equivalent to the above
        assert_relative_eq!(rotation2.angle_to(rotation1), core::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn length() {
        let rotation = DRot2 {
            sin: 10.0,
            cos: 5.0,
        };

        assert_eq!(rotation.length_squared(), 125.0);
        assert_eq!(rotation.length(), 11.18034);
        assert!((rotation.normalize().length() - 1.0).abs() < 10e-7);
    }

    #[test]
    fn is_near_identity() {
        assert!(!DRot2::radians(0.1).is_near_identity());
        assert!(!DRot2::radians(-0.1).is_near_identity());
        assert!(DRot2::radians(0.00001).is_near_identity());
        assert!(DRot2::radians(-0.00001).is_near_identity());
        assert!(DRot2::radians(0.0).is_near_identity());
    }

    #[test]
    fn normalize() {
        let rotation = DRot2 {
            sin: 10.0,
            cos: 5.0,
        };
        let normalized_rotation = rotation.normalize();

        assert_eq!(normalized_rotation.sin, 0.89442724);
        assert_eq!(normalized_rotation.cos, 0.44721362);

        assert!(!rotation.is_normalized());
        assert!(normalized_rotation.is_normalized());
    }

    #[test]
    fn fast_renormalize() {
        let rotation = DRot2 { sin: 1.0, cos: 0.5 };
        let normalized_rotation = rotation.normalize();

        let mut unnormalized_rot = rotation;
        let mut renormalized_rot = rotation;
        let mut initially_normalized_rot = normalized_rotation;
        let mut fully_normalized_rot = normalized_rotation;

        // Compute a 64x (=2⁶) multiple of the rotation.
        for _ in 0..6 {
            unnormalized_rot = unnormalized_rot * unnormalized_rot;
            renormalized_rot = renormalized_rot * renormalized_rot;
            initially_normalized_rot = initially_normalized_rot * initially_normalized_rot;
            fully_normalized_rot = fully_normalized_rot * fully_normalized_rot;

            renormalized_rot = renormalized_rot.fast_renormalize();
            fully_normalized_rot = fully_normalized_rot.normalize();
        }

        assert!(!unnormalized_rot.is_normalized());

        assert!(renormalized_rot.is_normalized());
        assert!(fully_normalized_rot.is_normalized());

        assert_relative_eq!(fully_normalized_rot, renormalized_rot, epsilon = 0.000001);
        assert_relative_eq!(
            fully_normalized_rot,
            unnormalized_rot.normalize(),
            epsilon = 0.000001
        );
        assert_relative_eq!(
            fully_normalized_rot,
            initially_normalized_rot.normalize(),
            epsilon = 0.000001
        );
    }

    #[test]
    fn try_normalize() {
        // Valid
        assert!(
            DRot2 {
                sin: 10.0,
                cos: 5.0,
            }
            .try_normalize()
            .is_some()
        );

        // NaN
        assert!(
            DRot2 {
                sin: f64::NAN,
                cos: 5.0,
            }
            .try_normalize()
            .is_none()
        );

        // Zero
        assert!(DRot2 { sin: 0.0, cos: 0.0 }.try_normalize().is_none());

        // Non-finite
        assert!(
            DRot2 {
                sin: f64::INFINITY,
                cos: 5.0,
            }
            .try_normalize()
            .is_none()
        );
    }

    #[test]
    fn nlerp() {
        let rot1 = DRot2::IDENTITY;
        let rot2 = DRot2::degrees(135.0);

        assert_eq!(rot1.nlerp(rot2, 1.0 / 3.0).as_degrees(), 28.675055);
        assert!(rot1.nlerp(rot2, 0.0).is_near_identity());
        assert_eq!(rot1.nlerp(rot2, 0.5).as_degrees(), 67.5);
        assert_eq!(rot1.nlerp(rot2, 1.0).as_degrees(), 135.0);

        let rot1 = DRot2::IDENTITY;
        let rot2 = DRot2::from_sin_cos(0.0, -1.0);

        assert!(rot1.nlerp(rot2, 1.0 / 3.0).is_near_identity());
        assert!(rot1.nlerp(rot2, 0.0).is_near_identity());
        // At 0.5, there is no valid rotation, so the fallback is the original angle.
        assert_eq!(rot1.nlerp(rot2, 0.5).as_degrees(), 0.0);
        assert_eq!(rot1.nlerp(rot2, 1.0).as_degrees().abs(), 180.0);
    }

    #[test]
    fn slerp() {
        let rot1 = DRot2::IDENTITY;
        let rot2 = DRot2::degrees(135.0);

        assert_eq!(rot1.slerp(rot2, 1.0 / 3.0).as_degrees(), 45.0);
        assert!(rot1.slerp(rot2, 0.0).is_near_identity());
        assert_eq!(rot1.slerp(rot2, 0.5).as_degrees(), 67.5);
        assert_eq!(rot1.slerp(rot2, 1.0).as_degrees(), 135.0);

        let rot1 = DRot2::IDENTITY;
        let rot2 = DRot2::from_sin_cos(0.0, -1.0);

        assert!((rot1.slerp(rot2, 1.0 / 3.0).as_degrees() - 60.0).abs() < 10e-6);
        assert!(rot1.slerp(rot2, 0.0).is_near_identity());
        assert_eq!(rot1.slerp(rot2, 0.5).as_degrees(), 90.0);
        assert_eq!(rot1.slerp(rot2, 1.0).as_degrees().abs(), 180.0);
    }
}
