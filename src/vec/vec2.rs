use core::ops::*;
use glam::{DVec2, Vec2};
use wide::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe, f32x4, f32x8, f64x2, f64x4};

use crate::SimdLaneCount;
#[cfg(feature = "f64")]
use crate::{BDVec2x2, BDVec2x4, DVec3x2, DVec3x4, boolf64x2, boolf64x4};
#[cfg(feature = "f32")]
use crate::{BVec2x4, BVec2x8, Vec3x4, Vec3x8, boolf32x4, boolf32x8};

macro_rules! wide_vec2s {
    ($(($nonwiden:ident, $n:ident, $v3t:ident, $bvt:ident) => ($nonwidet:ident, $t:ident, $bool:ident)),+) => {
        $(
        /// A 2-dimensional wide vector.
        #[derive(Clone, Copy, Debug, Default)]
        #[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::TypePath))]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[repr(C)]
        pub struct $n {
            /// The X component of the vector.
            pub x: $t,
            /// The Y component of the vector.
            pub y: $t,
        }

        impl $n {
            /// All zeros.
            pub const ZERO: Self = Self::new_splat(0.0, 0.0);

            /// All ones.
            pub const ONE: Self = Self::new_splat(1.0, 1.0);

            /// All negative ones.
            pub const NEG_ONE: Self = Self::new_splat(-1.0, -1.0);

            /// All `MIN`.
            pub const MIN: Self = Self::new_splat($nonwidet::MIN, $nonwidet::MIN);

            /// All `MAX`.
            pub const MAX: Self = Self::new_splat($nonwidet::MAX, $nonwidet::MAX);

            /// All `NAN`.
            pub const NAN: Self = Self::new_splat($nonwidet::NAN, $nonwidet::NAN);

            /// All `INFINITY`.
            pub const INFINITY: Self = Self::new_splat($nonwidet::INFINITY, $nonwidet::INFINITY);

            /// All `NEG_INFINITY`.
            pub const NEG_INFINITY: Self = Self::new_splat($nonwidet::NEG_INFINITY, $nonwidet::NEG_INFINITY);

            /// A unit vector pointing along the positive X axis.
            pub const X: Self = Self::new_splat(1.0, 0.0);

            /// A unit vector pointing along the negative X axis.
            pub const NEG_X: Self = Self::new_splat(-1.0, 0.0);

            /// A unit vector pointing along the positive Y axis.
            pub const Y: Self = Self::new_splat(0.0, 1.0);

            /// A unit vector pointing along the negative Y axis.
            pub const NEG_Y: Self = Self::new_splat(0.0, -1.0);

            /// The unit axes.
            pub const AXES: [Self; 2] = [Self::X, Self::Y];

            /// Creates a new vector.
            #[inline(always)]
            #[must_use]
            pub const fn new(x: $t, y: $t) -> Self {
                Self { x, y }
            }

            /// Creates a new vector with all lanes set to the same `x` and `y` values.
            #[inline]
            #[must_use]
            pub const fn new_splat(x: $nonwidet, y: $nonwidet) -> Self {
                Self {
                    x: $t::new([x; $t::LANES]),
                    y: $t::new([y; $t::LANES]),
                }
            }

            /// Creates a new vector with all lanes set to `v`.
            #[inline]
            #[must_use]
            pub const fn splat(v: $nonwiden) -> Self {
                Self {
                    x: $t::new([v.x; $t::LANES]),
                    y: $t::new([v.y; $t::LANES]),
                }
            }

            /// Blend two vectors together lanewise using `mask` as a mask.
            ///
            /// This is essentially a bitwise blend operation, such that any point where
            /// there is a 1 bit in `mask`, the output will put the bit from `if_true`, while
            /// where there is a 0 bit in `mask`, the output will put the bit from `if_false`.
            #[inline]
            #[must_use]
            pub fn blend(mask: $bool, if_true: Self, if_false: Self) -> Self {
                Self {
                    x: mask.to_raw().blend(if_true.x, if_false.x),
                    y: mask.to_raw().blend(if_true.y, if_false.y),
                }
            }

            /// Returns a vector containing each element of `self` modified by a mapping function `f`.
            #[inline]
            #[must_use]
            pub fn map<F>(self, f: F) -> Self
            where
                F: Fn($t) -> $t,
            {
                Self::new(f(self.x), f(self.y))
            }

            /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use
            /// for each element of `self`.
            ///
            /// A true element in the mask uses the corresponding element from `if_true`, and false
            /// uses the element from `if_false`.
            #[inline]
            #[must_use]
            pub fn select(mask: $bvt, if_true: Self, if_false: Self) -> Self {
                Self {
                    x: mask.test(0).to_raw().blend(if_true.x, if_false.x),
                    y: mask.test(1).to_raw().blend(if_true.y, if_false.y),
                }
            }

            /// Creates a new vector from an array.
            #[inline]
            #[must_use]
            pub const fn from_array(arr: [$t; 2]) -> Self {
                Self::new(arr[0], arr[1])
            }

            /// Returns the vector as an array.
            #[inline]
            #[must_use]
            pub const fn to_array(self) -> [$t; 2] {
                [self.x, self.y]
            }

            /// Creates a vector from the first 2 values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 2 elements long.
            #[inline]
            #[must_use]
            pub const fn from_slice(slice: &[$t]) -> Self {
                assert!(slice.len() == 2);
                Self::new(slice[0], slice[1])
            }

            /// Writes the elements of `self` to the first 2 elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 2 elements long.
            #[inline]
            pub fn write_to_slice(self, slice: &mut [$t]) {
                slice[..2].copy_from_slice(&self.to_array());
            }

            /// Creates a 3D vector from `self` and the given `z` value.
            #[inline]
            #[must_use]
            pub const fn extend(self, z: $t) -> $v3t {
                $v3t::new(self.x, self.y, z)
            }

            /// Creates a 2D vector from `self` with the given value of `x`.
            #[inline]
            #[must_use]
            pub const fn with_x(self, x: $t) -> Self {
                Self::new(x, self.y)
            }

            /// Creates a 2D vector from `self` with the given value of `y`.
            #[inline]
            #[must_use]
            pub const fn with_y(self, y: $t) -> Self {
                Self::new(self.x, y)
            }

            /// Computes the dot product of `self` and `rhs`.
            #[inline]
            #[must_use]
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y
            }

            /// Returns a vector where every component is the dot product of `self` and `rhs`.
            #[inline]
            #[must_use]
            pub fn dot_into_vec(self, rhs: Self) -> Self {
                let dot = self.dot(rhs);
                Self::new(dot, dot)
            }

            /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
            ///
            /// In other words this computes `[min(x, rhs.x), min(self.y, rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub fn min(self, rhs: Self) -> Self {
                Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
            }

            /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
            ///
            /// In other words this computes `[max(self.x, rhs.x), max(self.y, rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub fn max(self, rhs: Self) -> Self {
                Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
            }

            /// Component-wise clamping of values.
            ///
            /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
            #[inline]
            #[must_use]
            pub fn clamp(mut self, min: Self, max: Self) -> Self {
                self.x = self.x.max(min.x).min(max.x);
                self.y = self.y.max(min.y).min(max.y);
                self
            }

            /// Returns a vector with a length no less than `min` and no more than `max`.
            #[inline]
            #[must_use]
            pub fn clamp_length(self, min: $t, max: $t) -> Self {
                let length = self.length();
                let scale = (min / length).max($t::ONE).min(max / length);
                Self::new(self.x * scale, self.y * scale)
            }

            /// Returns a vector with a length no less than `min`.
            #[inline]
            #[must_use]
            pub fn clamp_length_min(self, min: $t) -> Self {
                let length = self.length();
                let scale = (min / length).max($t::ONE);
                Self::new(self.x * scale, self.y * scale)
            }

            /// Returns a vector with a length no more than `max`.
            #[inline]
            #[must_use]
            pub fn clamp_length_max(self, max: $t) -> Self {
                let length = self.length();
                let scale = (max / length).min($t::ONE);
                Self::new(self.x * scale, self.y * scale)
            }

            /// Returns the horizontal minimum of `self`.
            ///
            /// In other words this computes `min(x, y, ..)`.
            #[inline]
            #[must_use]
            pub fn min_element(self) -> $t {
                self.x.min(self.y)
            }

            /// Returns the horizontal maximum of `self`.
            ///
            /// In other words this computes `max(x, y, ..)`.
            #[inline]
            #[must_use]
            pub fn max_element(self) -> $t {
                self.x.max(self.y)
            }

            /// Returns the sum of all elements of `self`.
            ///
            /// In other words, this computes `self.x + self.y + ..`.
            #[inline]
            #[must_use]
            pub fn element_sum(self) -> $t {
                self.x + self.y
            }

            /// Returns the product of all elements of `self`.
            ///
            /// In other words, this computes `self.x * self.y * ..`.
            #[inline]
            #[must_use]
            pub fn element_product(self) -> $t {
                self.x * self.y
            }

            /// Returns a vector mask containing the result of a `==` comparison for each element of
            /// `self` and `rhs`.
            ///
            /// In other words, this computes `[self.x == rhs.x, self.y == rhs.y, ..]` for all
            /// elements.
            #[inline]
            #[must_use]
            pub fn cmpeq(self, rhs: Self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.cmp_eq(rhs.x)),
                    $bool::from_raw(self.y.cmp_eq(rhs.y)),
                )
            }

            /// Returns a vector mask containing the result of a `!=` comparison for each element of
            /// `self` and `rhs`.
            ///
            /// In other words this computes `[self.x != rhs.x, self.y != rhs.y, ..]` for all
            /// elements.
            #[inline]
            #[must_use]
            pub fn cmpne(self, rhs: Self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.cmp_ne(rhs.x)),
                    $bool::from_raw(self.y.cmp_ne(rhs.y)),
                )
            }

            /// Returns a vector mask containing the result of a `>=` comparison for each element of
            /// `self` and `rhs`.
            ///
            /// In other words this computes `[self.x >= rhs.x, self.y >= rhs.y, ..]` for all
            /// elements.
            #[inline]
            #[must_use]
            pub fn cmpge(self, rhs: Self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.cmp_ge(rhs.x)),
                    $bool::from_raw(self.y.cmp_ge(rhs.y)),
                )
            }

            /// Returns a vector mask containing the result of a `>` comparison for each element of
            /// `self` and `rhs`.
            ///
            /// In other words this computes `[self.x > rhs.x, self.y > rhs.y, ..]` for all
            /// elements.
            #[inline]
            #[must_use]
            pub fn cmpgt(self, rhs: Self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.cmp_gt(rhs.x)),
                    $bool::from_raw(self.y.cmp_gt(rhs.y)),
                )
            }

            /// Returns a vector mask containing the result of a `<=` comparison for each element of
            /// `self` and `rhs`.
            ///
            /// In other words this computes `[self.x <= rhs.x, self.y <= rhs.y, ..]` for all
            /// elements.
            #[inline]
            #[must_use]
            pub fn cmple(self, rhs: Self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.cmp_le(rhs.x)),
                    $bool::from_raw(self.y.cmp_le(rhs.y)),
                )
            }

            /// Returns a vector mask containing the result of a `<` comparison for each element of
            /// `self` and `rhs`.
            ///
            /// In other words this computes `[self.x < rhs.x, self.y < rhs.y, ..]` for all
            /// elements.
            #[inline]
            #[must_use]
            pub fn cmplt(self, rhs: Self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.cmp_lt(rhs.x)),
                    $bool::from_raw(self.y.cmp_lt(rhs.y)),
                )
            }

            /// Returns a vector containing the absolute value of each element of `self`.
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                Self::new(self.x.abs(), self.y.abs())
            }

            /// Returns a vector with signs of `rhs` and the magnitudes of `self`.
            #[inline]
            #[must_use]
            pub fn copysign(self, rhs: Self) -> Self {
                Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y))
            }

            /// Returns `true` if, and only if, all elements are finite.  If any element is either
            /// `NaN`, positive or negative infinity, this will return `false`.
            #[inline]
            #[must_use]
            pub fn is_finite(self) -> $bool {
                $bool::from_raw(self.x.is_finite()) & $bool::from_raw(self.y.is_finite())
            }

            /// Performs `is_finite` on each element of self, returning a vector mask of the results.
            ///
            /// In other words, this computes `[x.is_finite(), y.is_finite(), ...]`.
            #[inline]
            #[must_use]
            pub fn is_finite_mask(self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.is_finite()),
                    $bool::from_raw(self.y.is_finite()),
                )
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            #[must_use]
            pub fn is_nan(self) -> $bool {
                $bool::from_raw(self.x.is_nan()) | $bool::from_raw(self.y.is_nan())
            }

            /// Performs `is_nan` on each element of self, returning a vector mask of the results.
            ///
            /// In other words, this computes `[x.is_nan(), y.is_nan(), ...]`.
            #[inline]
            #[must_use]
            pub fn is_nan_mask(self) -> $bvt {
                $bvt::new(
                    $bool::from_raw(self.x.is_nan()),
                    $bool::from_raw(self.y.is_nan()),
                )
            }

            /// Computes the length of `self`.
            #[inline]
            #[must_use]
            pub fn length(self) -> $t {
                self.length_squared().sqrt()
            }

            /// Computes the squared length of `self`.
            ///
            /// This is faster than `length()` as it avoids a square root operation.
            #[inline]
            #[must_use]
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }

            /// Computes `1.0 / length()`.
            ///
            /// For valid results, `self` must _not_ be of length zero.
            #[inline]
            #[must_use]
            pub fn length_recip(self) -> $t {
                $t::ONE / self.length()
            }

            /// Computes the Euclidean distance between two points in space.
            #[inline]
            #[must_use]
            pub fn distance(self, rhs: Self) -> $t {
                (self - rhs).length()
            }

            /// Compute the squared euclidean distance between two points in space.
            #[inline]
            #[must_use]
            pub fn distance_squared(self, rhs: Self) -> $t {
                (self - rhs).length_squared()
            }

            /// Returns `self` normalized to length 1.0.
            ///
            /// For valid results, `self` must be finite and _not_ of length zero, nor very close to zero.
            #[inline]
            #[must_use]
            pub fn normalize(self) -> Self {
                self.mul(self.length_recip())
            }

            /// Returns `self` normalized to length 1.0 if possible, else returns a
            /// fallback value.
            ///
            /// In particular, if the input is zero (or very close to zero), or non-finite,
            /// the result of this operation will be the fallback value.
            ///
            /// See also [`Self::try_normalize()`].
            #[inline]
            #[must_use]
            pub fn normalize_or(self, fallback: Self) -> Self {
                let rcp = self.length_recip();
                let mask = $bool::from_raw(rcp.is_finite() & rcp.cmp_gt($t::ZERO));
                Self::blend(mask, self * rcp, fallback)
            }

            /// Returns `self` normalized to length 1.0 if possible, else returns zero.
            ///
            /// In particular, if the input is zero (or very close to zero), or non-finite,
            /// the result of this operation will be zero.
            ///
            /// See also [`Self::try_normalize()`].
            #[inline]
            #[must_use]
            pub fn normalize_or_zero(self) -> Self {
                self.normalize_or(Self::ZERO)
            }

            /// Returns `self` normalized to length 1.0 and the length of `self`.
            ///
            /// If `self` is zero length then `(Self::X, 0.0)` is returned.
            #[inline]
            #[must_use]
            pub fn normalize_and_length(self) -> (Self, $t) {
                let length = self.length();
                let rcp = $t::ONE / length;
                let mask = $bool::from_raw(rcp.is_finite() & rcp.cmp_gt($t::ZERO));
                (
                    Self::blend(mask, self * rcp, Self::X),
                    $t::blend(mask.to_raw(), length, $t::ZERO),
                )
            }

            /// Returns whether `self` is length `1.0` or not.
            ///
            /// Uses a precision threshold of approximately `1e-4`.
            #[inline]
            #[must_use]
            pub fn is_normalized(self) -> $bool {
                $bool::from_raw((self.length_squared() - 1.0).abs().cmp_le($t::splat(2e-4)))
            }

            /// Returns the vector projection of `self` onto `rhs`.
            ///
            /// `rhs` must be of non-zero length.
            #[inline]
            #[must_use]
            pub fn project_onto(self, rhs: Self) -> Self {
                rhs * self.dot(rhs) / rhs.length_squared()
            }

            /// Returns the vector rejection of `self` from `rhs`.
            ///
            /// The vector rejection is the vector perpendicular to the projection of `self` onto
            /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
            ///
            /// `rhs` must be of non-zero length.
            #[inline]
            #[must_use]
            pub fn reject_from(self, rhs: Self) -> Self {
                self - self.project_onto(rhs)
            }

            /// Returns the vector projection of `self` onto `rhs`.
            ///
            /// `rhs` must be normalized.
            #[inline]
            #[must_use]
            pub fn project_onto_normalized(self, rhs: Self) -> Self {
                rhs * self.dot(rhs)
            }

            /// Returns the vector rejection of `self` from `rhs`.
            ///
            /// The vector rejection is the vector perpendicular to the projection of `self` onto
            /// `rhs`, in rhs words the result of `self - self.project_onto(rhs)`.
            ///
            /// `rhs` must be normalized.
            #[inline]
            #[must_use]
            pub fn reject_from_normalized(self, rhs: Self) -> Self {
                self - self.project_onto_normalized(rhs)
            }

            /// Returns a vector containing the nearest integer to a number for each element of `self`.
            /// Round half-way cases away from `0.0`.
            #[inline]
            #[must_use]
            pub fn round(self) -> Self {
                Self::new(self.x.round(), self.y.round())
            }

            /// Returns a vector containing the largest integer less than or equal to a number for each
            /// element of `self`.
            #[inline]
            #[must_use]
            pub fn floor(self) -> Self {
                Self::new(self.x.floor(), self.y.floor())
            }

            /// Returns a vector containing the smallest integer greater than or equal to a number for
            /// each element of `self`.
            #[inline]
            #[must_use]
            pub fn ceil(self) -> Self {
                Self::new(self.x.ceil(), self.y.ceil())
            }

            /// Returns a vector containing the reciprocal `1.0 / n` of each element of `self`.
            #[inline]
            #[must_use]
            pub fn recip(self) -> Self {
                Self::new($t::ONE / self.x, $t::ONE / self.y)
            }

            /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
            ///
            /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
            /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
            /// extrapolated.
            #[inline]
            #[must_use]
            pub fn lerp(self, rhs: Self, s: $t) -> Self {
                self * ($t::ONE - s) + rhs * s
            }

            /// Calculates the midpoint between `self` and `rhs`.
            ///
            /// The midpoint is the average of, or halfway point between, two vectors.
            /// `a.midpoint(b)` should yield the same result as `a.lerp(b, 0.5)`
            /// while being slightly cheaper to compute.
            #[inline]
            pub fn midpoint(self, rhs: Self) -> Self {
                (self + rhs) * $t::splat(0.5)
            }

            /// Returns true if the absolute difference of all elements between `self` and `rhs` is
            /// less than or equal to `max_abs_diff`.
            ///
            /// This can be used to compare if two vectors contain similar elements. It works best when
            /// comparing with a known value. The `max_abs_diff` that should be used used depends on
            /// the values being compared against.
            ///
            /// For more see
            /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
            #[inline]
            #[must_use]
            pub fn abs_diff_eq(self, rhs: Self, max_abs_diff: $t) -> $bool {
                self.sub(rhs).abs().cmple(Self::new(max_abs_diff, max_abs_diff)).all()
            }

            /// Fused multiply-add. Computes `(self * a) + b` element-wise with only one rounding
            /// error, yielding a more accurate result than an unfused multiply-add.
            ///
            /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
            /// architecture has a dedicated fma CPU instruction. However, this is not always true,
            /// and will be heavily dependant on designing algorithms with specific target hardware in
            /// mind.
            #[inline]
            #[must_use]
            pub fn mul_add(self, a: Self, b: Self) -> Self {
                Self::new(
                    self.x.mul_add(a.x, b.x),
                    self.y.mul_add(a.y, b.y),
                )
            }

            /// Returns the reflection vector for a given incident vector `self` and surface normal
            /// `normal`.
            ///
            /// `normal` must be normalized.
            #[inline]
            #[must_use]
            pub fn reflect(self, normal: Self) -> Self {
                self - 2.0 * self.dot(normal) * normal
            }

            /// Returns the refraction direction for a given incident vector `self`, surface normal
            /// `normal` and ratio of indices of refraction, `eta`. When total internal reflection occurs,
            /// a zero vector will be returned.
            ///
            /// `self` and `normal` must be normalized.
            #[inline]
            pub fn refract(self, normal: Self, eta: $t) -> Self {
                let n = normal;
                let one = $t::splat(1.0);
                let ndi = n.dot(self);

                let k = one - eta * eta * (one - ndi * ndi);
                let mask = $bool::from_raw(k.cmp_gt($t::splat(0.0)));

                let out = self * eta - (eta * ndi + k.sqrt()) * n;

                Self::blend(mask, Self::ZERO, out)
            }

            /// Returns a vector that is equal to `self` rotated by 90 degrees.
            #[inline]
            #[must_use]
            pub fn perp(self) -> Self {
                Self::new(-self.y, self.x)
            }

            /// The perpendicular dot product of `self` and `rhs`.
            /// Also known as the wedge product, 2D cross product, and determinant.
            #[doc(alias = "wedge")]
            #[doc(alias = "cross")]
            #[doc(alias = "determinant")]
            #[inline]
            #[must_use]
            pub fn perp_dot(self, rhs: Self) -> $t {
                self.x * rhs.y - self.y * rhs.x
            }
        }

        impl From<$n> for [$t; 2] {
            #[inline]
            fn from(v: $n) -> Self {
                [v.x, v.y]
            }
        }

        impl From<[$t; 2]> for $n {
            #[inline]
            fn from(comps: [$t; 2]) -> Self {
                Self::new(comps[0], comps[1])
            }
        }

        impl From<&[$t; 2]> for $n {
            #[inline]
            fn from(comps: &[$t; 2]) -> Self {
                Self::from(*comps)
            }
        }

        impl From<&mut [$t; 2]> for $n {
            #[inline]
            fn from(comps: &mut [$t; 2]) -> Self {
                Self::from(*comps)
            }
        }

        impl From<($t, $t)> for $n {
            #[inline]
            fn from(comps: ($t, $t)) -> Self {
                Self::new(comps.0, comps.1)
            }
        }

        impl From<&($t, $t)> for $n {
            #[inline]
            fn from(comps: &($t, $t)) -> Self {
                Self::from(*comps)
            }
        }

        impl From<$n> for ($t, $t) {
            #[inline]
            fn from(v: $n) -> Self {
                (v.x, v.y)
            }
        }

        impl Add for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: $n) -> Self {
                $n::new(self.x + rhs.x, self.y + rhs.y)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: $n) {
                self.x += rhs.x;
                self.y += rhs.y;
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $n) -> Self {
                $n::new(self.x - rhs.x, self.y - rhs.y)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: $n) {
                self.x -= rhs.x;
                self.y -= rhs.y;
            }
        }

        impl Mul for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $n) -> Self {
                $n::new(self.x * rhs.x, self.y * rhs.y)
            }
        }

        impl Mul<$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> $n {
                $n::new(self * rhs.x, self * rhs.y)
            }
        }

        impl Mul<$t> for $n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $t) -> $n {
                $n::new(self.x * rhs, self.y * rhs)
            }
        }

        impl MulAssign for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $n) {
                self.x *= rhs.x;
                self.y *= rhs.y;
            }
        }

        impl MulAssign<$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                self.x *= rhs;
                self.y *= rhs;
            }
        }

        impl Div for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $n) -> Self {
                $n::new(self.x / rhs.x, self.y / rhs.y)
            }
        }

        impl Div<$t> for $n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $t) -> $n {
                $n::new(self.x / rhs, self.y / rhs)
            }
        }

        impl DivAssign for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $n) {
                self.x /= rhs.x;
                self.y /= rhs.y;
            }
        }

        impl DivAssign<$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                self.x /= rhs;
                self.y /= rhs;
            }
        }

        impl Neg for $n {
            type Output = $n;
            #[inline]
            fn neg(self) -> $n {
                self * $t::splat(-1.0)
            }
        }

        impl Index<usize> for $n {
            type Output = $t;

            fn index(&self, index: usize) -> &Self::Output {
                match index {
                    0 => &self.x,
                    1 => &self.y,
                    i => panic!("Invalid index {i} for vector of type: {}", core::any::type_name::<$n>()),
                }
            }
        }

        impl IndexMut<usize> for $n {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                match index {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    i => panic!("Invalid index {i} for vector of type: {}", core::any::type_name::<$n>()),
                }
            }
        }

        impl core::iter::Sum<$n> for $n {
            fn sum<I>(iter: I) -> Self where I: Iterator<Item = Self> {
                // Kahan summation algorithm
                // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
                let mut sum = $n::ZERO;
                let mut c = $n::ZERO;
                for v in iter {
                    let y = v - c;
                    let t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                sum
            }
        }

        impl From<$nonwiden> for $n {
            #[inline]
            fn from(vec: $nonwiden) -> Self {
                Self::splat(vec)
            }
        }

        impl From<$v3t> for $n {
            #[inline]
            fn from(vec: $v3t) -> Self {
                Self { x: vec.x, y: vec.y }
            }
        }
        )+
    };
}

#[cfg(feature = "f32")]
impl From<Vec2x4> for [Vec2; 4] {
    #[inline]
    fn from(v: Vec2x4) -> Self {
        let xs: [f32; 4] = v.x.into();
        let ys: [f32; 4] = v.y.into();
        [
            Vec2::new(xs[0], ys[0]),
            Vec2::new(xs[1], ys[1]),
            Vec2::new(xs[2], ys[2]),
            Vec2::new(xs[3], ys[3]),
        ]
    }
}

#[cfg(feature = "f32")]
impl From<[Vec2; 4]> for Vec2x4 {
    #[inline]
    fn from(vecs: [Vec2; 4]) -> Self {
        Self {
            x: f32x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f32x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
        }
    }
}

#[cfg(feature = "f32")]
impl From<Vec2x8> for [Vec2; 8] {
    #[inline]
    fn from(v: Vec2x8) -> Self {
        let xs: [f32; 8] = v.x.into();
        let ys: [f32; 8] = v.y.into();
        [
            Vec2::new(xs[0], ys[0]),
            Vec2::new(xs[1], ys[1]),
            Vec2::new(xs[2], ys[2]),
            Vec2::new(xs[3], ys[3]),
            Vec2::new(xs[4], ys[4]),
            Vec2::new(xs[5], ys[5]),
            Vec2::new(xs[6], ys[6]),
            Vec2::new(xs[7], ys[7]),
        ]
    }
}

#[cfg(feature = "f32")]
impl From<[Vec2; 8]> for Vec2x8 {
    #[inline]
    fn from(vecs: [Vec2; 8]) -> Self {
        Self {
            x: f32x8::from([
                vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x, vecs[4].x, vecs[5].x, vecs[6].x,
                vecs[7].x,
            ]),
            y: f32x8::from([
                vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y, vecs[4].y, vecs[5].y, vecs[6].y,
                vecs[7].y,
            ]),
        }
    }
}

#[cfg(feature = "f64")]
impl From<DVec2x2> for [DVec2; 2] {
    #[inline]
    fn from(v: DVec2x2) -> Self {
        let xs: [f64; 2] = v.x.into();
        let ys: [f64; 2] = v.y.into();
        [DVec2::new(xs[0], ys[0]), DVec2::new(xs[1], ys[1])]
    }
}

#[cfg(feature = "f64")]
impl From<[DVec2; 2]> for DVec2x2 {
    #[inline]
    fn from(vecs: [DVec2; 2]) -> Self {
        Self {
            x: f64x2::from([vecs[0].x, vecs[1].x]),
            y: f64x2::from([vecs[0].y, vecs[1].y]),
        }
    }
}

#[cfg(feature = "f64")]
impl From<DVec2x4> for [DVec2; 4] {
    #[inline]
    fn from(v: DVec2x4) -> Self {
        let xs: [f64; 4] = v.x.into();
        let ys: [f64; 4] = v.y.into();
        [
            DVec2::new(xs[0], ys[0]),
            DVec2::new(xs[1], ys[1]),
            DVec2::new(xs[2], ys[2]),
            DVec2::new(xs[3], ys[3]),
        ]
    }
}

#[cfg(feature = "f64")]
impl From<[DVec2; 4]> for DVec2x4 {
    #[inline]
    fn from(vecs: [DVec2; 4]) -> Self {
        Self {
            x: f64x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f64x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
        }
    }
}

#[cfg(feature = "f32")]
wide_vec2s!(
    (Vec2, Vec2x4, Vec3x4, BVec2x4) => (f32, f32x4, boolf32x4),
    (Vec2, Vec2x8, Vec3x8, BVec2x8) => (f32, f32x8, boolf32x8)
);

#[cfg(feature = "f64")]
wide_vec2s!(
    (DVec2, DVec2x2, DVec3x2, BDVec2x2) => (f64, f64x2, boolf64x2),
    (DVec2, DVec2x4, DVec3x4, BDVec2x4) => (f64, f64x4, boolf64x4)
);
