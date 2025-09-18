use core::ops::*;
use glam::{DVec4, Vec4};
use wide::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe, f32x4, f32x8, f64x2, f64x4};

use crate::SimdLaneCount;
#[cfg(feature = "f64")]
use crate::{BDVec4x2, BDVec4x4, DVec3x2, DVec3x4, boolf64x2, boolf64x4};
#[cfg(feature = "f32")]
use crate::{BVec4x4, BVec4x8, Vec3x4, Vec3x8, boolf32x4, boolf32x8};

macro_rules! wide_vec4s {
    ($(($v3t:ident, $nonwiden:ident, $n:ident, $bvt:ident) => ($nonwidet:ident, $t:ident, $bool:ident, $pow:ident)),+) => {
        $(
        /// A 4-dimensional wide vector.
        #[derive(Clone, Copy, Debug, Default)]
        #[cfg_attr(feature = "bevy_reflect", derive(bevy_reflect::TypePath))]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[repr(C)]
        pub struct $n {
            /// The X component of the vector.
            pub x: $t,
            /// The Y component of the vector.
            pub y: $t,
            /// The Z component of the vector.
            pub z: $t,
            /// The W component of the vector.
            pub w: $t,
        }

        impl $n {
            /// All zeros.
            pub const ZERO: Self = Self::new_splat(0.0, 0.0, 0.0, 0.0);

            /// All ones.
            pub const ONE: Self = Self::new_splat(1.0, 1.0, 1.0, 1.0);

            /// All negative ones.
            pub const NEG_ONE: Self = Self::new_splat(-1.0, -1.0, -1.0, -1.0);

            /// All `MIN`.
            pub const MIN: Self = Self::new_splat($nonwidet::MIN, $nonwidet::MIN, $nonwidet::MIN, $nonwidet::MIN);

            /// All `MAX`.
            pub const MAX: Self = Self::new_splat($nonwidet::MAX, $nonwidet::MAX, $nonwidet::MAX, $nonwidet::MAX);

            /// All `NAN`.
            pub const NAN: Self = Self::new_splat($nonwidet::NAN, $nonwidet::NAN, $nonwidet::NAN, $nonwidet::NAN);

            /// All `INFINITY`.
            pub const INFINITY: Self = Self::new_splat($nonwidet::INFINITY, $nonwidet::INFINITY, $nonwidet::INFINITY, $nonwidet::INFINITY);

            /// All `NEG_INFINITY`.
            pub const NEG_INFINITY: Self = Self::new_splat($nonwidet::NEG_INFINITY, $nonwidet::NEG_INFINITY, $nonwidet::NEG_INFINITY, $nonwidet::NEG_INFINITY);

            /// A unit vector pointing along the positive X axis.
            pub const X: Self = Self::new_splat(1.0, 0.0, 0.0, 0.0);

            /// A unit vector pointing along the negative X axis.
            pub const NEG_X: Self = Self::new_splat(-1.0, 0.0, 0.0, 0.0);

            /// A unit vector pointing along the positive Y axis.
            pub const Y: Self = Self::new_splat(0.0, 1.0, 0.0, 0.0);

            /// A unit vector pointing along the negative Y axis.
            pub const NEG_Y: Self = Self::new_splat(0.0, -1.0, 0.0, 0.0);

            /// A unit vector pointing along the positive Z axis.
            pub const Z: Self = Self::new_splat(0.0, 0.0, 1.0, 0.0);

            /// A unit vector pointing along the negative Z axis.
            pub const NEG_Z: Self = Self::new_splat(0.0, 0.0, -1.0, 0.0);

            /// A unit vector pointing along the positive W axis.
            pub const W: Self = Self::new_splat(0.0, 0.0, 0.0, 1.0);

            /// A unit vector pointing along the negative W axis.
            pub const NEG_W: Self = Self::new_splat(0.0, 0.0, 0.0, -1.0);

            /// The unit axes.
            pub const AXES: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

            /// Creates a new vector.
            #[inline(always)]
            #[must_use]
            pub const fn new(x: $t, y: $t, z: $t, w: $t) -> Self {
                Self { x, y, z, w }
            }

            /// Creates a new vector with all lanes set to the same `x`, `y`, `z`, and `w` values.
            #[inline]
            #[must_use]
            pub const fn new_splat(x: $nonwidet, y: $nonwidet, z: $nonwidet, w: $nonwidet) -> Self {
                Self {
                    x: $t::new([x; $t::LANES]),
                    y: $t::new([y; $t::LANES]),
                    z: $t::new([z; $t::LANES]),
                    w: $t::new([w; $t::LANES]),
                }
            }

            /// Creates a new vector with all lanes set to `v`.
            #[inline]
            #[must_use]
            pub fn splat(v: $nonwiden) -> Self {
                Self {
                    x: $t::new([v.x; $t::LANES]),
                    y: $t::new([v.y; $t::LANES]),
                    z: $t::new([v.z; $t::LANES]),
                    w: $t::new([v.w; $t::LANES]),
                }
            }

            /// Creates a new vector with all components set to `f`.
            #[inline]
            #[must_use]
            pub const fn broadcast(f: $t) -> Self {
                Self {
                    x: f,
                    y: f,
                    z: f,
                    w: f,
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
                    z: mask.to_raw().blend(if_true.z, if_false.z),
                    w: mask.to_raw().blend(if_true.w, if_false.w),
                }
            }

            /// Returns a vector containing each element of `self` modified by a mapping function `f`.
            #[inline]
            #[must_use]
            pub fn map<F>(self, f: F) -> Self
            where
                F: Fn($t) -> $t,
            {
                Self::new(f(self.x), f(self.y), f(self.z), f(self.w))
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
                    z: mask.test(2).to_raw().blend(if_true.z, if_false.z),
                    w: mask.test(3).to_raw().blend(if_true.w, if_false.w),
                }
            }

            /// Creates a new vector from an array.
            #[inline]
            #[must_use]
            pub const fn from_array(arr: [$t; 4]) -> Self {
                Self::new(arr[0], arr[1], arr[2], arr[3])
            }

            /// Returns the vector as an array.
            #[inline]
            #[must_use]
            pub const fn to_array(self) -> [$t; 4] {
                [self.x, self.y, self.z, self.w]
            }

            /// Creates a vector from the first 4 values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline]
            #[must_use]
            pub const fn from_slice(slice: &[$t]) -> Self {
                assert!(slice.len() == 4);
                Self::new(slice[0], slice[1], slice[2], slice[3])
            }

            /// Writes the elements of `self` to the first 4 elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline]
            pub fn write_to_slice(self, slice: &mut [$t]) {
                slice[..4].copy_from_slice(&self.to_array());
            }

            /// Creates a 3D vector from the `x` and `y` elements of `self`, discarding `z`.
            #[inline]
            #[must_use]
            pub const fn truncate(self) -> $v3t {
                $v3t::new(self.x, self.y, self.z)
            }

            /// Creates a 4D vector from `self` with the given value of `x`.
            #[inline]
            #[must_use]
            pub const fn with_x(self, x: $t) -> Self {
                Self::new(x, self.y, self.z, self.w)
            }

            /// Creates a 4D vector from `self` with the given value of `y`.
            #[inline]
            #[must_use]
            pub const fn with_y(self, y: $t) -> Self {
                Self::new(self.x, y, self.z, self.w)
            }

            /// Creates a 4D vector from `self` with the given value of `z`.
            #[inline]
            #[must_use]
            pub const fn with_z(self, z: $t) -> Self {
                Self::new(self.x, self.y, z, self.w)
            }

            /// Creates a 4D vector from `self` with the given value of `w`.
            #[inline]
            #[must_use]
            pub const fn with_w(self, w: $t) -> Self {
                Self::new(self.x, self.y, self.z, w)
            }

            /// Computes the dot product of `self` and `rhs`.
            #[inline]
            #[must_use]
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }

            /// Returns a vector where every component is the dot product of `self` and `rhs`.
            #[inline]
            #[must_use]
            pub fn dot_into_vec(self, rhs: Self) -> Self {
                let dot = self.dot(rhs);
                Self::new(dot, dot, dot, dot)
            }

            /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
            ///
            /// In other words this computes `[min(x, rhs.x), min(self.y, rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub fn min(self, rhs: Self) -> Self {
                Self::new(
                    self.x.min(rhs.x),
                    self.y.min(rhs.y),
                    self.z.min(rhs.z),
                    self.w.min(rhs.w),
                )
            }

            /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
            ///
            /// In other words this computes `[max(self.x, rhs.x), max(self.y, rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub fn max(self, rhs: Self) -> Self {
                Self::new(
                    self.x.max(rhs.x),
                    self.y.max(rhs.y),
                    self.z.max(rhs.z),
                    self.w.max(rhs.w),
                )
            }

            /// Component-wise clamping of values.
            ///
            /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
            #[inline]
            #[must_use]
            pub fn clamp(mut self, min: Self, max: Self) -> Self {
                self.x = self.x.max(min.x).min(max.x);
                self.y = self.y.max(min.y).min(max.y);
                self.z = self.z.max(min.z).min(max.z);
                self.w = self.w.max(min.w).min(max.w);
                self
            }

            /// Returns a vector with a length no less than `min` and no more than `max`.
            #[inline]
            #[must_use]
            pub fn clamp_length(self, min: $t, max: $t) -> Self {
                let length = self.length();
                let scale = (min / length).max($t::ONE).min(max / length);
                Self::new(self.x * scale, self.y * scale, self.z * scale, self.w * scale)
            }

            /// Returns a vector with a length no less than `min`.
            #[inline]
            #[must_use]
            pub fn clamp_length_min(self, min: $t) -> Self {
                let length = self.length();
                let scale = (min / length).max($t::ONE);
                Self::new(self.x * scale, self.y * scale, self.z * scale, self.w * scale)
            }

            /// Returns a vector with a length no more than `max`.
            #[inline]
            #[must_use]
            pub fn clamp_length_max(self, max: $t) -> Self {
                let length = self.length();
                let scale = (max / length).min($t::ONE);
                Self::new(self.x * scale, self.y * scale, self.z * scale, self.w * scale)
            }

            /// Returns the horizontal minimum of `self`.
            ///
            /// In other words this computes `min(x, y, ..)`.
            #[inline]
            #[must_use]
            pub fn min_element(self) -> $t {
                self.x.min(self.y).min(self.z).min(self.w)
            }

            /// Returns the horizontal maximum of `self`.
            ///
            /// In other words this computes `max(x, y, ..)`.
            #[inline]
            #[must_use]
            pub fn max_element(self) -> $t {
                self.x.max(self.y).max(self.z).max(self.w)
            }

            /// Returns the sum of all elements of `self`.
            ///
            /// In other words, this computes `self.x + self.y + ..`.
            #[inline]
            #[must_use]
            pub fn element_sum(self) -> $t {
                self.x + self.y + self.z + self.w
            }

            /// Returns the product of all elements of `self`.
            ///
            /// In other words, this computes `self.x * self.y * ..`.
            #[inline]
            #[must_use]
            pub fn element_product(self) -> $t {
                self.x * self.y * self.z * self.w
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
                    $bool::from_raw(self.z.cmp_eq(rhs.z)),
                    $bool::from_raw(self.w.cmp_eq(rhs.w)),
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
                    $bool::from_raw(self.z.cmp_ne(rhs.z)),
                    $bool::from_raw(self.w.cmp_ne(rhs.w)),
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
                    $bool::from_raw(self.z.cmp_ge(rhs.z)),
                    $bool::from_raw(self.w.cmp_ge(rhs.w)),
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
                    $bool::from_raw(self.z.cmp_gt(rhs.z)),
                    $bool::from_raw(self.w.cmp_gt(rhs.w)),
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
                    $bool::from_raw(self.z.cmp_le(rhs.z)),
                    $bool::from_raw(self.w.cmp_le(rhs.w)),
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
                    $bool::from_raw(self.z.cmp_lt(rhs.z)),
                    $bool::from_raw(self.w.cmp_lt(rhs.w)),
                )
            }

            /// Returns a vector containing the absolute value of each element of `self`.
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
            }

            /// Returns a vector with signs of `rhs` and the magnitudes of `self`.
            #[inline]
            #[must_use]
            pub fn copysign(self, rhs: Self) -> Self {
                Self::new(
                    self.x.copysign(rhs.x),
                    self.y.copysign(rhs.y),
                    self.z.copysign(rhs.z),
                    self.w.copysign(rhs.w),
                )
            }

            /// Returns `true` if, and only if, all elements are finite.  If any element is either
            /// `NaN`, positive or negative infinity, this will return `false`.
            #[inline]
            #[must_use]
            pub fn is_finite(self) -> $bool {
                $bool::from_raw(self.x.is_finite())
                    & $bool::from_raw(self.y.is_finite())
                    & $bool::from_raw(self.z.is_finite())
                    & $bool::from_raw(self.w.is_finite())
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
                    $bool::from_raw(self.z.is_finite()),
                    $bool::from_raw(self.w.is_finite())
                )
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            #[must_use]
            pub fn is_nan(self) -> $bool {
                $bool::from_raw(self.x.is_nan())
                    | $bool::from_raw(self.y.is_nan())
                    | $bool::from_raw(self.z.is_nan())
                    | $bool::from_raw(self.w.is_nan())
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
                    $bool::from_raw(self.z.is_nan()),
                    $bool::from_raw(self.w.is_nan())
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
                Self::new(self.x.round(), self.y.round(), self.z.round(), self.w.round())
            }

            /// Returns a vector containing the largest integer less than or equal to a number for each
            /// element of `self`.
            #[inline]
            #[must_use]
            pub fn floor(self) -> Self {
                Self::new(self.x.floor(), self.y.floor(), self.z.floor(), self.w.floor())
            }

            /// Returns a vector containing the smallest integer greater than or equal to a number for
            /// each element of `self`.
            #[inline]
            #[must_use]
            pub fn ceil(self) -> Self {
                Self::new(self.x.ceil(), self.y.ceil(), self.z.ceil(), self.w.ceil())
            }

            /// Returns a vector containing the fractional part of the vector as `self - self.floor()`.
            ///
            /// Note that this differs from the Rust implementation of `fract` which returns
            /// `self - self.trunc()`.
            ///
            /// Note that this is fast but not precise for large numbers.
            #[inline]
            #[must_use]
            pub fn fract_gl(self) -> Self {
                self - self.floor()
            }

            /// Returns a vector containing `e^self` (the exponential function) for each element of
            /// `self`.
            #[inline]
            #[must_use]
            pub fn exp(self) -> Self {
                Self::new(self.x.exp(), self.y.exp(), self.z.exp(), self.w.exp())
            }

            /// Returns a vector containing each element of `self` raised to the power of `n`.
            #[inline]
            #[must_use]
            pub fn powf(self, n: $t) -> Self {
                Self::new(
                    self.x.$pow(n),
                    self.y.$pow(n),
                    self.z.$pow(n),
                    self.w.$pow(n),
                )
            }

            /// Returns a vector containing the reciprocal `1.0 / n` of each element of `self`.
            #[inline]
            #[must_use]
            pub fn recip(self) -> Self {
                Self::new($t::ONE / self.x, $t::ONE / self.y, $t::ONE / self.z, $t::ONE / self.w)
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
                self.sub(rhs).abs().cmple(Self::new(max_abs_diff, max_abs_diff, max_abs_diff, max_abs_diff)).all()
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
                    self.z.mul_add(a.z, b.z),
                    self.w.mul_add(a.w, b.w),
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
                Self::new(comps[0], comps[1], comps[2], comps[3])
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
                Self::new(comps.0, comps.1, comps.2, comps.3)
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
            #[inline]
            fn add(self, rhs: $n) -> Self {
                $n::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: $n) {
                self.x += rhs.x;
                self.y += rhs.y;
                self.z += rhs.z;
                self.w += rhs.w;
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $n) -> Self {
                $n::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: $n) {
                self.x -= rhs.x;
                self.y -= rhs.y;
                self.z -= rhs.z;
                self.w -= rhs.w;
            }
        }

        impl Mul for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $n) -> Self {
                $n::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z, self.w * rhs.w)
            }
        }

        impl Mul<$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> $n {
                $n::new(self * rhs.x, self * rhs.y, self * rhs.z, self * rhs.w)
            }
        }

        impl Mul<$t> for $n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $t) -> $n {
                $n::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
            }
        }

        impl MulAssign for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $n) {
                self.x *= rhs.x;
                self.y *= rhs.y;
                self.z *= rhs.z;
                self.w *= rhs.w;
            }
        }

        impl MulAssign<$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                self.x *= rhs;
                self.y *= rhs;
                self.z *= rhs;
                self.w *= rhs;
            }
        }

        impl Div for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $n) -> Self {
                $n::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z, self.w / rhs.w)
            }
        }

        impl Div<$t> for $n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $t) -> $n {
                $n::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
            }
        }

        impl DivAssign for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $n) {
                self.x /= rhs.x;
                self.y /= rhs.y;
                self.z /= rhs.z;
                self.w /= rhs.w;
            }
        }

        impl DivAssign<$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                self.x /= rhs;
                self.y /= rhs;
                self.z /= rhs;
                self.w /= rhs;
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
                    2 => &self.z,
                    3 => &self.w,
                    i => panic!("Invalid index {i} for vector of type: {}", core::any::type_name::<$n>()),
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
        )+
    };
}

#[cfg(feature = "f32")]
impl From<Vec4x4> for [Vec4; 4] {
    #[inline]
    fn from(v: Vec4x4) -> Self {
        let xs: [f32; 4] = v.x.into();
        let ys: [f32; 4] = v.y.into();
        let zs: [f32; 4] = v.z.into();
        let ws: [f32; 4] = v.w.into();
        [
            Vec4::new(xs[0], ys[0], zs[0], ws[0]),
            Vec4::new(xs[1], ys[1], zs[1], ws[1]),
            Vec4::new(xs[2], ys[2], zs[2], ws[2]),
            Vec4::new(xs[3], ys[3], zs[3], ws[3]),
        ]
    }
}

#[cfg(feature = "f32")]
impl From<[Vec4; 4]> for Vec4x4 {
    #[inline]
    fn from(vecs: [Vec4; 4]) -> Self {
        Self {
            x: f32x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f32x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
            z: f32x4::from([vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z]),
            w: f32x4::from([vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w]),
        }
    }
}

#[cfg(feature = "f32")]
impl From<Vec4x8> for [Vec4; 8] {
    #[inline]
    fn from(v: Vec4x8) -> Self {
        let xs: [f32; 8] = v.x.into();
        let ys: [f32; 8] = v.y.into();
        let zs: [f32; 8] = v.z.into();
        let ws: [f32; 8] = v.w.into();
        [
            Vec4::new(xs[0], ys[0], zs[0], ws[0]),
            Vec4::new(xs[1], ys[1], zs[1], ws[1]),
            Vec4::new(xs[2], ys[2], zs[2], ws[2]),
            Vec4::new(xs[3], ys[3], zs[3], ws[3]),
            Vec4::new(xs[4], ys[4], zs[4], ws[4]),
            Vec4::new(xs[5], ys[5], zs[5], ws[5]),
            Vec4::new(xs[6], ys[6], zs[6], ws[6]),
            Vec4::new(xs[7], ys[7], zs[7], ws[7]),
        ]
    }
}

#[cfg(feature = "f32")]
impl From<[Vec4; 8]> for Vec4x8 {
    #[inline]
    fn from(vecs: [Vec4; 8]) -> Self {
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
impl From<DVec4x2> for [DVec4; 2] {
    #[inline]
    fn from(v: DVec4x2) -> Self {
        let xs: [f64; 2] = v.x.into();
        let ys: [f64; 2] = v.y.into();
        let zs: [f64; 2] = v.z.into();
        let ws: [f64; 2] = v.w.into();
        [
            DVec4::new(xs[0], ys[0], zs[0], ws[0]),
            DVec4::new(xs[1], ys[1], zs[1], ws[1]),
        ]
    }
}

#[cfg(feature = "f64")]
impl From<[DVec4; 2]> for DVec4x2 {
    #[inline]
    fn from(vecs: [DVec4; 2]) -> Self {
        Self {
            x: f64x2::from([vecs[0].x, vecs[1].x]),
            y: f64x2::from([vecs[0].y, vecs[1].y]),
            z: f64x2::from([vecs[0].z, vecs[1].z]),
            w: f64x2::from([vecs[0].w, vecs[1].w]),
        }
    }
}

#[cfg(feature = "f64")]
impl From<DVec4x4> for [DVec4; 4] {
    #[inline]
    fn from(v: DVec4x4) -> Self {
        let xs: [f64; 4] = v.x.into();
        let ys: [f64; 4] = v.y.into();
        let zs: [f64; 4] = v.z.into();
        let ws: [f64; 4] = v.w.into();
        [
            DVec4::new(xs[0], ys[0], zs[0], ws[0]),
            DVec4::new(xs[1], ys[1], zs[1], ws[1]),
            DVec4::new(xs[2], ys[2], zs[2], ws[2]),
            DVec4::new(xs[3], ys[3], zs[3], ws[3]),
        ]
    }
}

#[cfg(feature = "f64")]
impl From<[DVec4; 4]> for DVec4x4 {
    #[inline]
    fn from(vecs: [DVec4; 4]) -> Self {
        Self {
            x: f64x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f64x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
            z: f64x4::from([vecs[0].z, vecs[1].z, vecs[2].z, vecs[3].z]),
            w: f64x4::from([vecs[0].w, vecs[1].w, vecs[2].w, vecs[3].w]),
        }
    }
}

#[cfg(feature = "f32")]
wide_vec4s!(
    (Vec3x4, Vec4, Vec4x4, BVec4x4) => (f32, f32x4, boolf32x4, pow_f32x4),
    (Vec3x8, Vec4, Vec4x8, BVec4x8) => (f32, f32x8, boolf32x8, pow_f32x8)
);

#[cfg(feature = "f64")]
wide_vec4s!(
    (DVec3x2, DVec4, DVec4x2, BDVec4x2) => (f64, f64x2, boolf64x2, pow_f64x2),
    (DVec3x4, DVec4, DVec4x4, BDVec4x4) => (f64, f64x4, boolf64x4, pow_f64x4)
);
