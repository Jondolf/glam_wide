use core::{
    fmt::{Debug, Display},
    ops::*,
};
#[cfg(feature = "f32")]
use glam::IVec2;
use glam::{DVec2, Vec2};
use wide::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe, f32x4, f32x8, f64x2, f64x4};
#[cfg(feature = "f32")]
use wide::{i32x4, i64x4};

use crate::SimdLaneCount;
#[cfg(feature = "f64")]
use crate::{BDVec2x2, BDVec2x4, DVec3x2, DVec3x4, boolf64x2, boolf64x4};
#[cfg(feature = "f32")]
use crate::{BVec2x4, BVec2x8, Vec3x4, Vec3x8, boolf32x4, boolf32x8};

macro_rules! wide_ivec2s {
    ($(($nonwiden:ident, $n:ident, $v3t:ident, $bvt:ident, $uvec2:ident) => ($nonwidet:ident, $t:ident, $bool:ident); $($int_type:ident),+; $($int_vec_type:ident),+),+) => {
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

            /// A unit vector pointing along the positive X axis.
            pub const X: Self = Self::new_splat(1, 0);

            /// A unit vector pointing along the negative X axis.
            pub const NEG_X: Self = Self::new_splat(-1, 0);

            /// A unit vector pointing along the positive Y axis.
            pub const Y: Self = Self::new_splat(0, 1);

            /// A unit vector pointing along the negative Y axis.
            pub const NEG_Y: Self = Self::new_splat(0, -1);

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

            /// Creates a new vector with all components set to `f`.
            #[inline]
            #[must_use]
            pub const fn broadcast(f: $t) -> Self {
                Self {
                    x: f,
                    y: f,
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

            /// Computes the squared length of `self`.
            ///
            /// This is faster than `length()` as it avoids a square root operation.
            #[inline]
            #[must_use]
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }

            /// Compute the squared euclidean distance between two points in space.
            #[inline]
            #[must_use]
            pub fn distance_squared(self, rhs: Self) -> $t {
                (self - rhs).length_squared()
            }

            /// Computes the [manhattan distance] between two points.
            ///
            /// # Overflow
            ///
            /// This method may overflow if the result is greater than [`$t::MAX`].
            ///
            /// See also [`checked_manhattan_distance`][$n::checked_manhattan_distance].
            ///
            /// [manhattan distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
            #[inline]
            #[must_use]
            pub fn manhattan_distance(self, rhs: Self) -> $t {
                self.x.abs_diff(rhs.x) + self.y.abs_diff(rhs.y)
            }

            /// Computes the [manhattan distance] between two points.
            ///
            /// This will returns [`None`] if the result is greater than [`$t::MAX`].
            ///
            /// [manhattan distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
            #[inline]
            #[must_use]
            pub fn checked_manhattan_distance(self, rhs: Self) -> Option<$t> {
                let d = self.x.abs_diff(rhs.x);
                d.checked_add(self.y.abs_diff(rhs.y))
            }

            /// Computes the [chebyshev distance] between two points.
            ///
            /// [chebyshev distance]: https://en.wikipedia.org/wiki/Chebyshev_distance
            #[inline]
            #[must_use]
            pub fn chebyshev_distance(self, rhs: Self) -> $t {
                self.x.abs_diff(rhs.x).max(self.y.abs_diff(rhs.y))
            }

            /// Returns a vector that is equal to `self` rotated by 90 degrees.
            #[inline]
            #[must_use]
            pub fn perp(self) -> Self {
                Self {
                    x: -self.y,
                    y: self.x,
                }
            }

            /// The perpendicular dot product of `self` and `rhs`.
            /// Also known as the wedge product, 2D cross product, and determinant.
            #[doc(alias = "wedge")]
            #[doc(alias = "cross")]
            #[doc(alias = "determinant")]
            #[inline]
            #[must_use]
            pub fn perp_dot(self, rhs: Self) -> $t {
                (self.x * rhs.y) - (self.y * rhs.x)
            }

            /// Returns `rhs` rotated by the angle of `self`. If `self` is normalized,
            /// then this just rotation. This is what you usually want. Otherwise,
            /// it will be like a rotation with a multiplication by `self`'s length.
            #[inline]
            #[must_use]
            pub fn rotate(self, rhs: Self) -> Self {
                Self {
                    x: self.x * rhs.x - self.y * rhs.y,
                    y: self.y * rhs.x + self.x * rhs.y,
                }
            }

            /// Returns a vector containing the wrapping addition of `self` and `rhs`.
            ///
            /// In other words this computes `Some([self.x + rhs.x, self.y + rhs.y, ..])` but returns `None` on any overflow.
            #[inline]
            #[must_use]
            pub const fn checked_add(self, rhs: Self) -> Option<Self> {
                let x = match self.x.checked_add(rhs.x) {
                    Some(v) => v,
                    None => return None,
                };
                let y = match self.y.checked_add(rhs.y) {
                    Some(v) => v,
                    None => return None,
                };

                Some(Self { x, y })
            }

            /// Returns a vector containing the wrapping subtraction of `self` and `rhs`.
            ///
            /// In other words this computes `Some([self.x - rhs.x, self.y - rhs.y, ..])` but returns `None` on any overflow.
            #[inline]
            #[must_use]
            pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
                let x = match self.x.checked_sub(rhs.x) {
                    Some(v) => v,
                    None => return None,
                };
                let y = match self.y.checked_sub(rhs.y) {
                    Some(v) => v,
                    None => return None,
                };

                Some(Self { x, y })
            }

            /// Returns a vector containing the wrapping multiplication of `self` and `rhs`.
            ///
            /// In other words this computes `Some([self.x * rhs.x, self.y * rhs.y, ..])` but returns `None` on any overflow.
            #[inline]
            #[must_use]
            pub const fn checked_mul(self, rhs: Self) -> Option<Self> {
                let x = match self.x.checked_mul(rhs.x) {
                    Some(v) => v,
                    None => return None,
                };
                let y = match self.y.checked_mul(rhs.y) {
                    Some(v) => v,
                    None => return None,
                };

                Some(Self { x, y })
            }

            /// Returns a vector containing the wrapping division of `self` and `rhs`.
            ///
            /// In other words this computes `Some([self.x / rhs.x, self.y / rhs.y, ..])` but returns `None` on any division by zero.
            #[inline]
            #[must_use]
            pub const fn checked_div(self, rhs: Self) -> Option<Self> {
                let x = match self.x.checked_div(rhs.x) {
                    Some(v) => v,
                    None => return None,
                };
                let y = match self.y.checked_div(rhs.y) {
                    Some(v) => v,
                    None => return None,
                };

                Some(Self { x, y })
            }

            /// Returns a vector containing the wrapping addition of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.wrapping_add(rhs.x), self.y.wrapping_add(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn wrapping_add(self, rhs: Self) -> Self {
                Self {
                    x: self.x.wrapping_add(rhs.x),
                    y: self.y.wrapping_add(rhs.y),
                }
            }

            /// Returns a vector containing the wrapping subtraction of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.wrapping_sub(rhs.x), self.y.wrapping_sub(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn wrapping_sub(self, rhs: Self) -> Self {
                Self {
                    x: self.x.wrapping_sub(rhs.x),
                    y: self.y.wrapping_sub(rhs.y),
                }
            }

            /// Returns a vector containing the wrapping multiplication of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.wrapping_mul(rhs.x), self.y.wrapping_mul(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn wrapping_mul(self, rhs: Self) -> Self {
                Self {
                    x: self.x.wrapping_mul(rhs.x),
                    y: self.y.wrapping_mul(rhs.y),
                }
            }

            /// Returns a vector containing the wrapping division of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.wrapping_div(rhs.x), self.y.wrapping_div(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn wrapping_div(self, rhs: Self) -> Self {
                Self {
                    x: self.x.wrapping_div(rhs.x),
                    y: self.y.wrapping_div(rhs.y),
                }
            }

            /// Returns a vector containing the saturating addition of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.saturating_add(rhs.x), self.y.saturating_add(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn saturating_add(self, rhs: Self) -> Self {
                Self {
                    x: self.x.saturating_add(rhs.x),
                    y: self.y.saturating_add(rhs.y),
                }
            }

            /// Returns a vector containing the saturating subtraction of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.saturating_sub(rhs.x), self.y.saturating_sub(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn saturating_sub(self, rhs: Self) -> Self {
                Self {
                    x: self.x.saturating_sub(rhs.x),
                    y: self.y.saturating_sub(rhs.y),
                }
            }

            /// Returns a vector containing the saturating multiplication of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.saturating_mul(rhs.x), self.y.saturating_mul(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn saturating_mul(self, rhs: Self) -> Self {
                Self {
                    x: self.x.saturating_mul(rhs.x),
                    y: self.y.saturating_mul(rhs.y),
                }
            }

            /// Returns a vector containing the saturating division of `self` and `rhs`.
            ///
            /// In other words this computes `[self.x.saturating_div(rhs.x), self.y.saturating_div(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn saturating_div(self, rhs: Self) -> Self {
                Self {
                    x: self.x.saturating_div(rhs.x),
                    y: self.y.saturating_div(rhs.y),
                }
            }

            /// Returns a vector containing the wrapping addition of `self` and unsigned vector `rhs`.
            ///
            /// In other words this computes `Some([self.x + rhs.x, self.y + rhs.y, ..])` but returns `None` on any overflow.
            #[inline]
            #[must_use]
            pub const fn checked_add_unsigned(self, rhs: $uvec2) -> Option<Self> {
                let x = match self.x.checked_add_unsigned(rhs.x) {
                    Some(v) => v,
                    None => return None,
                };
                let y = match self.y.checked_add_unsigned(rhs.y) {
                    Some(v) => v,
                    None => return None,
                };

                Some(Self { x, y })
            }

            /// Returns a vector containing the wrapping subtraction of `self` and unsigned vector `rhs`.
            ///
            /// In other words this computes `Some([self.x - rhs.x, self.y - rhs.y, ..])` but returns `None` on any overflow.
            #[inline]
            #[must_use]
            pub const fn checked_sub_unsigned(self, rhs: $uvec2) -> Option<Self> {
                let x = match self.x.checked_sub_unsigned(rhs.x) {
                    Some(v) => v,
                    None => return None,
                };
                let y = match self.y.checked_sub_unsigned(rhs.y) {
                    Some(v) => v,
                    None => return None,
                };

                Some(Self { x, y })
            }

            /// Returns a vector containing the wrapping addition of `self` and unsigned vector `rhs`.
            ///
            /// In other words this computes `[self.x.wrapping_add_unsigned(rhs.x), self.y.wrapping_add_unsigned(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn wrapping_add_unsigned(self, rhs: $uvec2) -> Self {
                Self {
                    x: self.x.wrapping_add_unsigned(rhs.x),
                    y: self.y.wrapping_add_unsigned(rhs.y),
                }
            }

            /// Returns a vector containing the wrapping subtraction of `self` and unsigned vector `rhs`.
            ///
            /// In other words this computes `[self.x.wrapping_sub_unsigned(rhs.x), self.y.wrapping_sub_unsigned(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn wrapping_sub_unsigned(self, rhs: $uvec2) -> Self {
                Self {
                    x: self.x.wrapping_sub_unsigned(rhs.x),
                    y: self.y.wrapping_sub_unsigned(rhs.y),
                }
            }

            // Returns a vector containing the saturating addition of `self` and unsigned vector `rhs`.
            ///
            /// In other words this computes `[self.x.saturating_add_unsigned(rhs.x), self.y.saturating_add_unsigned(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn saturating_add_unsigned(self, rhs: $uvec2) -> Self {
                Self {
                    x: self.x.saturating_add_unsigned(rhs.x),
                    y: self.y.saturating_add_unsigned(rhs.y),
                }
            }

            /// Returns a vector containing the saturating subtraction of `self` and unsigned vector `rhs`.
            ///
            /// In other words this computes `[self.x.saturating_sub_unsigned(rhs.x), self.y.saturating_sub_unsigned(rhs.y), ..]`.
            #[inline]
            #[must_use]
            pub const fn saturating_sub_unsigned(self, rhs: $uvec2) -> Self {
                Self {
                    x: self.x.saturating_sub_unsigned(rhs.x),
                    y: self.y.saturating_sub_unsigned(rhs.y),
                }
            }
        }

        impl Div for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                Self {
                    x: self.x.div(rhs.x),
                    y: self.y.div(rhs.y),
                }
            }
        }

        impl Div<&Self> for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: &Self) -> Self {
                self.div(*rhs)
            }
        }

        impl Div<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$n) -> $n {
                (*self).div(*rhs)
            }
        }

        impl Div<$n> for &$n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $n) -> $n {
                (*self).div(rhs)
            }
        }

        impl DivAssign for $n {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                self.x.div_assign(rhs.x);
                self.y.div_assign(rhs.y);
            }
        }

        impl DivAssign<&Self> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: &Self) {
                self.div_assign(*rhs);
            }
        }

        impl Div<$t> for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $t) -> Self {
                Self {
                    x: self.x.div(rhs),
                    y: self.y.div(rhs),
                }
            }
        }

        impl Div<&$t> for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: &$t) -> Self {
                self.div(*rhs)
            }
        }

        impl Div<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$t) -> $n {
                (*self).div(*rhs)
            }
        }

        impl Div<$t> for &$n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $t) -> $n {
                (*self).div(rhs)
            }
        }

        impl DivAssign<$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                self.x.div_assign(rhs);
                self.y.div_assign(rhs);
            }
        }

        impl DivAssign<&$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: &$t) {
                self.div_assign(*rhs);
            }
        }

        impl Div<$n> for $t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $n) -> $n {
                $n {
                    x: self.div(rhs.x),
                    y: self.div(rhs.y),
                }
            }
        }

        impl Div<&$n> for $t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$n) -> $n {
                self.div(*rhs)
            }
        }

        impl Div<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$n) -> $n {
                (*self).div(*rhs)
            }
        }

        impl Div<$n> for &$t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $n) -> $n {
                (*self).div(rhs)
            }
        }

        impl Mul for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self {
                    x: self.x.mul(rhs.x),
                    y: self.y.mul(rhs.y),
                }
            }
        }

        impl Mul<&Self> for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: &Self) -> Self {
                self.mul(*rhs)
            }
        }

        impl Mul<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$n) -> $n {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$n> for &$n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> $n {
                (*self).mul(rhs)
            }
        }

        impl MulAssign for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                self.x.mul_assign(rhs.x);
                self.y.mul_assign(rhs.y);
            }
        }

        impl MulAssign<&Self> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: &Self) {
                self.mul_assign(*rhs);
            }
        }

        impl Mul<$t> for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $t) -> Self {
                Self {
                    x: self.x.mul(rhs),
                    y: self.y.mul(rhs),
                }
            }
        }

        impl Mul<&$t> for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: &$t) -> Self {
                self.mul(*rhs)
            }
        }

        impl Mul<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$t) -> $n {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$t> for &$n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $t) -> $n {
                (*self).mul(rhs)
            }
        }

        impl MulAssign<$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                self.x.mul_assign(rhs);
                self.y.mul_assign(rhs);
            }
        }

        impl MulAssign<&$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: &$t) {
                self.mul_assign(*rhs);
            }
        }

        impl Mul<$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> $n {
                $n {
                    x: self.mul(rhs.x),
                    y: self.mul(rhs.y),
                }
            }
        }

        impl Mul<&$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$n) -> $n {
                self.mul(*rhs)
            }
        }

        impl Mul<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$n) -> $n {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$n> for &$t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> $n {
                (*self).mul(rhs)
            }
        }

        impl Add for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self {
                    x: self.x.add(rhs.x),
                    y: self.y.add(rhs.y),
                }
            }
        }

        impl Add<&Self> for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: &Self) -> Self {
                self.add(*rhs)
            }
        }

        impl Add<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn add(self, rhs: &$n) -> $n {
                (*self).add(*rhs)
            }
        }

        impl Add<$n> for &$n {
            type Output = $n;
            #[inline]
            fn add(self, rhs: $n) -> $n {
                (*self).add(rhs)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                self.x.add_assign(rhs.x);
                self.y.add_assign(rhs.y);
            }
        }

        impl AddAssign<&Self> for $n {
            #[inline]
            fn add_assign(&mut self, rhs: &Self) {
                self.add_assign(*rhs);
            }
        }

        impl Add<$t> for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: $t) -> Self {
                Self {
                    x: self.x.add(rhs),
                    y: self.y.add(rhs),
                }
            }
        }

        impl Add<&$t> for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: &$t) -> Self {
                self.add(*rhs)
            }
        }

        impl Add<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn add(self, rhs: &$t) -> $n {
                (*self).add(*rhs)
            }
        }

        impl Add<$t> for &$n {
            type Output = $n;
            #[inline]
            fn add(self, rhs: $t) -> $n {
                (*self).add(rhs)
            }
        }

        impl AddAssign<$t> for $n {
            #[inline]
            fn add_assign(&mut self, rhs: $t) {
                self.x.add_assign(rhs);
                self.y.add_assign(rhs);
            }
        }

        impl AddAssign<&$t> for $n {
            #[inline]
            fn add_assign(&mut self, rhs: &$t) {
                self.add_assign(*rhs);
            }
        }

        impl Add<$n> for $t {
            type Output = $n;
            #[inline]
            fn add(self, rhs: $n) -> $n {
                $n {
                    x: self.add(rhs.x),
                    y: self.add(rhs.y),
                }
            }
        }

        impl Add<&$n> for $t {
            type Output = $n;
            #[inline]
            fn add(self, rhs: &$n) -> $n {
                self.add(*rhs)
            }
        }

        impl Add<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn add(self, rhs: &$n) -> $n {
                (*self).add(*rhs)
            }
        }

        impl Add<$n> for &$t {
            type Output = $n;
            #[inline]
            fn add(self, rhs: $n) -> $n {
                (*self).add(rhs)
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self {
                    x: self.x.sub(rhs.x),
                    y: self.y.sub(rhs.y),
                }
            }
        }

        impl Sub<&Self> for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: &Self) -> Self {
                self.sub(*rhs)
            }
        }

        impl Sub<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: &$n) -> $n {
                (*self).sub(*rhs)
            }
        }

        impl Sub<$n> for &$n {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: $n) -> $n {
                (*self).sub(rhs)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                self.x.sub_assign(rhs.x);
                self.y.sub_assign(rhs.y);
            }
        }

        impl SubAssign<&Self> for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: &Self) {
                self.sub_assign(*rhs);
            }
        }

        impl Sub<$t> for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $t) -> Self {
                Self {
                    x: self.x.sub(rhs),
                    y: self.y.sub(rhs),
                }
            }
        }

        impl Sub<&$t> for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: &$t) -> Self {
                self.sub(*rhs)
            }
        }

        impl Sub<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: &$t) -> $n {
                (*self).sub(*rhs)
            }
        }

        impl Sub<$t> for &$n {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: $t) -> $n {
                (*self).sub(rhs)
            }
        }

        impl SubAssign<$t> for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: $t) {
                self.x.sub_assign(rhs);
                self.y.sub_assign(rhs);
            }
        }

        impl SubAssign<&$t> for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: &$t) {
                self.sub_assign(*rhs);
            }
        }

        impl Sub<$n> for $t {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: $n) -> $n {
                $n {
                    x: self.sub(rhs.x),
                    y: self.sub(rhs.y),
                }
            }
        }

        impl Sub<&$n> for $t {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: &$n) -> $n {
                self.sub(*rhs)
            }
        }

        impl Sub<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: &$n) -> $n {
                (*self).sub(*rhs)
            }
        }

        impl Sub<$n> for &$t {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: $n) -> $n {
                (*self).sub(rhs)
            }
        }

        impl Rem for $n {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: Self) -> Self {
                Self {
                    x: self.x.rem(rhs.x),
                    y: self.y.rem(rhs.y),
                }
            }
        }

        impl Rem<&Self> for $n {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: &Self) -> Self {
                self.rem(*rhs)
            }
        }

        impl Rem<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: &$n) -> $n {
                (*self).rem(*rhs)
            }
        }

        impl Rem<$n> for &$n {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: $n) -> $n {
                (*self).rem(rhs)
            }
        }

        impl RemAssign for $n {
            #[inline]
            fn rem_assign(&mut self, rhs: Self) {
                self.x.rem_assign(rhs.x);
                self.y.rem_assign(rhs.y);
            }
        }

        impl RemAssign<&Self> for $n {
            #[inline]
            fn rem_assign(&mut self, rhs: &Self) {
                self.rem_assign(*rhs);
            }
        }

        impl Rem<$t> for $n {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: $t) -> Self {
                Self {
                    x: self.x.rem(rhs),
                    y: self.y.rem(rhs),
                }
            }
        }

        impl Rem<&$t> for $n {
            type Output = Self;
            #[inline]
            fn rem(self, rhs: &$t) -> Self {
                self.rem(*rhs)
            }
        }

        impl Rem<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: &$t) -> $n {
                (*self).rem(*rhs)
            }
        }

        impl Rem<$t> for &$n {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: $t) -> $n {
                (*self).rem(rhs)
            }
        }

        impl RemAssign<$t> for $n {
            #[inline]
            fn rem_assign(&mut self, rhs: $t) {
                self.x.rem_assign(rhs);
                self.y.rem_assign(rhs);
            }
        }

        impl RemAssign<&$t> for $n {
            #[inline]
            fn rem_assign(&mut self, rhs: &$t) {
                self.rem_assign(*rhs);
            }
        }

        impl Rem<$n> for $t {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: $n) -> $n {
                $n {
                    x: self.rem(rhs.x),
                    y: self.rem(rhs.y),
                }
            }
        }

        impl Rem<&$n> for $t {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: &$n) -> $n {
                self.rem(*rhs)
            }
        }

        impl Rem<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: &$n) -> $n {
                (*self).rem(*rhs)
            }
        }

        impl Rem<$n> for &$t {
            type Output = $n;
            #[inline]
            fn rem(self, rhs: $n) -> $n {
                (*self).rem(rhs)
            }
        }

        #[cfg(not(target_arch = "spirv"))]
        impl AsRef<[$t; 2]> for $n {
            #[inline]
            fn as_ref(&self) -> &[$t; 2] {
                unsafe { &*(self as *const Self as *const [$t; 2]) }
            }
        }

        #[cfg(not(target_arch = "spirv"))]
        impl AsMut<[$t; 2]> for $n {
            #[inline]
            fn as_mut(&mut self) -> &mut [$t; 2] {
                unsafe { &mut *(self as *mut Self as *mut [$t; 2]) }
            }
        }

        impl Sum for $n {
            #[inline]
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::ZERO, Self::add)
            }
        }

        impl<'a> Sum<&'a Self> for $n {
            #[inline]
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
            }
        }

        impl Product for $n {
            #[inline]
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::ONE, Self::mul)
            }
        }

        impl<'a> Product<&'a Self> for $n {
            #[inline]
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::ONE, |a, &b| Self::mul(a, b))
            }
        }

        impl Neg for $n {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self {
                    x: self.x.neg(),
                    y: self.y.neg(),
                }
            }
        }

        impl Neg for &$n {
            type Output = $n;
            #[inline]
            fn neg(self) -> $n {
                (*self).neg()
            }
        }

        impl Not for $n {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                Self {
                    x: self.x.not(),
                    y: self.y.not(),
                }
            }
        }

        impl Not for &$n {
            type Output = $n;
            #[inline]
            fn not(self) -> $n {
                (*self).not()
            }
        }

        impl BitAnd for $n {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x.bitand(rhs.x),
                    y: self.y.bitand(rhs.y),
                }
            }
        }

        impl BitAnd<&Self> for $n {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: &Self) -> Self {
                self.bitand(*rhs)
            }
        }

        impl BitAnd<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitand(self, rhs: &$n) -> $n {
                (*self).bitand(*rhs)
            }
        }

        impl BitAnd<$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitand(self, rhs: $n) -> $n {
                (*self).bitand(rhs)
            }
        }

        impl BitAndAssign for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.bitand(rhs);
            }
        }

        impl BitAndAssign<&Self> for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: &Self) {
                self.bitand_assign(*rhs);
            }
        }

        impl BitOr for $n {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x.bitor(rhs.x),
                    y: self.y.bitor(rhs.y),
                }
            }
        }

        impl BitOr<&Self> for $n {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: &Self) -> Self {
                self.bitor(*rhs)
            }
        }

        impl BitOr<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitor(self, rhs: &$n) -> $n {
                (*self).bitor(*rhs)
            }
        }

        impl BitOr<$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitor(self, rhs: $n) -> $n {
                (*self).bitor(rhs)
            }
        }

        impl BitOrAssign for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.bitor(rhs);
            }
        }

        impl BitOrAssign<&Self> for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: &Self) {
                self.bitor_assign(*rhs);
            }
        }

        impl BitXor for $n {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self {
                    x: self.x.bitxor(rhs.x),
                    y: self.y.bitxor(rhs.y),
                }
            }
        }

        impl BitXor<&Self> for $n {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: &Self) -> Self {
                self.bitxor(*rhs)
            }
        }

        impl BitXor<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitxor(self, rhs: &$n) -> $n {
                (*self).bitxor(*rhs)
            }
        }

        impl BitXor<$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitxor(self, rhs: $n) -> $n {
                (*self).bitxor(rhs)
            }
        }

        impl BitXorAssign for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.bitxor(rhs);
            }
        }

        impl BitXorAssign<&Self> for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: &Self) {
                self.bitxor_assign(*rhs);
            }
        }

        impl BitAnd<$t> for $n {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: $t) -> Self::Output {
                Self {
                    x: self.x.bitand(rhs),
                    y: self.y.bitand(rhs),
                }
            }
        }

        impl BitAnd<&$t> for $n {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: &$t) -> Self {
                self.bitand(*rhs)
            }
        }

        impl BitAnd<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn bitand(self, rhs: &$t) -> $n {
                (*self).bitand(*rhs)
            }
        }

        impl BitAnd<$t> for &$n {
            type Output = $n;
            #[inline]
            fn bitand(self, rhs: $t) -> $n {
                (*self).bitand(rhs)
            }
        }

        impl BitAndAssign<$t> for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: $t) {
                *self = self.bitand(rhs);
            }
        }

        impl BitAndAssign<&$t> for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: &$t) {
                self.bitand_assign(*rhs);
            }
        }

        impl BitOr<$t> for $n {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: $t) -> Self::Output {
                Self {
                    x: self.x.bitor(rhs),
                    y: self.y.bitor(rhs),
                }
            }
        }

        impl BitOr<&$t> for $n {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: &$t) -> Self {
                self.bitor(*rhs)
            }
        }

        impl BitOr<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn bitor(self, rhs: &$t) -> $n {
                (*self).bitor(*rhs)
            }
        }

        impl BitOr<$t> for &$n {
            type Output = $n;
            #[inline]
            fn bitor(self, rhs: $t) -> $n {
                (*self).bitor(rhs)
            }
        }

        impl BitOrAssign<$t> for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: $t) {
                *self = self.bitor(rhs);
            }
        }

        impl BitOrAssign<&$t> for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: &$t) {
                self.bitor_assign(*rhs);
            }
        }

        impl BitXor<$t> for $n {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: $t) -> Self::Output {
                Self {
                    x: self.x.bitxor(rhs),
                    y: self.y.bitxor(rhs),
                }
            }
        }

        impl BitXor<&$t> for $n {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: &$t) -> Self {
                self.bitxor(*rhs)
            }
        }

        impl BitXor<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn bitxor(self, rhs: &$t) -> $n {
                (*self).bitxor(*rhs)
            }
        }

        impl BitXor<$t> for &$n {
            type Output = $n;
            #[inline]
            fn bitxor(self, rhs: $t) -> $n {
                (*self).bitxor(rhs)
            }
        }

        impl BitXorAssign<$t> for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: $t) {
                *self = self.bitxor(rhs);
            }
        }

        impl BitXorAssign<&$t> for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: &$t) {
                self.bitxor_assign(*rhs);
            }
        }

        $(
        impl Shl<$int_type> for $n {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: $int_type) -> Self::Output {
                Self {
                    x: self.x.shl(rhs),
                    y: self.y.shl(rhs),
                }
            }
        }

        impl Shl<&$int_type> for $n {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: &$int_type) -> Self {
                self.shl(*rhs)
            }
        }

        impl Shl<&$int_type> for &$n {
            type Output = $n;
            #[inline]
            fn shl(self, rhs: &$int_type) -> $n {
                (*self).shl(*rhs)
            }
        }

        impl Shl<$int_type> for &$n {
            type Output = $n;
            #[inline]
            fn shl(self, rhs: $int_type) -> $n {
                (*self).shl(rhs)
            }
        }

        impl ShlAssign<$int_type> for $n {
            #[inline]
            fn shl_assign(&mut self, rhs: $int_type) {
                *self = self.shl(rhs);
            }
        }

        impl ShlAssign<&$int_type> for $n {
            #[inline]
            fn shl_assign(&mut self, rhs: &$int_type) {
                self.shl_assign(*rhs);
            }
        }

        impl Shr<$int_type> for $n {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: $int_type) -> Self::Output {
                Self {
                    x: self.x.shr(rhs),
                    y: self.y.shr(rhs),
                }
            }
        }

        impl Shr<&$int_type> for $n {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: &$int_type) -> Self {
                self.shr(*rhs)
            }
        }

        impl Shr<&$int_type> for &$n {
            type Output = $n;
            #[inline]
            fn shr(self, rhs: &$int_type) -> $n {
                (*self).shr(*rhs)
            }
        }

        impl Shr<$int_type> for &$n {
            type Output = $n;
            #[inline]
            fn shr(self, rhs: $int_type) -> $n {
                (*self).shr(rhs)
            }
        }

        impl ShrAssign<$int_type> for $n {
            #[inline]
            fn shr_assign(&mut self, rhs: $int_type) {
                *self = self.shr(rhs);
            }
        }

        impl ShrAssign<&$int_type> for $n {
            #[inline]
            fn shr_assign(&mut self, rhs: &$int_type) {
                self.shr_assign(*rhs);
            }
        }
        )*

        impl Shl for $n {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: Self) -> Self {
                Self {
                    x: self.x.shl(rhs.x),
                    y: self.y.shl(rhs.y),
                }
            }
        }

        impl Shl<&Self> for $n {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: &Self) -> Self {
                self.shl(*rhs)
            }
        }

        impl Shl<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn shl(self, rhs: &$n) -> $n {
                (*self).shl(*rhs)
            }
        }

        impl Shl<$n> for &$n {
            type Output = $n;
            #[inline]
            fn shl(self, rhs: $n) -> $n {
                (*self).shl(rhs)
            }
        }

        impl Shr for $n {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: Self) -> Self {
                Self {
                    x: self.x.shr(rhs.x),
                    y: self.y.shr(rhs.y),
                }
            }
        }

        impl Shr<&Self> for $n {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: &Self) -> Self {
                self.shr(*rhs)
            }
        }

        impl Shr<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn shr(self, rhs: &$n) -> $n {
                (*self).shr(*rhs)
            }
        }

        impl Shr<$n> for &$n {
            type Output = $n;
            #[inline]
            fn shr(self, rhs: $n) -> $n {
                (*self).shr(rhs)
            }
        }

        $(
        impl Shl<$int_vec_type> for $n {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: $int_vec_type) -> Self {
                Self {
                    x: self.x.shl(rhs.x),
                    y: self.y.shl(rhs.y),
                }
            }
        }

        impl Shl<&$int_vec_type> for $n {
            type Output = Self;
            #[inline]
            fn shl(self, rhs: &$int_vec_type) -> Self {
                self.shl(*rhs)
            }
        }

        impl Shl<&$int_vec_type> for &$n {
            type Output = $n;
            #[inline]
            fn shl(self, rhs: &$int_vec_type) -> $n {
                (*self).shl(*rhs)
            }
        }

        impl Shl<$int_vec_type> for &$n {
            type Output = $n;
            #[inline]
            fn shl(self, rhs: $int_vec_type) -> $n {
                (*self).shl(rhs)
            }
        }

        impl Shr<$int_vec_type> for $n {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: $int_vec_type) -> Self {
                Self {
                    x: self.x.shr(rhs.x),
                    y: self.y.shr(rhs.y),
                }
            }
        }

        impl Shr<&$int_vec_type> for $n {
            type Output = Self;
            #[inline]
            fn shr(self, rhs: &$int_vec_type) -> Self {
                self.shr(*rhs)
            }
        }

        impl Shr<&$int_vec_type> for &$n {
            type Output = $n;
            #[inline]
            fn shr(self, rhs: &$int_vec_type) -> $n {
                (*self).shr(*rhs)
            }
        }

        impl Shr<$int_vec_type> for &$n {
            type Output = $n;
            #[inline]
            fn shr(self, rhs: $int_vec_type) -> $n {
                (*self).shr(rhs)
            }
        }
        )*

        impl Index<usize> for $n {
            type Output = $t;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                match index {
                    0 => &self.x,
                    1 => &self.y,
                    _ => panic!("index out of bounds"),
                }
            }
        }

        impl IndexMut<usize> for $n {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                match index {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    _ => panic!("index out of bounds"),
                }
            }
        }

        impl fmt::Display for $n {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "[{}, {}]", self.x, self.y)
            }
        }

        impl fmt::Debug for $n {
            fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt.debug_tuple(stringify!($n))
                    .field(&self.x)
                    .field(&self.y)
                    .finish()
            }
        }

        impl From<[$t; 2]> for $n {
            #[inline]
            fn from(a: [$t; 2]) -> Self {
                Self::new(a[0], a[1])
            }
        }

        impl From<$n> for [$t; 2] {
            #[inline]
            fn from(v: $n) -> Self {
                [v.x, v.y]
            }
        }

        impl From<($t, $t)> for $n {
            #[inline]
            fn from(t: ($t, $t)) -> Self {
                Self::new(t.0, t.1)
            }
        }

        impl From<$n> for ($t, $t) {
            #[inline]
            fn from(v: $n) -> Self {
                (v.x, v.y)
            }
        }

        $(
        impl From<$int_vec_type> for $n {
            #[inline]
            fn from(v: $int_vec_type) -> Self {
                Self::new($t::from(v.x), $t::from(v.y))
            }
        }
        )*

        $(
        impl TryFrom<$int_vec_type> for $n {
            type Error = core::num::TryFromIntError;

            #[inline]
            fn try_from(v: $int_vec_type) -> Result<Self, Self::Error> {
                Ok(Self::new($t::try_from(v.x)?, $t::try_from(v.y)?))
            }
        }
        )*

        impl From<$bvt> for $n {
            #[inline]
            fn from(v: $bvt) -> Self {
                Self::new($t::from(v.x), $t::from(v.y))
            }
        }
        )+
    };
}

#[cfg(feature = "f32")]
wide_ivec2s!(
    (IVec2, IVec2x4, IVec3x4, BVec2x4, UVec2x4) => (i32, i32x4, boolf32x4); i64x4; I64Vec2x4
);
