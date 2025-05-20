use bevy_math::{DVec2, Vec2};
use core::ops::*;
use wide::{CmpGt, f32x4, f32x8, f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::vec3::{DVec3x2, DVec3x4};
use crate::vec3::{Vec3x4, Vec3x8};

macro_rules! vec2s {
    ($(($nonwiden:ident, $n:ident, $v3t:ident) => ($nonwidet:ident, $t:ident)),+) => {
        $(
        /// A 2-dimensional wide vector.
        #[derive(Clone, Copy, Debug, Default)]
        #[repr(C)]
        pub struct $n {
            /// The X component of the vector.
            pub x: $t,
            /// The Y component of the vector.
            pub y: $t,
        }

        impl $n {
            /// All zeros.
            pub const ZERO: Self = Self::new($t::ZERO, $t::ZERO);

            /// All ones.
            pub const ONE: Self = Self::new($t::ONE, $t::ONE);

            /// A unit vector pointing along the positive X axis.
            pub const X: Self = Self::new($t::ONE, $t::ZERO);

            /// A unit vector pointing along the positive Y axis.
            pub const Y: Self = Self::new($t::ZERO, $t::ONE);

            /// The unit axes.
            pub const AXES: [Self; 2] = [Self::X, Self::Y];

            /// Creates a new vector.
            #[inline(always)]
            #[must_use]
            pub const fn new(x: $t, y: $t) -> Self {
                Self { x, y }
            }

            /// Creates a new vector with all lanes set to the same `x` and `y` values.
            #[inline(always)]
            #[must_use]
            pub fn new_splat(x: $nonwidet, y: $nonwidet) -> Self {
                Self {
                    x: $t::splat(x),
                    y: $t::splat(y),
                }
            }

            /// Creates a new vector with all elements set to `v`.
            #[inline(always)]
            #[must_use]
            pub fn splat(v: $nonwiden) -> Self {
                Self {
                    x: $t::splat(v.x),
                    y: $t::splat(v.y),
                }
            }

            /// Blend two vectors together lanewise using `mask` as a mask.
            ///
            /// This is essentially a bitwise blend operation, such that any point where
            /// there is a 1 bit in `mask`, the output will put the bit from `if_true`, while
            /// where there is a 0 bit in `mask`, the output will put the bit from `if_false`.
            #[inline(always)]
            #[must_use]
            pub fn blend(mask: $t, if_true: Self, if_false: Self) -> Self {
                Self {
                    x: mask.blend(if_true.x, if_false.x),
                    y: mask.blend(if_true.y, if_false.y),
                }
            }

            /// Returns a vector containing each element of `self` modified by a mapping function `f`.
            #[inline(always)]
            #[must_use]
            pub fn map<F>(self, f: F) -> Self
            where
                F: Fn($t) -> $t,
            {
                Self::new(f(self.x), f(self.y))
            }

            /// Creates a new vector from an array.
            #[inline(always)]
            #[must_use]
            pub const fn from_array(arr: [$t; 2]) -> Self {
                Self::new(arr[0], arr[1])
            }

            /// Returns the vector as an array.
            #[inline(always)]
            #[must_use]
            pub const fn to_array(self) -> [$t; 2] {
                [self.x, self.y]
            }

            /// Creates a vector from the first 2 values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 2 elements long.
            #[inline(always)]
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
            #[inline(always)]
            pub fn write_to_slice(self, slice: &mut [$t]) {
                slice[..2].copy_from_slice(&self.to_array());
            }

            /// Creates a 3D vector from `self` and the given `z` value.
            #[inline(always)]
            #[must_use]
            pub const fn extend(self, z: $t) -> $v3t {
                $v3t::new(self.x, self.y, z)
            }

            /// Creates a 2D vector from `self` with the given value of `x`.
            #[inline(always)]
            #[must_use]
            pub fn with_x(self, x: $t) -> Self {
                Self::new(x, self.y)
            }

            /// Creates a 2D vector from `self` with the given value of `y`.
            #[inline(always)]
            #[must_use]
            pub fn with_y(self, y: $t) -> Self {
                Self::new(self.x, y)
            }

            /// Computes the dot product of `self` and `rhs`.
            #[inline(always)]
            #[must_use]
            pub fn dot(self, rhs: Self) -> $t {
                self.x * rhs.x + self.y * rhs.y
            }

            /// Returns a vector where every component is the dot product of `self` and `rhs`.
            #[inline(always)]
            #[must_use]
            pub fn dot_into_vec(self, rhs: Self) -> Self {
                let dot = self.dot(rhs);
                Self::new(dot, dot)
            }

            /// Returns a vector containing the minimum values for each element of `self` and `rhs`.
            ///
            /// In other words this computes `[min(x, rhs.x), min(self.y, rhs.y), ..]`.
            #[inline(always)]
            #[must_use]
            pub fn min(self, rhs: Self) -> Self {
                Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
            }

            /// Returns a vector containing the maximum values for each element of `self` and `rhs`.
            ///
            /// In other words this computes `[max(self.x, rhs.x), max(self.y, rhs.y), ..]`.
            #[inline(always)]
            #[must_use]
            pub fn max(self, rhs: Self) -> Self {
                Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
            }

            /// Component-wise clamping of values.
            ///
            /// Each element in `min` must be less-or-equal to the corresponding element in `max`.
            #[inline(always)]
            #[must_use]
            pub fn clamp(mut self, min: Self, max: Self) -> Self {
                self.x = self.x.max(min.x).min(max.x);
                self.y = self.y.max(min.y).min(max.y);
                self
            }

            /// Returns the horizontal minimum of `self`.
            ///
            /// In other words this computes `min(x, y, ..)`.
            #[inline(always)]
            #[must_use]
            pub fn min_element(self) -> $t {
                self.x.min(self.y)
            }

            /// Returns the horizontal maximum of `self`.
            ///
            /// In other words this computes `max(x, y, ..)`.
            #[inline(always)]
            #[must_use]
            pub fn max_element(self) -> $t {
                self.x.max(self.y)
            }

            /// Returns a vector containing the absolute value of each element of `self`.
            #[inline(always)]
            #[must_use]
            pub fn abs(self) -> Self {
                Self::new(self.x.abs(), self.y.abs())
            }

            /// Returns a vector with signs of `rhs` and the magnitudes of `self`.
            #[inline(always)]
            #[must_use]
            pub fn copysign(self, rhs: Self) -> Self {
                Self::new(self.x.copysign(rhs.x), self.y.copysign(rhs.y))
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
                self.dot(self)
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
                self.mul(self.length_recip())
            }

            /// Returns the vector projection of `self` onto `rhs`.
            ///
            /// `rhs` must be of non-zero length.
            #[inline(always)]
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
            #[inline(always)]
            #[must_use]
            pub fn reject_from(self, rhs: Self) -> Self {
                self - self.project_onto(rhs)
            }

            /// Returns the vector projection of `self` onto `rhs`.
            ///
            /// `rhs` must be normalized.
            #[inline(always)]
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
            #[inline(always)]
            #[must_use]
            pub fn reject_from_normalized(self, rhs: Self) -> Self {
                self - self.project_onto_normalized(rhs)
            }

            /// Returns a vector containing the nearest integer to a number for each element of `self`.
            /// Round half-way cases away from `0.0`.
            #[inline(always)]
            #[must_use]
            pub fn round(self) -> Self {
                Self::new(self.x.round(), self.y.round())
            }

            /// Returns a vector containing the largest integer less than or equal to a number for each
            /// element of `self`.
            #[inline(always)]
            #[must_use]
            pub fn floor(self) -> Self {
                Self::new(self.x.floor(), self.y.floor())
            }

            /// Returns a vector containing the smallest integer greater than or equal to a number for
            /// each element of `self`.
            #[inline(always)]
            #[must_use]
            pub fn ceil(self) -> Self {
                Self::new(self.x.ceil(), self.y.ceil())
            }

            /// Returns a vector containing the reciprocal `1.0 / n` of each element of `self`.
            #[inline(always)]
            #[must_use]
            pub fn recip(self) -> Self {
                Self::new($t::ONE / self.x, $t::ONE / self.y)
            }

            /// Performs a linear interpolation between `self` and `rhs` based on the value `s`.
            ///
            /// When `s` is `0.0`, the result will be equal to `self`.  When `s` is `1.0`, the result
            /// will be equal to `rhs`. When `s` is outside of range `[0, 1]`, the result is linearly
            /// extrapolated.
            #[inline(always)]
            #[must_use]
            pub fn lerp(self, rhs: Self, s: $t) -> Self {
                self * ($t::ONE - s) + rhs * s
            }

            /// Fused multiply-add. Computes `(self * a) + b` element-wise with only one rounding
            /// error, yielding a more accurate result than an unfused multiply-add.
            ///
            /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
            /// architecture has a dedicated fma CPU instruction. However, this is not always true,
            /// and will be heavily dependant on designing algorithms with specific target hardware in
            /// mind.
            #[inline(always)]
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
            #[inline(always)]
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
                let mask = k.cmp_gt($t::splat(0.0));

                let out = self * eta - (eta * ndi + k.sqrt()) * n;

                Self::blend(mask, Self::ZERO, out)
            }

            /// Returns a vector that is equal to `self` rotated by 90 degrees.
            #[inline(always)]
            #[must_use]
            pub fn perp(self) -> Self {
                Self::new(-self.y, self.x)
            }

            /// The perpendicular dot product of `self` and `rhs`.
            /// Also known as the wedge product, 2D cross product, and determinant.
            #[doc(alias = "wedge")]
            #[doc(alias = "cross")]
            #[doc(alias = "determinant")]
            #[inline(always)]
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
                    i => panic!("Invalid index {i} for vector of type: {}", std::any::type_name::<$n>()),
                }
            }
        }

        impl IndexMut<usize> for $n {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                match index {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    i => panic!("Invalid index {i} for vector of type: {}", std::any::type_name::<$n>()),
                }
            }
        }

        impl std::iter::Sum<$n> for $n {
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

impl From<[Vec2; 4]> for Vec2x4 {
    #[inline]
    fn from(vecs: [Vec2; 4]) -> Self {
        Self {
            x: f32x4::from([vecs[0].x, vecs[1].x, vecs[2].x, vecs[3].x]),
            y: f32x4::from([vecs[0].y, vecs[1].y, vecs[2].y, vecs[3].y]),
        }
    }
}

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

vec2s!(
    (Vec2, Vec2x4, Vec3x4) => (f32, f32x4),
    (Vec2, Vec2x8, Vec3x8) => (f32, f32x8)
);

#[cfg(feature = "f64")]
vec2s!(
    (DVec2, DVec2x2, DVec3x2) => (f64, f64x2),
    (DVec2, DVec2x4, DVec3x4) => (f64, f64x4)
);
