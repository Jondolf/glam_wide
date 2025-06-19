#[cfg(feature = "f64")]
use bevy_math::DMat2;
use bevy_math::Mat2;
use core::iter::{Product, Sum};
use core::ops::*;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DMat3x2, DMat3x4, DVec2x2, DVec2x4};
use crate::{Mat3x4, Mat3x8, Vec2x4, Vec2x8};

macro_rules! wide_mat2s {
    ($($n:ident => $nonwiden:ident, $m3t:ident, $v3t:ident, $vt:ident, $t:ident),+) => {
        $(
        /// A wide 2x2 column major matrix.
        #[derive(Clone, Copy, Debug)]
        #[repr(C)]
        pub struct $n {
            /// The first column of the matrix.
            pub x_axis: $vt,
            /// The second column of the matrix.
            pub y_axis: $vt,
        }

        impl $n {
            /// A 2x2 matrix with all elements set to `0.0`.
            pub const ZERO: Self = Self::from_cols($vt::ZERO, $vt::ZERO);

            /// A 2x2 identity matrix, where all diagonal elements are `1.0`,
            /// and all off-diagonal elements are `0.0`.
            pub const IDENTITY: Self = Self::from_cols($vt::X, $vt::Y);

            /// All NaNs.
            pub const NAN: Self = Self::from_cols($vt::NAN, $vt::NAN);

            #[inline(always)]
            const fn new(m00: $t, m01: $t, m10: $t, m11: $t) -> Self {
                $n {
                    x_axis: $vt::new(m00, m01),
                    y_axis: $vt::new(m10, m11),
                }
            }

            /// Creates a new 2x2 matrix with all lanes set to `m`.
            #[inline]
            #[must_use]
            pub fn splat(m: $nonwiden) -> Self {
                Self {
                    x_axis: $vt::splat(m.x_axis),
                    y_axis: $vt::splat(m.y_axis),
                }
            }

            /// Creates a 2x2 matrix from two column vectors.
            #[inline(always)]
            #[must_use]
            pub const fn from_cols(x_axis: $vt, y_axis: $vt) -> Self {
                Self { x_axis, y_axis }
            }

            /// Creates a 2x2 matrix from an array stored in column major order.
            /// If your data is stored in row major you will need to `transpose` the returned
            /// matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array(m: &[$t; 4]) -> Self {
                Self::new(m[0], m[1], m[2], m[3])
            }

            /// Creates an array storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline]
            #[must_use]
            pub const fn to_cols_array(&self) -> [$t; 4] {
                [self.x_axis.x, self.x_axis.y, self.y_axis.x, self.y_axis.y]
            }

            /// Creates a 2x2 matrix from a 2D array stored in column major order.
            /// If your data is in row major order you will need to `transpose` the returned
            /// matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array_2d(m: &[[$t; 2]; 2]) -> Self {
                Self::from_cols($vt::from_array(m[0]), $vt::from_array(m[1]))
            }

            /// Creates a 2D array storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline]
            #[must_use]
            pub const fn to_cols_array_2d(&self) -> [[$t; 2]; 2] {
                [self.x_axis.to_array(), self.y_axis.to_array()]
            }

            /// Creates a 2x2 matrix with its diagonal set to `diagonal` and all other entries set to `0.0`.
            #[doc(alias = "scale")]
            #[inline]
            #[must_use]
            pub const fn from_diagonal(diagonal: $vt) -> Self {
                Self::new(diagonal.x, $t::ZERO, $t::ZERO, diagonal.y)
            }

            /// Creates a 2x2 matrix containing the combining non-uniform `scale` and rotation of
            /// `angle` (in radians).
            #[inline]
            #[must_use]
            pub fn from_scale_angle(scale: $vt, angle: $t) -> Self {
                let (sin, cos) = angle.sin_cos();
                Self::new(cos * scale.x, sin * scale.x, -sin * scale.y, cos * scale.y)
            }

            /// Creates a 2x2 matrix containing a rotation of `angle` (in radians).
            #[inline]
            #[must_use]
            pub fn from_angle(angle: $t) -> Self {
                let (sin, cos) = angle.sin_cos();
                Self::new(cos, sin, -sin, cos)
            }

            /// Creates a 2x2 matrix from a 3x3 matrix, discarding the 2nd row and column.
            #[inline]
            #[must_use]
            pub fn from_mat3(m: $m3t) -> Self {
                Self::from_cols(m.x_axis.truncate(), m.y_axis.truncate())
            }

            /// Creates a 2x2 matrix from the first 4 values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline]
            #[must_use]
            pub const fn from_cols_slice(slice: &[$t]) -> Self {
                Self::new(slice[0], slice[1], slice[2], slice[3])
            }

            /// Writes the columns of `self` to the first 4 elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline]
            pub fn write_cols_to_slice(self, slice: &mut [$t]) {
                slice[0] = self.x_axis.x;
                slice[1] = self.x_axis.y;
                slice[2] = self.y_axis.x;
                slice[3] = self.y_axis.y;
            }

            /// Returns the matrix column for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 1.
            #[inline]
            #[must_use]
            pub fn col(&self, index: usize) -> $vt {
                match index {
                    0 => self.x_axis,
                    1 => self.y_axis,
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns a mutable reference to the matrix column for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 1.
            #[inline]
            pub fn col_mut(&mut self, index: usize) -> &mut $vt {
                match index {
                    0 => &mut self.x_axis,
                    1 => &mut self.y_axis,
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns the matrix row for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 1.
            #[inline]
            #[must_use]
            pub fn row(&self, index: usize) -> $vt {
                match index {
                    0 => $vt::new(self.x_axis.x, self.y_axis.x),
                    1 => $vt::new(self.x_axis.y, self.y_axis.y),
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns the transpose of `self`.
            #[inline]
            #[must_use]
            pub fn transpose(&self) -> Self {
                Self {
                    x_axis: $vt::new(self.x_axis.x, self.y_axis.x),
                    y_axis: $vt::new(self.x_axis.y, self.y_axis.y),
                }
            }

            /// Returns the determinant of `self`.
            #[inline]
            #[must_use]
            pub fn determinant(&self) -> $t {
                self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
            }

            /// Returns the inverse of `self`.
            ///
            /// If the matrix is not invertible the returned matrix will be invalid.
            #[inline]
            #[must_use]
            pub fn inverse(&self) -> Self {
                let det = self.determinant();
                let inv_det = $t::ONE / det;
                Self::new(
                    self.y_axis.y * inv_det,
                    self.x_axis.y * -inv_det,
                    self.y_axis.x * -inv_det,
                    self.x_axis.x * inv_det,
                )
            }

            /// Transforms a 2D vector.
            #[inline]
            #[must_use]
            pub fn mul_vec2(&self, rhs: $vt) -> $vt {
                #[allow(clippy::suspicious_operation_groupings)]
                $vt::new(
                    (self.x_axis.x * rhs.x) + (self.y_axis.x * rhs.y),
                    (self.x_axis.y * rhs.x) + (self.y_axis.y * rhs.y),
                )
            }

            /// Multiplies two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn mul_mat2(&self, rhs: &Self) -> Self {
                Self::from_cols(self.mul(rhs.x_axis), self.mul(rhs.y_axis))
            }

            /// Adds two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn add_mat2(&self, rhs: &Self) -> Self {
                Self::from_cols(self.x_axis.add(rhs.x_axis), self.y_axis.add(rhs.y_axis))
            }

            /// Subtracts two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn sub_mat2(&self, rhs: &Self) -> Self {
                Self::from_cols(self.x_axis.sub(rhs.x_axis), self.y_axis.sub(rhs.y_axis))
            }

            /// Multiplies a 2x2 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn mul_scalar(&self, rhs: $t) -> Self {
                Self::from_cols(self.x_axis.mul(rhs), self.y_axis.mul(rhs))
            }

            /// Divides a 2x2 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn div_scalar(&self, rhs: $t) -> Self {
                let rhs = $vt::new(rhs, rhs);
                Self::from_cols(self.x_axis.div(rhs), self.y_axis.div(rhs))
            }

            /// Takes the absolute value of each element in `self`
            #[inline]
            #[must_use]
            pub fn abs(&self) -> Self {
                Self::from_cols(self.x_axis.abs(), self.y_axis.abs())
            }
        }

        impl Default for $n {
            #[inline(always)]
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl Add for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                self.add_mat2(&rhs)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = self.add_mat2(&rhs);
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                self.sub_mat2(&rhs)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.sub_mat2(&rhs);
            }
        }

        impl Neg for $n {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self::from_cols(self.x_axis.neg(), self.y_axis.neg())
            }
        }

        impl Mul for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                self.mul_mat2(&rhs)
            }
        }

        impl MulAssign for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = self.mul_mat2(&rhs);
            }
        }

        impl Mul<$vt> for $n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: $vt) -> Self::Output {
                self.mul_vec2(rhs)
            }
        }

        impl Mul<$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                rhs.mul_scalar(self)
            }
        }

        impl Mul<$t> for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $t) -> Self::Output {
                self.mul_scalar(rhs)
            }
        }

        impl MulAssign<$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                *self = self.mul_scalar(rhs);
            }
        }

        impl Div<$n> for $t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $n) -> Self::Output {
                rhs.div_scalar(self)
            }
        }

        impl Div<$t> for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $t) -> Self::Output {
                self.div_scalar(rhs)
            }
        }

        impl DivAssign<$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                *self = self.div_scalar(rhs);
            }
        }

        impl Sum for $n {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::ZERO, Self::add)
            }
        }

        impl<'a> Sum<&'a Self> for $n {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
            }
        }

        impl Product for $n {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = Self>,
            {
                iter.fold(Self::IDENTITY, Self::mul)
            }
        }

        impl<'a> Product<&'a Self> for $n {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::IDENTITY, |a, &b| Self::mul(a, b))
            }
        }
        )+
    }
}

wide_mat2s!(
    Mat2x4 => Mat2, Mat3x4, Vec3x4, Vec2x4, f32x4,
    Mat2x8 => Mat2, Mat3x8, Vec3x8, Vec2x8, f32x8
);

#[cfg(feature = "f64")]
wide_mat2s!(
    DMat2x2 => DMat2, DMat3x2, DVec3x2, DVec2x2, f64x2,
    DMat2x4 => DMat2, DMat3x4, DVec3x4, DVec2x4, f64x4
);
