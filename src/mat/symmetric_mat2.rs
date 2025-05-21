#[cfg(feature = "f64")]
use bevy_math::{DMat2, DVec2};
use bevy_math::{Mat2, Vec2};
use core::iter::Sum;
use core::ops::*;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DMat2x2, DMat2x4, DVec2x2, DVec2x4};
use crate::{FloatExt, Mat2x4, Mat2x8, SimdFloatExt, SimdLaneCount, Vec2x4, Vec2x8};

macro_rules! symmetric_mat2s {
    ($($n:ident => $nonsymmetricn:ident, $vt:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        /// The bottom left triangle (including the diagonal) of a symmetric 2x2 column-major matrix.
        ///
        /// This is useful for storing a symmetric 2x2 matrix in a more compact form and performing some
        /// matrix operations more efficiently.
        ///
        /// Some defining properties of symmetric matrices include:
        ///
        /// - The matrix is equal to its transpose.
        /// - The matrix has real eigenvalues.
        /// - The eigenvectors corresponding to the eigenvalues are orthogonal.
        /// - The matrix is always diagonalizable.
        ///
        /// The sum and difference of two symmetric matrices is always symmetric.
        /// However, the product of two symmetric matrices is *only* symmetric
        /// if the matrices are commutable, meaning that `AB = BA`.
        #[derive(Clone, Copy, Debug)]
        pub struct $n {
            /// The first element of the first column.
            pub m00: $t,
            /// The second element of the first column.
            pub m01: $t,
            /// The second element of the second column.
            pub m11: $t,
        }

        impl $n {
            /// A symmetric 2x2 matrix with all elements set to `0.0`.
            pub const ZERO: Self = Self::new($t::ZERO, $t::ZERO, $t::ZERO);

            /// A symmetric 2x2 identity matrix, where all diagonal elements are `1.0`,
            /// and all off-diagonal elements are `0.0`.
            pub const IDENTITY: Self = Self::new($t::ONE, $t::ZERO, $t::ONE);

            /// All NaNs.
            pub const NAN: Self = Self::new($t::NAN, $t::NAN, $t::NAN);

            /// Creates a new symmetric 2x2 matrix from its bottom left triangle, including diagonal elements.
            ///
            /// The elements are in column-major order `mCR`, where `C` is the column index
            /// and `R` is the row index.
            #[inline(always)]
            #[must_use]
            pub const fn new(
                m00: $t,
                m01: $t,
                m11: $t,
            ) -> Self {
                Self { m00, m01, m11 }
            }

            /// Creates a symmetric 2x2 matrix from three column vectors.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            #[inline(always)]
            #[must_use]
            pub const fn from_cols_unchecked(x_axis: $vt, y_axis: $vt) -> Self {
                Self {
                    m00: x_axis.x,
                    m01: x_axis.y,
                    m11: y_axis.y,
                }
            }

            /// Creates a symmetric 2x2 matrix from an array stored in column major order.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array_unchecked(m: &[$t; 4]) -> Self {
                Self::new(m[0], m[1], m[3])
            }

            /// Creates an array storing data in column major order.
            #[inline]
            #[must_use]
            pub const fn to_cols_array(&self) -> [$t; 4] {
                [self.m00, self.m01, self.m01, self.m11]
            }

            /// Creates a symmetric 2x2 matrix from a 2D array stored in column major order.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array_2d_unchecked(m: &[[$t; 2]; 2]) -> Self {
                Self::from_cols_unchecked(
                    $vt::from_array(m[0]),
                    $vt::from_array(m[1]),
                )
            }

            /// Creates a 2D array storing data in column major order.
            #[inline]
            #[must_use]
            pub const fn to_cols_array_2d(&self) -> [[$t; 2]; 2] {
                [[self.m00, self.m01], [self.m01, self.m11]]
            }

            /// Creates a 2x2 matrix from the first 4 values in `slice`.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 4 elements long.
            #[inline]
            #[must_use]
            pub const fn from_cols_slice(slice: &[$t]) -> Self {
                Self::new(slice[0], slice[1], slice[3])
            }

            /// Creates a symmetric 2x2 matrix with its diagonal set to `diagonal` and all other entries set to `0.0`.
            #[inline]
            #[must_use]
            #[doc(alias = "scale")]
            pub const fn from_diagonal(diagonal: $vt) -> Self {
                Self::new(diagonal.x, $t::ZERO, diagonal.y)
            }

            /// Creates a symmetric 2x2 matrix from a 2x2 matrix.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given matrix is truly symmetric matrix.
            #[inline]
            #[must_use]
            pub fn from_mat2_unchecked(mat: $nonsymmetricn) -> Self {
                Self::new(
                    mat.x_axis.x,
                    mat.x_axis.y,
                    mat.y_axis.y,
                )
            }

            /// Creates a 2x2 matrix from the symmetric 2x2 matrix in `self`.
            #[inline]
            #[must_use]
            pub const fn to_mat3(&self) -> $nonsymmetricn {
                $nonsymmetricn::from_cols_array(&self.to_cols_array())
            }

            /// Returns the matrix column for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 1.
            #[inline]
            #[must_use]
            pub const fn col(&self, index: usize) -> $vt {
                match index {
                    0 => $vt::new(self.m00, self.m01),
                    1 => $vt::new(self.m01, self.m11),
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
            pub const fn row(&self, index: usize) -> $vt {
                match index {
                    0 => $vt::new(self.m00, self.m01),
                    1 => $vt::new(self.m01, self.m11),
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns the determinant of `self`.
            #[inline]
            #[must_use]
            pub fn determinant(&self) -> $t {
                // A = [ a c ]
                //     | c b |
                //
                // det(A) = ab - c^2
                let [a, b, c] = [self.m00, self.m11, self.m01];
                a * b - c * c
            }

            /// Returns the inverse of `self`.
            ///
            /// If the matrix is not invertible the returned matrix will be invalid.
            #[inline]
            #[must_use]
            pub fn inverse(&self) -> Self {
                let inv_det = 1.0 / self.determinant();
                Self {
                    m00: self.m11 * inv_det,
                    m01: -self.m01 * inv_det,
                    m11: self.m00 * inv_det,
                }
            }

            /// Takes the absolute value of each element in `self`.
            #[inline]
            #[must_use]
            pub fn abs(&self) -> Self {
                Self::new(
                    self.m00.abs(),
                    self.m01.abs(),
                    self.m11.abs(),
                )
            }

            /// Transforms a 2D vector.
            #[inline]
            #[must_use]
            pub fn mul_vec2(&self, rhs: $vt) -> $vt {
                let mut res = self.col(0).mul(rhs.x);
                res = res.add(self.col(1).mul(rhs.y));
                res
            }

            /// Multiplies two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn mul_mat2(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.mul(rhs.x_axis),
                    self.mul(rhs.y_axis),
                )
            }

            /// Adds two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn add_mat2(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.col(0).add(rhs.x_axis),
                    self.col(1).add(rhs.y_axis),
                )
            }

            /// Subtracts two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn sub_mat2(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.col(0).sub(rhs.x_axis),
                    self.col(1).sub(rhs.y_axis),
                )
            }

            /// Multiplies two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn mul_symmetric_mat2(&self, rhs: &Self) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.mul_vec2(rhs.col(0)),
                    self.mul_vec2(rhs.col(1)),
                )
            }

            /// Adds two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn add_symmetric_mat2(&self, rhs: &Self) -> Self {
                Self::new(
                    self.m00 + rhs.m00,
                    self.m01 + rhs.m01,
                    self.m11 + rhs.m11,
                )
            }

            /// Subtracts two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn sub_symmetric_mat2(&self, rhs: &Self) -> Self {
                Self::new(
                    self.m00 - rhs.m00,
                    self.m01 - rhs.m01,
                    self.m11 - rhs.m11,
                )
            }

            /// Multiplies a 2x2 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn mul_scalar(&self, rhs: $t) -> Self {
                Self::new(
                    self.m00 * rhs,
                    self.m01 * rhs,
                    self.m11 * rhs,
                )
            }

            /// Divides a 2x2 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn div_scalar(&self, rhs: $t) -> Self {
                Self::new(
                    self.m00 / rhs,
                    self.m01 / rhs,
                    self.m11 / rhs,
                )
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
                self.add_symmetric_mat2(&rhs)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Add<$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: $nonsymmetricn) -> Self::Output {
                self.add_mat2(&rhs)
            }
        }

        impl Add<$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: $n) -> Self::Output {
                rhs.add_mat2(&self)
            }
        }

        impl AddAssign<$n> for $nonsymmetricn {
            #[inline]
            fn add_assign(&mut self, rhs: $n) {
                *self = self.add(rhs);
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                self.sub_symmetric_mat2(&rhs)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Sub<$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: $nonsymmetricn) -> Self::Output {
                self.sub_mat2(&rhs)
            }
        }

        impl Sub<$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: $n) -> Self::Output {
                rhs.sub_mat2(&self)
            }
        }

        impl SubAssign<$n> for $nonsymmetricn {
            #[inline]
            fn sub_assign(&mut self, rhs: $n) {
                *self = self.sub(rhs);
            }
        }

        impl Neg for $n {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self::new(
                    -self.m00, -self.m01, -self.m11,
                )
            }
        }

        impl Mul<$n> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                self.mul_symmetric_mat2(&rhs)
            }
        }

        impl Mul<$n> for $nonsymmetricn {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                // TODO: Implement a `mul_symmetric_mat2` method for 2x2 matrices.
                Self::from_cols_array_2d(&[
                    [
                        self.x_axis.x * rhs.m00 + self.y_axis.x * rhs.m01,
                        self.x_axis.y * rhs.m00 + self.y_axis.y * rhs.m01,
                    ],
                    [
                        self.x_axis.x * rhs.m01 + self.y_axis.x * rhs.m11,
                        self.x_axis.y * rhs.m01 + self.y_axis.y * rhs.m11,
                    ],
                ])
            }
        }

        impl MulAssign<$n> for $nonsymmetricn {
            #[inline]
            fn mul_assign(&mut self, rhs: $n) {
                *self = self.mul(rhs);
            }
        }

        impl Mul<$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: $nonsymmetricn) -> Self::Output {
                self.mul_mat2(&rhs)
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

        impl From<$nonsymmetricn> for $n {
            #[inline]
            fn from(mat: $nonsymmetricn) -> Self {
                Self::from_mat2_unchecked(mat)
            }
        }

        impl From<$n> for $nonsymmetricn {
            #[inline]
            fn from(mat: $n) -> Self {
                Self::from_cols(mat.col(0), mat.col(1))
            }
        }

        impl Sum<$n> for $n {
            fn sum<I: Iterator<Item = $n>>(iter: I) -> Self {
                iter.fold(Self::ZERO, Self::add)
            }
        }

        impl<'a> Sum<&'a $n> for $n {
            fn sum<I: Iterator<Item = &'a $n>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, &b| a.add(b))
            }
        }
        )+
    }
}

macro_rules! impl_scalar_symmetric_mat2s {
    ($($n:ident),+) => {
        $(
        impl $n {
            /// Returns `true` if, and only if, all elements are finite.
            /// If any element is either `NaN` or positive or negative infinity, this will return `false`.
            #[inline]
            #[must_use]
            pub fn is_finite(&self) -> bool {
                self.m00.is_finite() && self.m01.is_finite() && self.m11.is_finite()
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            #[must_use]
            pub fn is_nan(&self) -> bool {
                self.m00.is_nan() || self.m01.is_nan() || self.m11.is_nan()
            }
        }
        )+
    }
}

macro_rules! impl_wide_symmetric_mat2s {
    ($($n:ident => $nonwiden:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        impl $n {
            /// Creates a new symmetric 2x2 matrix from its bottom left triangle, including diagonal elements,
            /// with all lanes set to the same values.
            ///
            /// The elements are in column-major order `mCR`, where `C` is the column index
            /// and `R` is the row index.
            #[inline]
            #[must_use]
            pub const fn new_splat(
                m00: $nonwidet,
                m01: $nonwidet,
                m11: $nonwidet,
            ) -> Self {
                Self {
                    m00: $t::new([m00; $t::LANES]),
                    m01: $t::new([m01; $t::LANES]),
                    m11: $t::new([m11; $t::LANES]),
                }
            }

            /// Creates a new wide symmetric 2x2 matrix with all lanes set to `m`.
            #[inline]
            #[must_use]
            pub const fn splat(m: $nonwiden) -> Self {
                Self {
                    m00: $t::new([m.m00; $t::LANES]),
                    m01: $t::new([m.m01; $t::LANES]),
                    m11: $t::new([m.m11; $t::LANES]),
                }
            }
        }
        )+
    }
}

symmetric_mat2s!(
    SymmetricMat2 => Mat2, Vec2, f32, f32,
    SymmetricMat2x4 => Mat2x4, Vec2x4, f32x4, f32,
    SymmetricMat2x8 => Mat2x8, Vec2x8, f32x8, f32
);

#[cfg(feature = "f64")]
symmetric_mat2s!(
    DSymmetricMat2 => DMat2, DVec2, f64, f64,
    DSymmetricMat2x2 => DMat2x2, DVec2x2, f64x2, f64,
    DSymmetricMat2x4 => DMat2x4, DVec2x4, f64x4, f64
);

impl_scalar_symmetric_mat2s!(SymmetricMat2);

#[cfg(feature = "f64")]
impl_scalar_symmetric_mat2s!(DSymmetricMat2);

impl_wide_symmetric_mat2s!(
    SymmetricMat2x4 => SymmetricMat2, f32x4, f32,
    SymmetricMat2x8 => SymmetricMat2, f32x8, f32
);

#[cfg(feature = "f64")]
impl_wide_symmetric_mat2s!(
    DSymmetricMat2x2 => DSymmetricMat2, f64x2, f64,
    DSymmetricMat2x4 => DSymmetricMat2, f64x4, f64
);
