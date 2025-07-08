#[cfg(feature = "f64")]
use bevy_math::{DMat3, DVec3};
use bevy_math::{Mat3, Vec3, Vec3A};
use core::iter::Sum;
use core::ops::*;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DMat3x2, DMat3x4, DVec3x2, DVec3x4};
use crate::{
    FloatExt, Mat3x4, Mat3x8, MatConversionError, MatExt, SimdFloatExt, SimdLaneCount, Vec3x4,
    Vec3x8,
};

macro_rules! symmetric_mat3s {
    ($reflect_trait:path, $($n:ident => $nonsymmetricn:ident, $v2t:ident, $vt:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        /// The bottom left triangle (including the diagonal) of a symmetric 3x3 column-major matrix.
        ///
        /// This is useful for storing a symmetric 3x3 matrix in a more compact form and performing some
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
        #[cfg_attr(feature = "bevy_reflect", derive($reflect_trait))]
        #[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
        pub struct $n {
            /// The first element of the first column.
            pub m00: $t,
            /// The second element of the first column.
            pub m01: $t,
            /// The third element of the first column.
            pub m02: $t,
            /// The second element of the second column.
            pub m11: $t,
            /// The third element of the second column.
            pub m12: $t,
            /// The third element of the third column.
            pub m22: $t,
        }

        impl $n {
            /// A symmetric 3x3 matrix with all elements set to `0.0`.
            pub const ZERO: Self = Self::new($t::ZERO, $t::ZERO, $t::ZERO, $t::ZERO, $t::ZERO, $t::ZERO);

            /// A symmetric 3x3 identity matrix, where all diagonal elements are `1.0`,
            /// and all off-diagonal elements are `0.0`.
            pub const IDENTITY: Self = Self::new($t::ONE, $t::ZERO, $t::ZERO, $t::ONE, $t::ZERO, $t::ONE);

            /// All NaNs.
            pub const NAN: Self = Self::new(
                $t::NAN,
                $t::NAN,
                $t::NAN,
                $t::NAN,
                $t::NAN,
                $t::NAN,
            );

            /// Creates a new symmetric 3x3 matrix from its bottom left triangle, including diagonal elements.
            ///
            /// The elements are in column-major order `mCR`, where `C` is the column index
            /// and `R` is the row index.
            #[inline(always)]
            #[must_use]
            pub const fn new(
                m00: $t,
                m01: $t,
                m02: $t,
                m11: $t,
                m12: $t,
                m22: $t,
            ) -> Self {
                Self {
                    m00,
                    m01,
                    m02,
                    m11,
                    m12,
                    m22,
                }
            }

            /// Creates a symmetric 3x3 matrix from three column vectors.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            #[inline(always)]
            #[must_use]
            pub const fn from_cols_unchecked(x_axis: $vt, y_axis: $vt, z_axis: $vt) -> Self {
                Self {
                    m00: x_axis.x,
                    m01: x_axis.y,
                    m02: x_axis.z,
                    m11: y_axis.y,
                    m12: y_axis.z,
                    m22: z_axis.z,
                }
            }

            /// Creates a symmetric 3x3 matrix from an array stored in column major order.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array_unchecked(m: &[$t; 9]) -> Self {
                Self::new(m[0], m[1], m[2], m[4], m[5], m[8])
            }

            /// Creates an array storing data in column major order.
            #[inline]
            #[must_use]
            pub const fn to_cols_array(&self) -> [$t; 9] {
                [
                    self.m00, self.m01, self.m02, self.m01, self.m11, self.m12, self.m02, self.m12,
                    self.m22,
                ]
            }

            /// Creates a symmetric 3x3 matrix from a 2D array stored in column major order.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array_2d_unchecked(m: &[[$t; 3]; 3]) -> Self {
                Self::from_cols_unchecked(
                    $vt::from_array(m[0]),
                    $vt::from_array(m[1]),
                    $vt::from_array(m[2]),
                )
            }

            /// Creates a 2D array storing data in column major order.
            #[inline]
            #[must_use]
            pub const fn to_cols_array_2d(&self) -> [[$t; 3]; 3] {
                [
                    [self.m00, self.m01, self.m02],
                    [self.m01, self.m11, self.m12],
                    [self.m02, self.m12, self.m22],
                ]
            }

            /// Creates a 3x3 matrix from the first 9 values in `slice`.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given columns truly produce a symmetric matrix.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 9 elements long.
            #[inline]
            #[must_use]
            pub const fn from_cols_slice(slice: &[$t]) -> Self {
                Self::new(slice[0], slice[1], slice[2], slice[4], slice[5], slice[8])
            }

            /// Creates a symmetric 3x3 matrix with its diagonal set to `diagonal` and all other entries set to `0.0`.
            #[inline]
            #[must_use]
            #[doc(alias = "scale")]
            pub const fn from_diagonal(diagonal: $vt) -> Self {
                Self::new(diagonal.x, $t::ZERO, $t::ZERO, diagonal.y, $t::ZERO, diagonal.z)
            }

            /// Creates a symmetric 3x3 matrix from a 3x3 matrix.
            ///
            /// Only the lower left triangle of the matrix is used. No check is performed to ensure
            /// that the given matrix is truly symmetric.
            #[inline]
            #[must_use]
            pub const fn from_mat3_unchecked(mat: $nonsymmetricn) -> Self {
                Self::new(
                    mat.x_axis.x,
                    mat.x_axis.y,
                    mat.x_axis.z,
                    mat.y_axis.y,
                    mat.y_axis.z,
                    mat.z_axis.z,
                )
            }

            /// Creates a 3x3 matrix from the symmetric 3x3 matrix in `self`.
            #[inline]
            #[must_use]
            pub const fn to_mat3(&self) -> $nonsymmetricn {
                $nonsymmetricn::from_cols_array(&self.to_cols_array())
            }

            /// Creates a new symmetric 3x3 matrix from the outer product `v * v^T`.
            #[inline(always)]
            #[must_use]
            pub fn from_outer_product(v: $vt) -> Self {
                Self::new(
                    v.x * v.x,
                    v.x * v.y,
                    v.x * v.z,
                    v.y * v.y,
                    v.y * v.z,
                    v.z * v.z,
                )
            }

            /// Returns the matrix column for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 2.
            #[inline]
            #[must_use]
            pub const fn col(&self, index: usize) -> $vt {
                match index {
                    0 => $vt::new(self.m00, self.m01, self.m02),
                    1 => $vt::new(self.m01, self.m11, self.m12),
                    2 => $vt::new(self.m02, self.m12, self.m22),
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns the matrix row for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 2.
            #[inline]
            #[must_use]
            pub const fn row(&self, index: usize) -> $vt {
                match index {
                    0 => $vt::new(self.m00, self.m01, self.m02),
                    1 => $vt::new(self.m01, self.m11, self.m12),
                    2 => $vt::new(self.m02, self.m12, self.m22),
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns the determinant of `self`.
            #[inline]
            #[must_use]
            pub fn determinant(&self) -> $t {
                //     [ a d e ]
                // A = | d b f |
                //     [ e f c ]
                //
                // det(A) = abc + 2def - af^2 - bd^2 - ce^2
                let [a, b, c] = [self.m00, self.m11, self.m22];
                let [d, e, f] = [self.m01, self.m02, self.m12];
                a * b * c + 2.0 * d * e * f - a * f * f - b * d * d - c * e * e
            }

            /// Returns the inverse of `self`.
            ///
            /// If the matrix is not invertible the returned matrix will be invalid.
            #[inline]
            #[must_use]
            pub fn inverse(&self) -> Self {
                let m00 = self.m11 * self.m22 - self.m12 * self.m12;
                let m01 = self.m12 * self.m02 - self.m22 * self.m01;
                let m02 = self.m01 * self.m12 - self.m02 * self.m11;

                let inverse_determinant = 1.0 / (m00 * self.m00 + m01 * self.m01 + m02 * self.m02);

                let m11 = self.m22 * self.m00 - self.m02 * self.m02;
                let m12 = self.m02 * self.m01 - self.m00 * self.m12;
                let m22 = self.m00 * self.m11 - self.m01 * self.m01;

                Self {
                    m00: m00 * inverse_determinant,
                    m01: m01 * inverse_determinant,
                    m02: m02 * inverse_determinant,
                    m11: m11 * inverse_determinant,
                    m12: m12 * inverse_determinant,
                    m22: m22 * inverse_determinant,
                }
            }

            /// Takes the absolute value of each element in `self`.
            #[inline]
            #[must_use]
            pub fn abs(&self) -> Self {
                Self::new(
                    self.m00.abs(),
                    self.m01.abs(),
                    self.m02.abs(),
                    self.m11.abs(),
                    self.m12.abs(),
                    self.m22.abs(),
                )
            }

            /* TODO
            /// Computes `(self * other.transpose()).transpose()`.
            #[inline]
            #[must_use]
            pub fn mul_by_transposed_mat2x3(&self, other: Matrix2x3) -> Matrix2x3 {
                Matrix2x3::new(
                    self.m00 * other.x_axis.x + self.m01 * other.y_axis.x + self.m02 * other.z_axis.x,
                    self.m00 * other.x_axis.y + self.m01 * other.y_axis.y + self.m02 * other.z_axis.y,
                    self.m01 * other.x_axis.x + self.m11 * other.y_axis.x + self.m12 * other.z_axis.x,
                    self.m01 * other.x_axis.y + self.m11 * other.y_axis.y + self.m12 * other.z_axis.y,
                    self.m02 * other.x_axis.x + self.m12 * other.y_axis.x + self.m22 * other.z_axis.x,
                    self.m02 * other.x_axis.y + self.m12 * other.y_axis.y + self.m22 * other.z_axis.y,
                )
            }
            */

            /// Computes `skew_symmetric(vec) * self * skew_symmetric(vec).transpose()` for a symmetric matrix `self`.
            #[inline]
            #[must_use]
            pub fn skew(&self, vec: $vt) -> Self {
                // 27 multiplications and 14 additions

                let xzy = vec.x * self.m12;
                let yzx = vec.y * self.m02;
                let zyx = vec.z * self.m01;

                let ixy = vec.y * self.m12 - vec.z * self.m11;
                let ixz = vec.y * self.m22 - vec.z * self.m12;
                let iyx = vec.z * self.m00 - vec.x * self.m02;
                let iyy = zyx - xzy;

                let iyz = vec.z * self.m02 - vec.x * self.m22;
                let izx = vec.x * self.m01 - vec.y * self.m00;
                let izy = vec.x * self.m11 - vec.y * self.m01;
                let izz = xzy - yzx;

                Self::new(
                    vec.y * ixz - vec.z * ixy,
                    vec.y * iyz - vec.z * iyy,
                    vec.y * izz - vec.z * izy,
                    vec.z * iyx - vec.x * iyz,
                    vec.z * izx - vec.x * izz,
                    vec.x * izy - vec.y * izx,
                )
            }

            /// Transforms a 3D vector.
            #[inline]
            #[must_use]
            pub fn mul_vec3(&self, rhs: $vt) -> $vt {
                let mut res = self.col(0).mul(rhs.x);
                res = res.add(self.col(1).mul(rhs.y));
                res = res.add(self.col(2).mul(rhs.z));
                res
            }

            /// Solves `self * x = rhs` for `x` using the LDLT decomposition.
            ///
            /// `self` must be a positive semidefinite matrix.
            #[inline]
            #[must_use]
            pub fn ldlt_solve(&self, rhs: $vt) -> $vt {
                let d1 = self.m00;
                let inv_d1 = $t::ONE / d1;
                let l21 = inv_d1 * self.m01;
                let l31 = inv_d1 * self.m02;
                let d2 = self.m11 - l21 * l21 * d1;
                let inv_d2 = $t::ONE / d2;
                let l32 = inv_d2 * (self.m12 - l21 * l31 * d1);
                let d3 = self.m22 - l31 * l31 * d1 - l32 * l32 * d2;
                let inv_d3 = $t::ONE / d3;

                // Forward substitution: Solve L * y = b
                let y1 = rhs.x;
                let y2 = rhs.y - l21 * y1;
                let y3 = rhs.z - l31 * y1 - l32 * y2;

                // Diagonal: Solve D * z = y
                let z1 = y1 * inv_d1;
                let z2 = y2 * inv_d2;
                let z3 = y3 * inv_d3;

                // Backward substitution: Solve L^T * x = z
                let x3 = z3;
                let x2 = z2 - l32 * x3;
                let x1 = z1 - l21 * x2 - l31 * x3;

                $vt::new(x1, x2, x3)
            }

            /// Multiplies two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn mul_mat3(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.mul(rhs.x_axis),
                    self.mul(rhs.y_axis),
                    self.mul(rhs.z_axis),
                )
            }

            /// Adds two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn add_mat3(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.col(0).add(rhs.x_axis),
                    self.col(1).add(rhs.y_axis),
                    self.col(2).add(rhs.z_axis),
                )
            }

            /// Subtracts two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn sub_mat3(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.col(0).sub(rhs.x_axis),
                    self.col(1).sub(rhs.y_axis),
                    self.col(2).sub(rhs.z_axis),
                )
            }

            /// Multiplies two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn mul_symmetric_mat3(&self, rhs: &Self) -> $nonsymmetricn {
                $nonsymmetricn::from_cols(
                    self.mul_vec3(rhs.col(0)),
                    self.mul_vec3(rhs.col(1)),
                    self.mul_vec3(rhs.col(2)),
                )
            }

            /// Adds two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn add_symmetric_mat3(&self, rhs: &Self) -> Self {
                Self::new(
                    self.m00 + rhs.m00,
                    self.m01 + rhs.m01,
                    self.m02 + rhs.m02,
                    self.m11 + rhs.m11,
                    self.m12 + rhs.m12,
                    self.m22 + rhs.m22,
                )
            }

            /// Subtracts two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn sub_symmetric_mat3(&self, rhs: &Self) -> Self {
                Self::new(
                    self.m00 - rhs.m00,
                    self.m01 - rhs.m01,
                    self.m02 - rhs.m02,
                    self.m11 - rhs.m11,
                    self.m12 - rhs.m12,
                    self.m22 - rhs.m22,
                )
            }

            /// Multiplies a 3x3 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn mul_scalar(&self, rhs: $t) -> Self {
                Self::new(
                    self.m00 * rhs,
                    self.m01 * rhs,
                    self.m02 * rhs,
                    self.m11 * rhs,
                    self.m12 * rhs,
                    self.m22 * rhs,
                )
            }

            /// Divides a 3x3 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn div_scalar(&self, rhs: $t) -> Self {
                Self::new(
                    self.m00 / rhs,
                    self.m01 / rhs,
                    self.m02 / rhs,
                    self.m11 / rhs,
                    self.m12 / rhs,
                    self.m22 / rhs,
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
                self.add_symmetric_mat3(&rhs)
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
                self.add_mat3(&rhs)
            }
        }

        impl Add<$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: $n) -> Self::Output {
                rhs.add_mat3(&self)
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
                self.sub_symmetric_mat3(&rhs)
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
                self.sub_mat3(&rhs)
            }
        }

        impl Sub<$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: $n) -> Self::Output {
                rhs.sub_mat3(&self)
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
                    -self.m00, -self.m01, -self.m02, -self.m11, -self.m12, -self.m22,
                )
            }
        }

        impl Mul<$n> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                self.mul_symmetric_mat3(&rhs)
            }
        }

        impl Mul<$n> for $nonsymmetricn {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                // TODO: Implement a `mul_symmetric_mat3` method for 3x3 matrices.
                Self::from_cols_array_2d(&[
                    [
                        self.x_axis.x * rhs.m00 + self.y_axis.x * rhs.m01 + self.z_axis.x * rhs.m02,
                        self.x_axis.y * rhs.m00 + self.y_axis.y * rhs.m01 + self.z_axis.y * rhs.m02,
                        self.x_axis.z * rhs.m00 + self.y_axis.z * rhs.m01 + self.z_axis.z * rhs.m02,
                    ],
                    [
                        self.x_axis.x * rhs.m01 + self.y_axis.x * rhs.m11 + self.z_axis.x * rhs.m12,
                        self.x_axis.y * rhs.m01 + self.y_axis.y * rhs.m11 + self.z_axis.y * rhs.m12,
                        self.x_axis.z * rhs.m01 + self.y_axis.z * rhs.m11 + self.z_axis.z * rhs.m12,
                    ],
                    [
                        self.x_axis.x * rhs.m02 + self.y_axis.x * rhs.m12 + self.z_axis.x * rhs.m22,
                        self.x_axis.y * rhs.m02 + self.y_axis.y * rhs.m12 + self.z_axis.y * rhs.m22,
                        self.x_axis.z * rhs.m02 + self.y_axis.z * rhs.m12 + self.z_axis.z * rhs.m22,
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
                self.mul_mat3(&rhs)
            }
        }

        /* TODO
        impl Mul<$n> for Matrix2x3 {
            type Output = Matrix2x3;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                Matrix2x3::new(
                    self.x_axis.x * rhs.m00 + self.y_axis.x * rhs.m01 + self.z_axis.x * rhs.m02,
                    self.x_axis.y * rhs.m00 + self.y_axis.y * rhs.m01 + self.z_axis.y * rhs.m02,
                    self.x_axis.x * rhs.m01 + self.y_axis.x * rhs.m11 + self.z_axis.x * rhs.m12,
                    self.x_axis.y * rhs.m01 + self.y_axis.y * rhs.m11 + self.z_axis.y * rhs.m12,
                    self.x_axis.x * rhs.m02 + self.y_axis.x * rhs.m12 + self.z_axis.x * rhs.m22,
                    self.x_axis.y * rhs.m02 + self.y_axis.y * rhs.m12 + self.z_axis.y * rhs.m22,
                )
            }
        }
        */

        impl Mul<$vt> for $n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: $vt) -> Self::Output {
                self.mul_vec3(rhs)
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

        impl From<$n> for $nonsymmetricn {
            #[inline]
            fn from(mat: $n) -> Self {
                Self::from_cols(mat.col(0), mat.col(1), mat.col(2))
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

macro_rules! impl_scalar_symmetric_mat3s {
    ($($n:ident => $nonsymmetricn:ident),+) => {
        $(
        impl $n {
            /// Tries to create a symmetric 3x3 matrix from a 3x3 matrix.
            ///
            /// # Errors
            ///
            /// Returns a [`MatConversionError`] if the given matrix is not symmetric.
            #[inline]
            pub fn try_from_mat3(mat: $nonsymmetricn) -> Result<Self, MatConversionError> {
                if mat.is_symmetric() {
                    Ok(Self::from_mat3_unchecked(mat))
                } else {
                    Err(MatConversionError::Asymmetric)
                }
            }

            /// Returns `true` if, and only if, all elements are finite.
            /// If any element is either `NaN` or positive or negative infinity, this will return `false`.
            #[inline]
            #[must_use]
            pub fn is_finite(&self) -> bool {
                self.m00.is_finite()
                    && self.m01.is_finite()
                    && self.m11.is_finite()
                    && self.m02.is_finite()
                    && self.m12.is_finite()
                    && self.m22.is_finite()
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            #[must_use]
            pub fn is_nan(&self) -> bool {
                self.m00.is_nan()
                    || self.m01.is_nan()
                    || self.m11.is_nan()
                    || self.m02.is_nan()
                    || self.m12.is_nan()
                    || self.m22.is_nan()
            }
        }

        impl TryFrom<$nonsymmetricn> for $n {
            type Error = MatConversionError;

            #[inline]
            fn try_from(mat: $nonsymmetricn) -> Result<Self, Self::Error> {
                Self::try_from_mat3(mat)
            }
        }

        impl PartialEq for $n {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.m00 == other.m00
                    && self.m01 == other.m01
                    && self.m11 == other.m11
                    && self.m02 == other.m02
                    && self.m12 == other.m12
                    && self.m22 == other.m22
            }
        }
        )+
    }
}

impl SymmetricMat3 {
    /// Transforms a [`Vec3A`].
    #[inline]
    #[must_use]
    pub fn mul_vec3a(&self, rhs: Vec3A) -> Vec3A {
        self.mul_vec3(rhs.into()).into()
    }
}

impl Mul<Vec3A> for SymmetricMat3 {
    type Output = Vec3A;
    #[inline]
    fn mul(self, rhs: Vec3A) -> Self::Output {
        self.mul_vec3a(rhs)
    }
}

macro_rules! impl_wide_symmetric_mat3s {
    ($($n:ident => $nonwiden:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        impl $n {
            /// Creates a new symmetric 3x3 matrix from its bottom left triangle, including diagonal elements,
            /// with all lanes set to the same values.
            ///
            /// The elements are in column-major order `mCR`, where `C` is the column index
            /// and `R` is the row index.
            #[inline]
            #[must_use]
            pub const fn new_splat(
                m00: $nonwidet,
                m01: $nonwidet,
                m02: $nonwidet,
                m11: $nonwidet,
                m12: $nonwidet,
                m22: $nonwidet,
            ) -> Self {
                Self {
                    m00: $t::new([m00; $t::LANES]),
                    m01: $t::new([m01; $t::LANES]),
                    m02: $t::new([m02; $t::LANES]),
                    m11: $t::new([m11; $t::LANES]),
                    m12: $t::new([m12; $t::LANES]),
                    m22: $t::new([m22; $t::LANES]),
                }
            }

            /// Creates a new symmetric 3x3 matrix with all lanes set to `m`.
            #[inline]
            #[must_use]
            pub const fn splat(m: $nonwiden) -> Self {
                Self {
                    m00: $t::new([m.m00; $t::LANES]),
                    m01: $t::new([m.m01; $t::LANES]),
                    m02: $t::new([m.m02; $t::LANES]),
                    m11: $t::new([m.m11; $t::LANES]),
                    m12: $t::new([m.m12; $t::LANES]),
                    m22: $t::new([m.m22; $t::LANES]),
                }
            }
        }
        )+
    }
}

symmetric_mat3s!(
    bevy_reflect::Reflect,
    SymmetricMat3 => Mat3, Vec2, Vec3, f32, f32
);

symmetric_mat3s!(
    bevy_reflect::TypePath,
    SymmetricMat3x4 => Mat3x4, Vec2x4, Vec3x4, f32x4, f32,
    SymmetricMat3x8 => Mat3x8, Vec2x8, Vec3x8, f32x8, f32
);

#[cfg(feature = "f64")]
symmetric_mat3s!(
    bevy_reflect::Reflect,
    DSymmetricMat3 => DMat3, DVec2, DVec3, f64, f64
);

#[cfg(feature = "f64")]
symmetric_mat3s!(
    bevy_reflect::TypePath,
    DSymmetricMat3x2 => DMat3x2, DVec2x2, DVec3x2, f64x2, f64,
    DSymmetricMat3x4 => DMat3x4, DVec2x4, DVec3x4, f64x4, f64
);

impl_scalar_symmetric_mat3s!(SymmetricMat3 => Mat3);

#[cfg(feature = "f64")]
impl_scalar_symmetric_mat3s!(DSymmetricMat3 => DMat3);

impl_wide_symmetric_mat3s!(
    SymmetricMat3x4 => SymmetricMat3, f32x4, f32,
    SymmetricMat3x8 => SymmetricMat3, f32x8, f32
);

#[cfg(feature = "f64")]
impl_wide_symmetric_mat3s!(
    DSymmetricMat3x2 => DSymmetricMat3, f64x2, f64,
    DSymmetricMat3x4 => DSymmetricMat3, f64x4, f64
);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use bevy_math::Vec3;

    use crate::SymmetricMat3;

    #[test]
    fn ldlt_solve() {
        let sym3 = SymmetricMat3::new(4.0, 1.0, 5.0, 0.0, 2.0, 6.0);

        // Known solution x
        let x = Vec3::new(1.0, 2.0, 3.0);

        // Compute rhs = A * x
        let rhs = sym3.mul_vec3(x);
        assert_eq!(rhs, Vec3::new(21.0, 7.0, 27.0));

        // Solve
        let sol = sym3.ldlt_solve(rhs);

        // Check solution
        assert_relative_eq!(sol, x, epsilon = 1e-4);
    }

    #[test]
    fn ldlt_solve_identity() {
        let sym3 = SymmetricMat3::IDENTITY;

        // Known solution x
        let x = Vec3::new(7.0, -3.0, 2.5);

        // Compute rhs = A * x
        let rhs = sym3.mul_vec3(x);

        // Solve
        let sol = sym3.ldlt_solve(rhs);

        // Check solution
        assert_relative_eq!(sol, x, epsilon = 1e-6);
    }
}
