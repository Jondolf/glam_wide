use core::iter::Sum;
use core::ops::*;
#[cfg(feature = "f64")]
use glam_matrix_extensions::SymmetricDMat3;
#[cfg(feature = "f32")]
use glam_matrix_extensions::SymmetricMat3;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DMat3x2, DMat3x4, DMat23x2, DMat23x4, DMat32x2, DMat32x4, DVec3x2, DVec3x4};
#[cfg(feature = "f32")]
use crate::{Mat3x4, Mat3x8, Mat23x4, Mat23x8, Mat32x4, Mat32x8, Vec3x4, Vec3x8};
use crate::{SimdFloatExt, SimdLaneCount};

macro_rules! wide_symmetric_mat3s {
    ($($n:ident => $nonwiden:ident, $nonsymmetricn:ident, $m23t:ident, $m32t:ident, $v2t:ident, $vt:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        /// The bottom left triangle (including the diagonal) of a wide symmetric 3x3 column-major matrix.
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
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

            /// Returns the diagonal of the matrix.
            #[inline]
            #[must_use]
            pub fn diagonal(&self) -> $vt {
                $vt::new(self.m00, self.m11, self.m22)
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
                let inv_d1 = 1.0 / d1;
                let l21 = inv_d1 * self.m01;
                let l31 = inv_d1 * self.m02;
                let d2 = self.m11 - l21 * l21 * d1;
                let inv_d2 = 1.0 / d2;
                let l32 = inv_d2 * (self.m12 - l21 * l31 * d1);
                let d3 = self.m22 - l31 * l31 * d1 - l32 * l32 * d2;
                let inv_d3 = 1.0 / d3;

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
                self.mul(rhs)
            }

            /// Multiplies `self` by a 3x2 matrix, `self * rhs`.
            #[inline]
            #[must_use]
            pub fn mul_mat32(&self, rhs: &$m32t) -> $m32t {
                self.mul(rhs)
            }

            /// Computes `a * transpose(b)`, assuming `a = b * M` for some symmetric matrix `M`.
            ///
            /// This effectively completes the second half of the sandwich product `b * M * transpose(b)`.
            #[inline]
            #[must_use]
            pub fn complete_mat23_sandwich(a: &$m23t, b: &$m23t) -> Self {
                Self::new(
                    a.col(0).dot(b.col(0)),
                    a.col(1).dot(b.col(0)),
                    a.col(2).dot(b.col(0)),
                    a.col(1).dot(b.col(1)),
                    a.col(2).dot(b.col(1)),
                    a.col(2).dot(b.col(2)),
                )
            }

            /// Computes `a * transpose(b)`, assuming `a = b * M` for some symmetric matrix `M`.
            ///
            /// This effectively completes the second half of the sandwich product `b * M * transpose(b)`.
            #[inline]
            #[must_use]
            pub fn complete_mat32_sandwich(a: &$m32t, b: &$m32t) -> Self {
                Self::new(
                    a.row(0).dot(b.row(0)),
                    a.row(1).dot(b.row(0)),
                    a.row(2).dot(b.row(0)),
                    a.row(1).dot(b.row(1)),
                    a.row(2).dot(b.row(1)),
                    a.row(2).dot(b.row(2)),
                )
            }

            /// Adds two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn add_mat3(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                self.add(rhs)
            }

            /// Subtracts two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn sub_mat3(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                self.sub(rhs)
            }

            /// Multiplies two symmetric 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn mul_symmetric_mat3(&self, rhs: &Self) -> $nonsymmetricn {
                self.mul(rhs)
            }

            /// Adds two symmetric 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn add_symmetric_mat3(&self, rhs: &Self) -> Self {
                self.add(rhs)
            }

            /// Subtracts two symmetric 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn sub_symmetric_mat3(&self, rhs: &Self) -> Self {
                self.sub(rhs)
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
                Self::new(
                    self.m00 + rhs.m00,
                    self.m01 + rhs.m01,
                    self.m02 + rhs.m02,
                    self.m11 + rhs.m11,
                    self.m12 + rhs.m12,
                    self.m22 + rhs.m22,
                )
            }
        }

        impl Add<&Self> for $n {
            type Output = Self;
            #[inline]
            fn add(self, rhs: &Self) -> Self::Output {
                self.add(*rhs)
            }
        }

        impl Add<Self> for &$n {
            type Output = $n;
            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                (*self).add(rhs)
            }
        }

        impl Add<&Self> for &$n {
            type Output = $n;
            #[inline]
            fn add(self, rhs: &Self) -> Self::Output {
                (*self).add(*rhs)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = self.add(rhs);
            }
        }

        impl AddAssign<&Self> for $n {
            #[inline]
            fn add_assign(&mut self, rhs: &Self) {
                self.add_assign(*rhs);
            }
        }

        impl Add<$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: $nonsymmetricn) -> Self::Output {
                $nonsymmetricn::from_cols(
                    self.col(0).add(rhs.x_axis),
                    self.col(1).add(rhs.y_axis),
                    self.col(2).add(rhs.z_axis),
                )
            }
        }

        impl Add<&$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: &$nonsymmetricn) -> Self::Output {
                self.add(*rhs)
            }
        }

        impl Add<$nonsymmetricn> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: $nonsymmetricn) -> Self::Output {
                (*self).add(rhs)
            }
        }

        impl Add<&$nonsymmetricn> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: &$nonsymmetricn) -> Self::Output {
                (*self).add(*rhs)
            }
        }

        impl Add<$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: $n) -> Self::Output {
                rhs.add(&self)
            }
        }

        impl Add<&$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: &$n) -> Self::Output {
                self.add(*rhs)
            }
        }

        impl Add<&$n> for &$nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn add(self, rhs: &$n) -> Self::Output {
                (*self).add(*rhs)
            }
        }

        impl AddAssign<$n> for $nonsymmetricn {
            #[inline]
            fn add_assign(&mut self, rhs: $n) {
                *self = self.add(rhs);
            }
        }

        impl AddAssign<&$n> for $nonsymmetricn {
            #[inline]
            fn add_assign(&mut self, rhs: &$n) {
                *self = self.add(*rhs);
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::new(
                    self.m00 - rhs.m00,
                    self.m01 - rhs.m01,
                    self.m02 - rhs.m02,
                    self.m11 - rhs.m11,
                    self.m12 - rhs.m12,
                    self.m22 - rhs.m22,
                )
            }
        }

        impl Sub<&Self> for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: &Self) -> Self::Output {
                self.sub(*rhs)
            }
        }

        impl Sub<Self> for &$n {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                (*self).sub(rhs)
            }
        }

        impl Sub<&Self> for &$n {
            type Output = $n;
            #[inline]
            fn sub(self, rhs: &Self) -> Self::Output {
                (*self).sub(*rhs)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.sub(rhs);
            }
        }

        impl SubAssign<&Self> for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: &Self) {
                self.sub_assign(*rhs);
            }
        }

        impl Sub<$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: $nonsymmetricn) -> Self::Output {
                $nonsymmetricn::from_cols(
                    self.col(0).sub(rhs.x_axis),
                    self.col(1).sub(rhs.y_axis),
                    self.col(2).sub(rhs.z_axis),
                )
            }
        }

        impl Sub<&$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: &$nonsymmetricn) -> Self::Output {
                self.sub(*rhs)
            }
        }

        impl Sub<$nonsymmetricn> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: $nonsymmetricn) -> Self::Output {
                (*self).sub(rhs)
            }
        }

        impl Sub<&$nonsymmetricn> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: &$nonsymmetricn) -> Self::Output {
                (*self).sub(*rhs)
            }
        }

        impl Sub<$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: $n) -> Self::Output {
                rhs.sub(&self)
            }
        }

        impl Sub<&$n> for $nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: &$n) -> Self::Output {
                self.sub(*rhs)
            }
        }

        impl Sub<&$n> for &$nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn sub(self, rhs: &$n) -> Self::Output {
                (*self).sub(*rhs)
            }
        }

        impl SubAssign<$n> for $nonsymmetricn {
            #[inline]
            fn sub_assign(&mut self, rhs: $n) {
                *self = self.sub(rhs);
            }
        }

        impl SubAssign<&$n> for $nonsymmetricn {
            #[inline]
            fn sub_assign(&mut self, rhs: &$n) {
                *self = self.sub(*rhs);
            }
        }

        impl Neg for $n {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self::new(
                    -self.m00,
                    -self.m01,
                    -self.m02,
                    -self.m11,
                    -self.m12,
                    -self.m22,
                )
            }
        }

        impl Neg for &$n {
            type Output = $n;
            #[inline]
            fn neg(self) -> Self::Output {
                (*self).neg()
            }
        }

        impl Mul for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                $nonsymmetricn::from_cols(
                    self.mul(rhs.col(0)),
                    self.mul(rhs.col(1)),
                    self.mul(rhs.col(2)),
                )
            }
        }

        impl Mul<&Self> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: &Self) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<Self> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&Self> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: &Self) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$n> for $nonsymmetricn {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
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

        impl Mul<&$n> for $nonsymmetricn {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: &$n) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$n> for &$nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$n> for &$nonsymmetricn {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: &$n) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl MulAssign<$n> for $nonsymmetricn {
            #[inline]
            fn mul_assign(&mut self, rhs: $n) {
                *self = self.mul(rhs);
            }
        }

        impl MulAssign<&$n> for $nonsymmetricn {
            #[inline]
            fn mul_assign(&mut self, rhs: &$n) {
                *self = self.mul(*rhs);
            }
        }

        impl Mul<$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: $nonsymmetricn) -> Self::Output {
                $nonsymmetricn::from_cols(
                    self.mul(rhs.x_axis),
                    self.mul(rhs.y_axis),
                    self.mul(rhs.z_axis),
                )
            }
        }

        impl Mul<&$nonsymmetricn> for $n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: &$nonsymmetricn) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$nonsymmetricn> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: $nonsymmetricn) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$nonsymmetricn> for &$n {
            type Output = $nonsymmetricn;
            #[inline]
            fn mul(self, rhs: &$nonsymmetricn) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$m32t> for $n {
            type Output = $m32t;
            #[inline]
            fn mul(self, rhs: $m32t) -> Self::Output {
                $m32t::from_cols(
                    $vt::new(
                        self.row(0).dot(rhs.x_axis),
                        self.row(1).dot(rhs.x_axis),
                        self.row(2).dot(rhs.x_axis),
                    ),
                    $vt::new(
                        self.row(0).dot(rhs.y_axis),
                        self.row(1).dot(rhs.y_axis),
                        self.row(2).dot(rhs.y_axis),
                    ),
                )
            }
        }

        impl Mul<&$m32t> for $n {
            type Output = $m32t;
            #[inline]
            fn mul(self, rhs: &$m32t) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$m32t> for &$n {
            type Output = $m32t;
            #[inline]
            fn mul(self, rhs: $m32t) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$m32t> for &$n {
            type Output = $m32t;
            #[inline]
            fn mul(self, rhs: &$m32t) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$vt> for $n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: $vt) -> Self::Output {
                self.mul_vec3(rhs)
            }
        }

        impl Mul<&$vt> for $n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: &$vt) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$vt> for &$n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: $vt) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$vt> for &$n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: &$vt) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                rhs.mul_scalar(self)
            }
        }

        impl Mul<&$n> for $t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$n) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$n> for &$t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $n) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$n) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$t> for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $t) -> Self::Output {
                self.mul_scalar(rhs)
            }
        }

        impl Mul<&$t> for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: &$t) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$t> for &$n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: $t) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn mul(self, rhs: &$t) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl MulAssign<$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: $t) {
                *self = self.mul(rhs);
            }
        }

        impl MulAssign<&$t> for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: &$t) {
                self.mul_assign(*rhs);
            }
        }

        impl Div<$n> for $t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $n) -> Self::Output {
                rhs.div_scalar(self)
            }
        }

        impl Div<&$n> for $t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$n) -> Self::Output {
                self.div(*rhs)
            }
        }

        impl Div<$n> for &$t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $n) -> Self::Output {
                (*self).div(rhs)
            }
        }

        impl Div<&$n> for &$t {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$n) -> Self::Output {
                (*self).div(*rhs)
            }
        }

        impl Div<$t> for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $t) -> Self::Output {
                self.div_scalar(rhs)
            }
        }

        impl Div<&$t> for $n {
            type Output = Self;
            #[inline]
            fn div(self, rhs: &$t) -> Self::Output {
                self.div(*rhs)
            }
        }

        impl Div<$t> for &$n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: $t) -> Self::Output {
                (*self).div(rhs)
            }
        }

        impl Div<&$t> for &$n {
            type Output = $n;
            #[inline]
            fn div(self, rhs: &$t) -> Self::Output {
                (*self).div(*rhs)
            }
        }

        impl DivAssign<$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: $t) {
                *self = self.div(rhs);
            }
        }

        impl DivAssign<&$t> for $n {
            #[inline]
            fn div_assign(&mut self, rhs: &$t) {
                self.div_assign(*rhs);
            }
        }

        impl From<$n> for $nonsymmetricn {
            #[inline]
            fn from(mat: $n) -> Self {
                mat.to_mat3()
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

#[cfg(feature = "f32")]
wide_symmetric_mat3s!(
    SymmetricMat3x4 => SymmetricMat3, Mat3x4, Mat23x4, Mat32x4, Vec2x4, Vec3x4, f32x4, f32,
    SymmetricMat3x8 => SymmetricMat3, Mat3x8, Mat23x8, Mat32x8, Vec2x8, Vec3x8, f32x8, f32
);

#[cfg(feature = "f64")]
wide_symmetric_mat3s!(
    SymmetricDMat3x2 => SymmetricDMat3, DMat3x2, DMat23x2, DMat32x2, DVec2x2, DVec3x2, f64x2, f64,
    SymmetricDMat3x4 => SymmetricDMat3, DMat3x4, DMat23x4, DMat32x4, DVec2x4, DVec3x4, f64x4, f64
);
