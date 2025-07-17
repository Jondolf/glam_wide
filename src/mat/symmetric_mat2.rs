use core::iter::Sum;
use core::ops::*;
#[cfg(feature = "f64")]
use glam_matrix_extensions::SymmetricDMat2;
#[cfg(feature = "f32")]
use glam_matrix_extensions::SymmetricMat2;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DMat2x2, DMat2x4, DMat23x2, DMat23x4, DMat32x2, DMat32x4, DVec2x2, DVec2x4};
#[cfg(feature = "f32")]
use crate::{Mat2x4, Mat2x8, Mat23x4, Mat23x8, Mat32x4, Mat32x8, Vec2x4, Vec2x8};
use crate::{SimdFloatExt, SimdLaneCount};

macro_rules! wide_symmetric_mat2s {
    ($($n:ident => $nonwiden:ident, $nonsymmetricn:ident, $m23t:ident, $m32t:ident, $vt:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        /// The bottom left triangle (including the diagonal) of a wide symmetric 2x2 column-major matrix.
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
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
            /// that the given matrix is truly symmetric.
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
            pub const fn to_mat2(&self) -> $nonsymmetricn {
                $nonsymmetricn::from_cols_array(&self.to_cols_array())
            }

            /// Creates a new symmetric 2x2 matrix from the outer product `v * v^T`.
            #[inline(always)]
            #[must_use]
            pub fn from_outer_product(v: $vt) -> Self {
                Self::new(v.x * v.x, v.x * v.y, v.y * v.y)
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

            /// Returns the diagonal of the matrix.
            #[inline]
            #[must_use]
            pub fn diagonal(&self) -> $vt {
                $vt::new(self.m00, self.m11)
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
                self.mul(rhs)
            }

            /// Multiplies `self` by a 2x3 matrix, `self * rhs`.
            #[inline]
            #[must_use]
            pub fn mul_mat23(&self, rhs: &$m23t) -> $m23t {
                self.mul(rhs)
            }

            /// Computes `a * transpose(b)`, assuming `a = b * M` for some symmetric matrix `M`.
            ///
            /// This effectively completes the second half of the sandwich product `b * M * transpose(b)`.
            #[inline]
            #[must_use]
            pub fn complete_mat23_sandwich(a: &$m23t, b: &$m23t) -> Self {
                Self::new(
                    a.row(0).dot(b.row(0)),
                    a.row(1).dot(b.row(0)),
                    a.row(1).dot(b.row(1)),
                )
            }

            /// Computes `a * transpose(b)`, assuming `a = b * M` for some symmetric matrix `M`.
            ///
            /// This effectively completes the second half of the sandwich product `b * M * transpose(b)`.
            #[inline]
            #[must_use]
            pub fn complete_mat32_sandwich(a: &$m32t, b: &$m32t) -> Self {
                Self::new(
                    a.col(0).dot(b.col(0)),
                    a.col(1).dot(b.col(0)),
                    a.col(1).dot(b.col(1)),
                )
            }

            /// Adds two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn add_mat2(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                self.add(rhs)
            }

            /// Subtracts two 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn sub_mat2(&self, rhs: &$nonsymmetricn) -> $nonsymmetricn {
                self.sub(rhs)
            }

            /// Multiplies two symmetric 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn mul_symmetric_mat2(&self, rhs: &Self) -> $nonsymmetricn {
                self.mul(rhs)
            }

            /// Adds two symmetric 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn add_symmetric_mat2(&self, rhs: &Self) -> Self {
                self.add(rhs)
            }

            /// Subtracts two symmetric 2x2 matrices.
            #[inline]
            #[must_use]
            pub fn sub_symmetric_mat2(&self, rhs: &Self) -> Self {
                self.sub(rhs)
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
                Self::new(
                    self.m00 + rhs.m00,
                    self.m01 + rhs.m01,
                    self.m11 + rhs.m11,
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
                    self.m11 - rhs.m11,
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
                    -self.m00, -self.m01, -self.m11,
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
                $nonsymmetricn::from_cols(self.mul(rhs.col(0)), self.mul(rhs.col(1)))
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
                Self::from_cols_array(&[
                    self.x_axis.x * rhs.m00 + self.y_axis.x * rhs.m01,
                    self.x_axis.y * rhs.m00 + self.y_axis.y * rhs.m01,
                    self.x_axis.x * rhs.m01 + self.y_axis.x * rhs.m11,
                    self.x_axis.y * rhs.m01 + self.y_axis.y * rhs.m11,
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

        impl Mul<$m23t> for $n {
            type Output = $m23t;
            #[inline]
            fn mul(self, rhs: $m23t) -> Self::Output {
                $m23t::from_cols(
                    $vt::new(
                        self.row(0).dot(rhs.x_axis),
                        self.row(1).dot(rhs.x_axis),
                    ),
                    $vt::new(
                        self.row(0).dot(rhs.y_axis),
                        self.row(1).dot(rhs.y_axis),
                    ),
                    $vt::new(
                        self.row(0).dot(rhs.z_axis),
                        self.row(1).dot(rhs.z_axis),
                    ),
                )
            }
        }

        impl Mul<&$m23t> for $n {
            type Output = $m23t;
            #[inline]
            fn mul(self, rhs: &$m23t) -> Self::Output {
                self.mul(*rhs)
            }
        }

        impl Mul<$m23t> for &$n {
            type Output = $m23t;
            #[inline]
            fn mul(self, rhs: $m23t) -> Self::Output {
                (*self).mul(rhs)
            }
        }

        impl Mul<&$m23t> for &$n {
            type Output = $m23t;
            #[inline]
            fn mul(self, rhs: &$m23t) -> Self::Output {
                (*self).mul(*rhs)
            }
        }

        impl Mul<$vt> for $n {
            type Output = $vt;
            #[inline]
            fn mul(self, rhs: $vt) -> Self::Output {
                self.mul_vec2(rhs)
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
                mat.to_mat2()
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
wide_symmetric_mat2s!(
    SymmetricMat2x4 => SymmetricMat2, Mat2x4, Mat23x4, Mat32x4, Vec2x4, f32x4, f32,
    SymmetricMat2x8 => SymmetricMat2, Mat2x8, Mat23x8, Mat32x8, Vec2x8, f32x8, f32
);

#[cfg(feature = "f64")]
wide_symmetric_mat2s!(
    SymmetricDMat2x2 => SymmetricDMat2, DMat2x2, DMat23x2, DMat32x2, DVec2x2, f64x2, f64,
    SymmetricDMat2x4 => SymmetricDMat2, DMat2x4, DMat23x4, DMat32x4, DVec2x4, f64x4, f64
);
