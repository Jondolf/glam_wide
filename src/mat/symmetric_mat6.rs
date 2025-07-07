#[cfg(feature = "f64")]
use bevy_math::{DMat3, DVec3};
use bevy_math::{Mat3, Vec3};
use core::iter::Sum;
use core::ops::*;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{
    DMat3Ext, DMat3x2, DMat3x4, DSymmetricMat3, DSymmetricMat3x2, DSymmetricMat3x4, DVec3x2,
    DVec3x4,
};
use crate::{
    FloatExt, Mat3Ext, Mat3x4, Mat3x8, SymmetricMat3, SymmetricMat3x4, SymmetricMat3x8, Vec3x4,
    Vec3x8,
};

macro_rules! symmetric_mat6s {
    ($reflect_trait:path, $($n:ident => $symmetricm3t:ident, $m3t:ident, $v3t:ident, $t:ident, $nonwidet:ident),+) => {
        $(
        /// The bottom left triangle (including the diagonal) of a symmetric 6x6 column-major matrix.
        ///
        /// This is useful for storing a symmetric 6x6 matrix in a more compact form and performing some
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
        ///
        /// The 6x6 matrix is represented as:
        ///
        /// ```text
        /// [ A  BT ]
        /// [ B  D  ]
        /// ```
        #[derive(Clone, Copy, Debug)]
        #[cfg_attr(feature = "bevy_reflect", derive($reflect_trait))]
        #[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
        pub struct $n {
            /// The bottom left triangle of the top left 3x3 block of the matrix,
            /// including the diagonal.
            pub a: $symmetricm3t,
            /// The bottom left 3x3 block of the matrix.
            pub b: $m3t,
            /// The bottom left triangle of the bottom right 3x3 block of the matrix,
            /// including the diagonal.
            pub d: $symmetricm3t,
        }

        impl $n {
            /// A symmetric 6x6 matrix with all elements set to `0.0`.
            pub const ZERO: Self = Self::new(
                $symmetricm3t::ZERO,
                $m3t::ZERO,
                $symmetricm3t::ZERO,
            );

            /// A symmetric 6x6 identity matrix, where all diagonal elements are `1.0`,
            /// and all off-diagonal elements are `0.0`.
            pub const IDENTITY: Self = Self::new(
                $symmetricm3t::IDENTITY,
                $m3t::ZERO,
                $symmetricm3t::IDENTITY,
            );

            /// All NaNs.
            pub const NAN: Self = Self::new(
                $symmetricm3t::NAN,
                $m3t::NAN,
                $symmetricm3t::NAN,
            );

            /// Creates a new symmetric 6x6 matrix from its bottom left triangle, including diagonal elements.
            ///
            /// The matrix is represented as:
            ///
            /// ```text
            /// [ A  BT ]
            /// [ B  D  ]
            /// ```
            #[inline(always)]
            #[must_use]
            pub const fn new(
                a: $symmetricm3t,
                b: $m3t,
                d: $symmetricm3t,
            ) -> Self {
                Self { a, b, d }
            }

            /// Creates a new symmetric 6x6 matrix from the outer product `[v1, v2] * [v1, v2]^T`.
            #[inline]
            #[must_use]
            pub fn from_outer_product(v1: $v3t, v2:$v3t) -> Self {
                Self::new(
                    $symmetricm3t::from_outer_product(v1),
                    $m3t::from_outer_product(v1, v2),
                    $symmetricm3t::from_outer_product(v2),
                )
            }

            /// Takes the absolute value of each element in `self`.
            #[inline]
            #[must_use]
            pub fn abs(&self) -> Self {
                Self::new(
                    self.a.abs(),
                    self.b.abs(),
                    self.d.abs(),
                )
            }

            /// Transforms a 6x1 vector that is split into two 3x1 vectors.
            #[inline]
            #[must_use]
            pub fn mul_vec6(&self, rhs1: $v3t, rhs2: $v3t) -> ($v3t, $v3t) {
                let res1 = $v3t::new(
                    rhs1.x * self.a.m00 + rhs1.y * self.a.m01 + rhs1.z * self.a.m02 + rhs2.dot(self.b.row(0)),
                    rhs1.x * self.a.m01 + rhs1.y * self.a.m11 + rhs1.z * self.a.m12 + rhs2.dot(self.b.row(1)),
                    rhs1.x * self.a.m02 + rhs1.y * self.a.m12 + rhs1.z * self.a.m22 + rhs2.dot(self.b.row(2)),
                );
                let res2 = $v3t::new(
                    rhs1.dot(self.b.row(0)) + rhs2.x * self.d.m00 + rhs2.y * self.d.m01 + rhs2.z * self.d.m02,
                    rhs1.dot(self.b.row(1)) + rhs2.x * self.d.m01 + rhs2.y * self.d.m11 + rhs2.z * self.d.m12,
                    rhs1.dot(self.b.row(2)) + rhs2.x * self.d.m02 + rhs2.y * self.d.m12 + rhs2.z * self.d.m22,
                );
                (res1, res2)
            }

            /// Solves `self * [x1, x2] = [rhs1, rhs2]` for `x1` and `x2` using the LDLT decomposition.
            ///
            /// `self` must be a positive semidefinite matrix.
            #[inline]
            #[must_use]
            pub fn ldlt_solve(&self, rhs1: $v3t, rhs2: $v3t) -> ($v3t, $v3t) {
                let (a, b, d) = (self.a, self.b, self.d);

                // Reference: Symmetric6x6Wide in Bepu
                // https://github.com/bepu/bepuphysics2/blob/bfb11dc2020555b09978c473d9655509e844032c/BepuUtilities/Symmetric6x6Wide.cs#L84

                let d1 = a.m00;
                let inv_d1 = $t::ONE / d1;
                let l21 = inv_d1 * a.m01;
                let l31 = inv_d1 * a.m02;
                let l41 = inv_d1 * b.x_axis.x;
                let l51 = inv_d1 * b.y_axis.x;
                let l61 = inv_d1 * b.z_axis.x;
                let d2 = a.m11 - l21 * l21 * d1;
                let inv_d2 = $t::ONE / d2;
                let l32 = inv_d2 * (a.m12 - l21 * l31 * d1);
                let l42 = inv_d2 * (b.x_axis.y - l21 * l41 * d1);
                let l52 = inv_d2 * (b.y_axis.y - l21 * l51 * d1);
                let l62 = inv_d2 * (b.z_axis.y - l21 * l61 * d1);
                let d3 = a.m22 - l31 * l31 * d1 - l32 * l32 * d2;
                let inv_d3 = $t::ONE / d3;
                let l43 = inv_d3 * (b.x_axis.z - l31 * l41 * d1 - l32 * l42 * d2);
                let l53 = inv_d3 * (b.y_axis.z - l31 * l51 * d1 - l32 * l52 * d2);
                let l63 = inv_d3 * (b.z_axis.z - l31 * l61 * d1 - l32 * l62 * d2);
                let d4 = d.m00 - l41 * l41 * d1 - l42 * l42 * d2 - l43 * l43 * d3;
                let inv_d4 = $t::ONE / d4;
                let l54 = inv_d4 * (d.m01 - l41 * l51 * d1 - l42 * l52 * d2 - l43 * l53 * d3);
                let l64 = inv_d4 * (d.m02 - l41 * l61 * d1 - l42 * l62 * d2 - l43 * l63 * d3);
                let d5 = d.m11 - l51 * l51 * d1 - l52 * l52 * d2 - l53 * l53 * d3 - l54 * l54 * d4;
                let inv_d5 = $t::ONE / d5;
                let l65 = inv_d5 * (d.m12 - l51 * l61 * d1 - l52 * l62 * d2 - l53 * l63 * d3 - l54 * l64 * d4);
                let d6 = d.m22 - l61 * l61 * d1 - l62 * l62 * d2 - l63 * l63 * d3 - l64 * l64 * d4 - l65 * l65 * d5;
                let inv_d6 = $t::ONE / d6;

                // We now have the components of L and D, so we can solve the system.
                let mut x1 = rhs1;
                let mut x2 = rhs2;
                x1.y -= l21 * x1.x;
                x1.z -= l31 * x1.x + l32 * x1.y;
                x2.x -= l41 * x1.x + l42 * x1.y + l43 * x1.z;
                x2.y -= l51 * x1.x + l52 * x1.y + l53 * x1.z + l54 * x2.x;
                x2.z -= l61 * x1.x + l62 * x1.y + l63 * x1.z + l64 * x2.x + l65 * x2.y;

                x2.z *= inv_d6;
                x2.y = x2.y * inv_d5 - l65 * x2.z;
                x2.x = x2.x * inv_d4 - l64 * x2.z - l54 * x2.y;
                x1.z = x1.z * inv_d3 - l63 * x2.z - l53 * x2.y - l43 * x2.x;
                x1.y = x1.y * inv_d2 - l62 * x2.z - l52 * x2.y - l42 * x2.x - l32 * x1.z;
                x1.x = x1.x * inv_d1 - l61 * x2.z - l51 * x2.y - l41 * x2.x - l31 * x1.z - l21 * x1.y;

                (x1, x2)
            }

            /// Adds two 6x6 matrices.
            #[inline]
            #[must_use]
            pub fn add_symmetric_mat3(&self, rhs: &Self) -> Self {
                Self::new(
                    self.a.add_symmetric_mat3(&rhs.a),
                    self.b.add_mat3(&rhs.b),
                    self.d.add_symmetric_mat3(&rhs.d),
                )
            }

            /// Subtracts two 6x6 matrices.
            #[inline]
            #[must_use]
            pub fn sub_symmetric_mat3(&self, rhs: &Self) -> Self {
                Self::new(
                    self.a.sub_symmetric_mat3(&rhs.a),
                    self.b.sub_mat3(&rhs.b),
                    self.d.sub_symmetric_mat3(&rhs.d),
                )
            }

            /// Multiplies a 6x6 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn mul_scalar(&self, rhs: $t) -> Self {
                Self::new(
                    self.a.mul_scalar(rhs),
                    self.b.mul_scalar(rhs),
                    self.d.mul_scalar(rhs),
                )
            }

            /// Divides a 6x6 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn div_scalar(&self, rhs: $t) -> Self {
                Self::new(
                    self.a.div_scalar(rhs),
                    self.b.div_scalar(rhs),
                    self.d.div_scalar(rhs),
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

        impl Neg for $n {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self::new(
                    -self.a,
                    -self.b,
                    -self.d,
                )
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

macro_rules! impl_scalar_symmetric_mat6s {
    ($($n:ident),+) => {
        $(
        impl $n {
            /// Returns `true` if, and only if, all elements are finite.
            /// If any element is either `NaN` or positive or negative infinity, this will return `false`.
            #[inline]
            #[must_use]
            pub fn is_finite(&self) -> bool {
                self.a.is_finite()
                    && self.b.is_finite()
                    && self.d.is_finite()
            }

            /// Returns `true` if any elements are `NaN`.
            #[inline]
            #[must_use]
            pub fn is_nan(&self) -> bool {
                self.a.is_nan()
                    || self.b.is_nan()
                    || self.d.is_nan()
            }
        }
        )+
    }
}

macro_rules! impl_wide_symmetric_mat6s {
    ($($n:ident => $nonwiden:ident, $widesymmetricm3t:ident, $widem3t:ident, $nonwidesymmetricm3t:ident, $nonwidem3t:ident),+) => {
        $(
        impl $n {
            /// Creates a new symmetric 6x6 matrix from its bottom left triangle, including diagonal elements,
            /// with all lanes set to the same values.
            ///
            /// The matrix is represented as:
            ///
            /// ```text
            /// [ A  BT ]
            /// [ B  D  ]
            /// ```
            #[inline]
            #[must_use]
            pub const fn new_splat(
                a: $nonwidesymmetricm3t,
                b: $nonwidem3t,
                d: $nonwidesymmetricm3t,
            ) -> Self {
                Self {
                    a: $widesymmetricm3t::splat(a),
                    b: $widem3t::splat(b),
                    d: $widesymmetricm3t::splat(d),
                }
            }

            /// Creates a new wide symmetric 6x6 matrix with all lanes set to `m`.
            #[inline]
            #[must_use]
            pub const fn splat(m: $nonwiden) -> Self {
                Self {
                    a: $widesymmetricm3t::splat(m.a),
                    b: $widem3t::splat(m.b),
                    d: $widesymmetricm3t::splat(m.d),
                }
            }
        }
        )+
    }
}

symmetric_mat6s!(
    bevy_reflect::Reflect,
    SymmetricMat6 => SymmetricMat3, Mat3, Vec3, f32, f32
);

symmetric_mat6s!(
    bevy_reflect::TypePath,
    SymmetricMat6x4 => SymmetricMat3x4, Mat3x4, Vec3x4, f32x4, f32,
    SymmetricMat6x8 => SymmetricMat3x8, Mat3x8, Vec3x8, f32x8, f32
);

#[cfg(feature = "f64")]
symmetric_mat6s!(
    bevy_reflect::Reflect,
    DSymmetricMat6 => DSymmetricMat3, DMat3, DVec3, f64, f64
);

#[cfg(feature = "f64")]
symmetric_mat6s!(
    bevy_reflect::TypePath,
    DSymmetricMat6x2 => DSymmetricMat3x2, DMat3x2, DVec3x2, f64x2, f64,
    DSymmetricMat6x4 => DSymmetricMat3x4, DMat3x4, DVec3x4, f64x4, f64
);

impl_scalar_symmetric_mat6s!(SymmetricMat6);

#[cfg(feature = "f64")]
impl_scalar_symmetric_mat6s!(DSymmetricMat6);

impl_wide_symmetric_mat6s!(
    SymmetricMat6x4 => SymmetricMat6, SymmetricMat3x4, Mat3x4, SymmetricMat3, Mat3,
    SymmetricMat6x8 => SymmetricMat6, SymmetricMat3x8, Mat3x8, SymmetricMat3, Mat3
);

#[cfg(feature = "f64")]
impl_wide_symmetric_mat6s!(
    DSymmetricMat6x2 => DSymmetricMat6, DSymmetricMat3x2, DMat3x2, DSymmetricMat3, DMat3,
    DSymmetricMat6x4 => DSymmetricMat6, DSymmetricMat3x4, DMat3x4, DSymmetricMat3, DMat3
);

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use bevy_math::{Mat3, Vec3};

    use crate::{SymmetricMat3, SymmetricMat6};

    #[test]
    fn ldlt_solve() {
        let a = SymmetricMat3::new(4.0, 1.0, 5.0, 0.0, 2.0, 6.0);
        let b = Mat3::IDENTITY;
        let d = SymmetricMat3::new(7.0, 0.0, 8.0, 0.0, 0.0, 9.0);
        let sym6 = SymmetricMat6 { a, b, d };

        // Known solution (x1, x2)
        let x1 = Vec3::new(1.0, 2.0, 3.0);
        let x2 = Vec3::new(4.0, 5.0, 6.0);

        // Compute rhs = A * x
        let (rhs1, rhs2) = sym6.mul_vec6(x1, x2);
        assert_eq!(rhs1, Vec3::new(25.0, 12.0, 33.0));
        assert_eq!(rhs2, Vec3::new(77.0, 2.0, 89.0));

        // Solve
        let (sol1, sol2) = sym6.ldlt_solve(rhs1, rhs2);

        // Check solution
        assert_relative_eq!(sol1, x1, epsilon = 1e-4);
        assert_relative_eq!(sol2, x2, epsilon = 1e-4);
    }

    #[test]
    fn ldlt_solve_identity() {
        let sym6 = SymmetricMat6::IDENTITY;

        // Known solution (x1, x2)
        let x1 = Vec3::new(7.0, -3.0, 2.5);
        let x2 = Vec3::new(-1.0, 4.5, 0.0);

        // Compute rhs = A * x
        let (rhs1, rhs2) = sym6.mul_vec6(x1, x2);

        // Solve
        let (sol1, sol2) = sym6.ldlt_solve(rhs1, rhs2);

        // Check solution
        assert_relative_eq!(sol1, x1, epsilon = 1e-6);
        assert_relative_eq!(sol2, x2, epsilon = 1e-6);
    }
}
