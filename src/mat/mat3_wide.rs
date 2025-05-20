use core::iter::{Product, Sum};
use core::ops::*;
use wide::{f32x4, f32x8};
#[cfg(feature = "f64")]
use wide::{f64x2, f64x4};

#[cfg(feature = "f64")]
use crate::{DMat2x2, DMat2x4, DQuatx2, DQuatx4, DVec2x2, DVec2x4, DVec3x2, DVec3x4};
use crate::{Mat2x4, Mat2x8, Quatx4, Quatx8, Vec2x4, Vec2x8, Vec3x4, Vec3x8};

macro_rules! wide_mat3s {
    ($($n:ident => $qt:ident, $m2t:ident, $v2t:ident, $vt:ident, $t:ident),+) => {
        $(
        /// A wide 3x3 column major matrix.
        ///
        /// This 3x3 matrix type features convenience methods for creating and using linear and
        /// affine transformations.
        ///
        /// Linear transformations including 3D rotation and scale can be created using methods
        /// such as [`Self::from_diagonal()`], [`Self::from_quat()`], [`Self::from_axis_angle()`],
        /// [`Self::from_rotation_x()`], [`Self::from_rotation_y()`], or
        /// [`Self::from_rotation_z()`].
        ///
        /// The resulting matrices can be use to transform 3D vectors using regular vector
        /// multiplication.
        ///
        /// Affine transformations including 2D translation, rotation and scale can be created
        /// using methods such as [`Self::from_translation()`], [`Self::from_angle()`],
        /// [`Self::from_scale()`] and [`Self::from_scale_angle_translation()`].
        ///
        /// The [`Self::transform_point2()`] and [`Self::transform_vector2()`] convenience methods
        /// are provided for performing affine transforms on 2D vectors and points. These multiply
        /// 2D inputs as 3D vectors with an implicit `z` value of `1.0` for points and `0.0` for
        /// vectors respectively. These methods assume that `Self` contains a valid affine
        /// transform.
        #[derive(Clone, Copy, Debug)]
        #[repr(C)]
        pub struct $n {
            /// The first column of the matrix.
            pub x_axis: $vt,
            /// The second column of the matrix.
            pub y_axis: $vt,
            /// The third column of the matrix.
            pub z_axis: $vt,
        }

        impl $n {
            /// A 3x3 matrix with all elements set to `0.0`.
            pub const ZERO: Self = Self::from_cols($vt::ZERO, $vt::ZERO, $vt::ZERO);

            /// A 3x3 identity matrix, where all diagonal elements are `1.0`,
            /// and all off-diagonal elements are `0.0`.
            pub const IDENTITY: Self = Self::from_cols($vt::X, $vt::Y, $vt::Z);

            /// All NaNs.
            pub const NAN: Self = Self::from_cols($vt::NAN, $vt::NAN, $vt::NAN);

            #[expect(clippy::too_many_arguments)]
            #[inline(always)]
            #[must_use]
            const fn new(
                m00: $t,
                m01: $t,
                m02: $t,
                m10: $t,
                m11: $t,
                m12: $t,
                m20: $t,
                m21: $t,
                m22: $t,
            ) -> Self {
                $n {
                    x_axis: $vt::new(m00, m01, m02),
                    y_axis: $vt::new(m10, m11, m12),
                    z_axis: $vt::new(m20, m21, m22),
                }
            }

            /// Creates a 3x3 matrix from three column vectors.
            #[inline(always)]
            #[must_use]
            pub const fn from_cols(x_axis: $vt, y_axis: $vt, z_axis: $vt) -> Self {
                Self {
                    x_axis,
                    y_axis,
                    z_axis,
                }
            }

            /// Creates a 3x3 matrix from an array stored in column major order.
            /// If your data is stored in row major you will need to `transpose` the returned
            /// matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array(m: &[$t; 9]) -> Self {
                Self::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
            }

            /// Creates an array storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline]
            #[must_use]
            pub const fn to_cols_array(&self) -> [$t; 9] {
                [
                    self.x_axis.x,
                    self.x_axis.y,
                    self.x_axis.z,
                    self.y_axis.x,
                    self.y_axis.y,
                    self.y_axis.z,
                    self.z_axis.x,
                    self.z_axis.y,
                    self.z_axis.z,
                ]
            }

            /// Creates a 3x3 matrix from a 2D array stored in column major order.
            /// If your data is in row major order you will need to `transpose` the returned
            /// matrix.
            #[inline]
            #[must_use]
            pub const fn from_cols_array_2d(m: &[[$t; 3]; 3]) -> Self {
                Self::from_cols(
                    $vt::from_array(m[0]),
                    $vt::from_array(m[1]),
                    $vt::from_array(m[2]),
                )
            }

            /// Creates a 2D array storing data in column major order.
            /// If you require data in row major order `transpose` the matrix first.
            #[inline]
            #[must_use]
            pub const fn to_cols_array_2d(&self) -> [[$t; 3]; 3] {
                [
                    self.x_axis.to_array(),
                    self.y_axis.to_array(),
                    self.z_axis.to_array(),
                ]
            }

            /// Creates a 3x3 matrix with its diagonal set to `diagonal` and all other entries set to `0.0`.
            #[doc(alias = "scale")]
            #[inline]
            #[must_use]
            pub const fn from_diagonal(diagonal: $vt) -> Self {
                Self::new(
                    diagonal.x, $t::ZERO, $t::ZERO, $t::ZERO, diagonal.y, $t::ZERO, $t::ZERO, $t::ZERO, diagonal.z,
                )
            }

            /// Creates a 3D rotation matrix from the given quaternion.
            ///
            /// The quaternion must be normalized for the rotation matrix to be valid.
            #[inline]
            #[must_use]
            pub fn from_quat(rotation: $qt) -> Self {
                let x2 = rotation.x + rotation.x;
                let y2 = rotation.y + rotation.y;
                let z2 = rotation.z + rotation.z;
                let xx = rotation.x * x2;
                let xy = rotation.x * y2;
                let xz = rotation.x * z2;
                let yy = rotation.y * y2;
                let yz = rotation.y * z2;
                let zz = rotation.z * z2;
                let wx = rotation.w * x2;
                let wy = rotation.w * y2;
                let wz = rotation.w * z2;

                Self::from_cols(
                    $vt::new($t::ONE - (yy + zz), xy + wz, xz - wy),
                    $vt::new(xy - wz, $t::ONE - (xx + zz), yz + wx),
                    $vt::new(xz + wy, yz - wx, $t::ONE - (xx + yy)),
                )
            }

            /// Creates a 3D rotation matrix from a normalized rotation `axis` and `angle` (in
            /// radians).
            ///
            /// The `axis` must be normalized for the rotation matrix to be valid.
            #[inline]
            #[must_use]
            pub fn from_axis_angle(axis: $vt, angle: $t) -> Self {
                let (sin, cos) = (angle).sin_cos();
                let (xsin, ysin, zsin) = axis.mul(sin).into();
                let (x, y, z) = axis.into();
                let (x2, y2, z2) = axis.mul(axis).into();
                let omc = $t::ONE - cos;
                let xyomc = x * y * omc;
                let xzomc = x * z * omc;
                let yzomc = y * z * omc;
                Self::from_cols(
                    $vt::new(x2 * omc + cos, xyomc + zsin, xzomc - ysin),
                    $vt::new(xyomc - zsin, y2 * omc + cos, yzomc + xsin),
                    $vt::new(xzomc + ysin, yzomc - xsin, z2 * omc + cos),
                )
            }

            /// Creates a 3D rotation matrix from `angle` (in radians) around the x axis.
            #[inline]
            #[must_use]
            pub fn from_rotation_x(angle: $t) -> Self {
                let (sina, cosa) = angle.sin_cos();
                Self::from_cols(
                    $vt::X,
                    $vt::new($t::ZERO, cosa, sina),
                    $vt::new($t::ZERO, -sina, cosa),
                )
            }

            /// Creates a 3D rotation matrix from `angle` (in radians) around the y axis.
            #[inline]
            #[must_use]
            pub fn from_rotation_y(angle: $t) -> Self {
                let (sina, cosa) = angle.sin_cos();
                Self::from_cols(
                    $vt::new(cosa, $t::ZERO, -sina),
                    $vt::Y,
                    $vt::new(sina, $t::ZERO, cosa),
                )
            }

            /// Creates a 3D rotation matrix from `angle` (in radians) around the z axis.
            #[inline]
            #[must_use]
            pub fn from_rotation_z(angle: $t) -> Self {
                let (sina, cosa) = angle.sin_cos();
                Self::from_cols(
                    $vt::new(cosa, sina, $t::ZERO),
                    $vt::new(-sina, cosa, $t::ZERO),
                    $vt::Z,
                )
            }

            /// Creates an affine transformation matrix from the given 2D `translation`.
            ///
            /// The resulting matrix can be used to transform 2D points and vectors. See
            /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
            #[inline]
            #[must_use]
            pub fn from_translation(translation: $v2t) -> Self {
                Self::from_cols(
                    $vt::X,
                    $vt::Y,
                    $vt::new(translation.x, translation.y, $t::ONE),
                )
            }

            /// Creates an affine transformation matrix from the given 2D rotation `angle` (in
            /// radians).
            ///
            /// The resulting matrix can be used to transform 2D points and vectors. See
            /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
            #[inline]
            #[must_use]
            pub fn from_angle(angle: $t) -> Self {
                let (sin, cos) = angle.sin_cos();
                Self::from_cols($vt::new(cos, sin, $t::ZERO), $vt::new(-sin, cos, $t::ZERO), $vt::Z)
            }

            /// Creates an affine transformation matrix from the given 2D `scale`, rotation `angle` (in
            /// radians) and `translation`.
            ///
            /// The resulting matrix can be used to transform 2D points and vectors. See
            /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
            #[inline]
            #[must_use]
            pub fn from_scale_angle_translation(scale: $v2t, angle: $t, translation: $v2t) -> Self {
                let (sin, cos) = angle.sin_cos();
                Self::from_cols(
                    $vt::new(cos * scale.x, sin * scale.x, $t::ZERO),
                    $vt::new(-sin * scale.y, cos * scale.y, $t::ZERO),
                    $vt::new(translation.x, translation.y, $t::ONE),
                )
            }

            /// Creates an affine transformation matrix from the given non-uniform 2D `scale`.
            ///
            /// The resulting matrix can be used to transform 2D points and vectors. See
            /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
            #[inline]
            #[must_use]
            pub fn from_scale(scale: $v2t) -> Self {
                Self::from_cols(
                    $vt::new(scale.x, $t::ZERO, $t::ZERO),
                    $vt::new($t::ZERO, scale.y, $t::ZERO),
                    $vt::Z,
                )
            }

            /// Creates an affine transformation matrix from the given 2x2 matrix.
            ///
            /// The resulting matrix can be used to transform 2D points and vectors. See
            /// [`Self::transform_point2()`] and [`Self::transform_vector2()`].
            #[inline]
            pub fn from_mat2(m: $m2t) -> Self {
                Self::from_cols(m.x_axis.extend($t::ZERO), m.y_axis.extend($t::ZERO), $vt::Z)
            }

            /// Creates a 3x3 matrix from the first 9 values in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 9 elements long.
            #[inline]
            #[must_use]
            pub const fn from_cols_slice(slice: &[$t]) -> Self {
                Self::new(
                    slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7], slice[8],
                )
            }

            /// Writes the columns of `self` to the first 9 elements in `slice`.
            ///
            /// # Panics
            ///
            /// Panics if `slice` is less than 9 elements long.
            #[inline]
            pub fn write_cols_to_slice(self, slice: &mut [$t]) {
                slice[0] = self.x_axis.x;
                slice[1] = self.x_axis.y;
                slice[2] = self.x_axis.z;
                slice[3] = self.y_axis.x;
                slice[4] = self.y_axis.y;
                slice[5] = self.y_axis.z;
                slice[6] = self.z_axis.x;
                slice[7] = self.z_axis.y;
                slice[8] = self.z_axis.z;
            }

            /// Returns the matrix column for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 2.
            #[inline]
            #[must_use]
            pub fn col(&self, index: usize) -> $vt {
                match index {
                    0 => self.x_axis,
                    1 => self.y_axis,
                    2 => self.z_axis,
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns a mutable reference to the matrix column for the given `index`.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than 2.
            #[inline]
            pub fn col_mut(&mut self, index: usize) -> &mut $vt {
                match index {
                    0 => &mut self.x_axis,
                    1 => &mut self.y_axis,
                    2 => &mut self.z_axis,
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
            pub fn row(&self, index: usize) -> $vt {
                match index {
                    0 => $vt::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
                    1 => $vt::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
                    2 => $vt::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
                    _ => panic!("index out of bounds"),
                }
            }

            /// Returns the transpose of `self`.
            #[inline]
            #[must_use]
            pub fn transpose(&self) -> Self {
                Self {
                    x_axis: $vt::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
                    y_axis: $vt::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
                    z_axis: $vt::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
                }
            }

            /// Returns the determinant of `self`.
            #[inline]
            #[must_use]
            pub fn determinant(&self) -> $t {
                self.z_axis.dot(self.x_axis.cross(self.y_axis))
            }

            /// Returns the inverse of `self`.
            ///
            /// If the matrix is not invertible the returned matrix will be invalid.
            #[inline]
            #[must_use]
            pub fn inverse(&self) -> Self {
                let tmp0 = self.y_axis.cross(self.z_axis);
                let tmp1 = self.z_axis.cross(self.x_axis);
                let tmp2 = self.x_axis.cross(self.y_axis);
                let det = self.z_axis.dot(tmp2);
                let inv_det = $t::ONE / det;
                let inv_det = $vt::new(inv_det, inv_det, inv_det);
                Self::from_cols(tmp0.mul(inv_det), tmp1.mul(inv_det), tmp2.mul(inv_det)).transpose()
            }

            /// Transforms the given 2D vector as a point.
            ///
            /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `1.0`.
            ///
            /// This method assumes that `self` contains a valid affine transform.
            #[inline]
            #[must_use]
            pub fn transform_point2(&self, rhs: $v2t) -> $v2t {
                $m2t::from_cols(self.x_axis.truncate(), self.y_axis.truncate()) * rhs + self.z_axis.truncate()
            }

            /// Rotates the given 2D vector.
            ///
            /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `0.0`.
            ///
            /// This method assumes that `self` contains a valid affine transform.
            #[inline]
            #[must_use]
            pub fn transform_vector2(&self, rhs: $v2t) -> $v2t {
                $m2t::from_cols(self.x_axis.truncate(), self.y_axis.truncate()) * rhs
            }

            /// Creates a left-handed view matrix using a facing direction and an up direction.
            ///
            /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
            ///
            /// `dir` and `up` must be normalized.
            #[inline]
            #[must_use]
            pub fn look_to_lh(dir: $vt, up: $vt) -> Self {
                Self::look_to_rh(-dir, up)
            }

            /// Creates a right-handed view matrix using a facing direction and an up direction.
            ///
            /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
            ///
            /// `dir` and `up` must be normalized.
            #[inline]
            #[must_use]
            pub fn look_to_rh(dir: $vt, up: $vt) -> Self {
                let f = dir;
                let s = f.cross(up).normalize();
                let u = s.cross(f);

                Self::from_cols(
                    $vt::new(s.x, u.x, -f.x),
                    $vt::new(s.y, u.y, -f.y),
                    $vt::new(s.z, u.z, -f.z),
                )
            }

            /// Creates a left-handed view matrix using a camera position, a focal point and an up
            /// direction.
            ///
            /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=forward`.
            ///
            /// `up` must be normalized.
            #[inline]
            #[must_use]
            pub fn look_at_lh(eye: $vt, center: $vt, up: $vt) -> Self {
                Self::look_to_lh(center.sub(eye).normalize(), up)
            }

            /// Creates a right-handed view matrix using a camera position, a focal point and an up
            /// direction.
            ///
            /// For a view coordinate system with `+X=right`, `+Y=up` and `+Z=back`.
            ///
            /// `up` must be normalized.
            #[inline]
            pub fn look_at_rh(eye: $vt, center: $vt, up: $vt) -> Self {
                Self::look_to_rh(center.sub(eye).normalize(), up)
            }

            /// Transforms a 3D vector.
            #[inline]
            #[must_use]
            pub fn mul_vec3(&self, rhs: $vt) -> $vt {
                let mut res = self.x_axis.mul(rhs.x);
                res = res.add(self.y_axis.mul(rhs.y));
                res = res.add(self.z_axis.mul(rhs.z));
                res
            }

            /// Multiplies two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn mul_mat3(&self, rhs: &Self) -> Self {
                Self::from_cols(
                    self.mul(rhs.x_axis),
                    self.mul(rhs.y_axis),
                    self.mul(rhs.z_axis),
                )
            }

            /// Adds two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn add_mat3(&self, rhs: &Self) -> Self {
                Self::from_cols(
                    self.x_axis.add(rhs.x_axis),
                    self.y_axis.add(rhs.y_axis),
                    self.z_axis.add(rhs.z_axis),
                )
            }

            /// Subtracts two 3x3 matrices.
            #[inline]
            #[must_use]
            pub fn sub_mat3(&self, rhs: &Self) -> Self {
                Self::from_cols(
                    self.x_axis.sub(rhs.x_axis),
                    self.y_axis.sub(rhs.y_axis),
                    self.z_axis.sub(rhs.z_axis),
                )
            }

            /// Multiplies a 3x3 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn mul_scalar(&self, rhs: $t) -> Self {
                Self::from_cols(
                    self.x_axis.mul(rhs),
                    self.y_axis.mul(rhs),
                    self.z_axis.mul(rhs),
                )
            }

            /// Divides a 3x3 matrix by a scalar.
            #[inline]
            #[must_use]
            pub fn div_scalar(&self, rhs: $t) -> Self {
                let rhs = $vt::new(rhs, rhs, rhs);
                Self::from_cols(
                    self.x_axis.div(rhs),
                    self.y_axis.div(rhs),
                    self.z_axis.div(rhs),
                )
            }

            /// Takes the absolute value of each element in `self`
            #[inline]
            #[must_use]
            pub fn abs(&self) -> Self {
                Self::from_cols(self.x_axis.abs(), self.y_axis.abs(), self.z_axis.abs())
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
                self.add_mat3(&rhs)
            }
        }

        impl AddAssign for $n {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = self.add_mat3(&rhs);
            }
        }

        impl Sub for $n {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                self.sub_mat3(&rhs)
            }
        }

        impl SubAssign for $n {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.sub_mat3(&rhs);
            }
        }

        impl Neg for $n {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                Self::from_cols(self.x_axis.neg(), self.y_axis.neg(), self.z_axis.neg())
            }
        }

        impl Mul for $n {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                self.mul_mat3(&rhs)
            }
        }

        impl MulAssign for $n {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = self.mul_mat3(&rhs);
            }
        }

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

wide_mat3s!(
    Mat3x4 => Quatx4, Mat2x4, Vec2x4, Vec3x4, f32x4,
    Mat3x8 => Quatx8, Mat2x8, Vec2x8, Vec3x8, f32x8
);

#[cfg(feature = "f64")]
wide_mat3s!(
    DMat3x2 => DQuatx2, DMat2x2, DVec2x2, DVec3x2, f64x2,
    DMat3x4 => DQuatx4, DMat2x4, DVec2x4, DVec3x4, f64x4
);
