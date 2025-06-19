use bevy_math::{DMat3, DVec3, Mat3, Mat3A, Vec3, Vec3A};

//TODO: We should reduce duplication with a macro if we add more methods.

/// An extension trait for [`Mat3`].
pub trait Mat3Ext {
    /// Creates a new 3x3 matrix from the outer product `v1 * v2^T`.
    #[must_use]
    fn from_outer_product(v1: Vec3, v2: Vec3) -> Self;
}

/// An extension trait for [`Mat3A`].
pub trait Mat3AExt {
    /// Creates a new 3x3 matrix from the outer product `v1 * v2^T`.
    #[must_use]
    fn from_outer_product(v1: Vec3A, v2: Vec3A) -> Self;
}

/// An extension trait for [`DMat3`].
pub trait DMat3Ext {
    /// Creates a new 3x3 matrix from the outer product `v1 * v2^T`.
    #[must_use]
    fn from_outer_product(v1: DVec3, v2: DVec3) -> Self;
}

impl Mat3Ext for Mat3 {
    /// Creates a new 3x3 matrix from the outer product `v1 * v2^T`.
    #[inline]
    fn from_outer_product(v1: Vec3, v2: Vec3) -> Self {
        Self::from_cols_array(&[
            v1.x * v2.x,
            v1.x * v2.y,
            v1.x * v2.z,
            v1.y * v2.x,
            v1.y * v2.y,
            v1.y * v2.z,
            v1.z * v2.x,
            v1.z * v2.y,
            v1.z * v2.z,
        ])
    }
}

impl Mat3AExt for Mat3A {
    /// Creates a new 3x3 matrix from the outer product `v1 * v2^T`.
    #[inline]
    fn from_outer_product(v1: Vec3A, v2: Vec3A) -> Self {
        Self::from_cols_array(&[
            v1.x * v2.x,
            v1.x * v2.y,
            v1.x * v2.z,
            v1.y * v2.x,
            v1.y * v2.y,
            v1.y * v2.z,
            v1.z * v2.x,
            v1.z * v2.y,
            v1.z * v2.z,
        ])
    }
}

impl DMat3Ext for DMat3 {
    /// Creates a new 3x3 matrix from the outer product `v1 * v2^T`.
    #[inline]
    fn from_outer_product(v1: DVec3, v2: DVec3) -> Self {
        Self::from_cols_array(&[
            v1.x * v2.x,
            v1.x * v2.y,
            v1.x * v2.z,
            v1.y * v2.x,
            v1.y * v2.y,
            v1.y * v2.z,
            v1.z * v2.x,
            v1.z * v2.y,
            v1.z * v2.z,
        ])
    }
}
