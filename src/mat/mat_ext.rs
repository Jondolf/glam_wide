#[cfg(feature = "f64")]
use bevy_math::{DMat2, DMat3, DMat4};
use bevy_math::{Mat2, Mat3, Mat3A, Mat4};

/// An extension trait for matrices.
pub trait MatExt {
    /// Returns `true` if the matrix is symmetric.
    #[must_use]
    fn is_symmetric(&self) -> bool;
}

impl MatExt for Mat2 {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
    }
}

#[cfg(feature = "f64")]
impl MatExt for DMat2 {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
    }
}

impl MatExt for Mat3 {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
            && self.x_axis.z == self.z_axis.x
            && self.y_axis.z == self.z_axis.y
    }
}

#[cfg(feature = "f64")]
impl MatExt for DMat3 {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
            && self.x_axis.z == self.z_axis.x
            && self.y_axis.z == self.z_axis.y
    }
}

impl MatExt for Mat3A {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
            && self.x_axis.z == self.z_axis.x
            && self.y_axis.z == self.z_axis.y
    }
}

impl MatExt for Mat4 {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
            && self.x_axis.z == self.z_axis.x
            && self.x_axis.w == self.w_axis.x
            && self.y_axis.z == self.z_axis.y
            && self.y_axis.w == self.w_axis.y
            && self.z_axis.w == self.w_axis.z
    }
}

#[cfg(feature = "f64")]
impl MatExt for DMat4 {
    #[inline]
    fn is_symmetric(&self) -> bool {
        self.x_axis.y == self.y_axis.x
            && self.x_axis.z == self.z_axis.x
            && self.x_axis.w == self.w_axis.x
            && self.y_axis.z == self.z_axis.y
            && self.y_axis.w == self.w_axis.y
            && self.z_axis.w == self.w_axis.z
    }
}
