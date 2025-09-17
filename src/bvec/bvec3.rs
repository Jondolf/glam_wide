use core::ops::*;
use glam::BVec3;

use crate::SimdLaneCount;
#[cfg(feature = "f32")]
use crate::{boolf32x4, boolf32x8};
#[cfg(feature = "f64")]
use crate::{boolf64x2, boolf64x4};

macro_rules! wide_bvec3s {
    ($(($nonwiden:ident, $n:ident) => $t:ident),*) => {
        $(
        /// A 3-dimensional wide `bool` vector mask.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $n {
            /// The X component of the vector mask.
            pub x: $t,
            /// The Y component of the vector mask.
            pub y: $t,
            /// The Z component of the vector mask.
            pub z: $t,
        }

        impl $n {
            /// All false.
            pub const FALSE: Self = Self::new($t::FALSE, $t::FALSE, $t::FALSE);

            /// Creates a new vector mask.
            #[inline(always)]
            #[must_use]
            pub const fn new(x: $t, y: $t, z: $t) -> Self {
                Self { x, y, z }
            }

            /// Creates a new vector mask with all lanes set to the same `x`, `y`, and `z` values.
            #[inline]
            #[must_use]
            pub const fn new_splat(x: bool, y: bool, z: bool) -> Self {
                Self {
                    x: $t::new([x; $t::LANES]),
                    y: $t::new([y; $t::LANES]),
                    z: $t::new([z; $t::LANES]),
                }
            }

            /// Creates a new vector mask with all lanes set to `v`.
            #[inline]
            #[must_use]
            pub const fn splat(v: $nonwiden) -> Self {
                Self {
                    x: $t::new([v.x; $t::LANES]),
                    y: $t::new([v.y; $t::LANES]),
                    z: $t::new([v.z; $t::LANES]),
                }
            }

            /// Blend two vector masks together lanewise using `mask` as a mask.
            ///
            /// This is essentially a bitwise blend operation, such that any point where
            /// there is a 1 bit in `mask`, the output will put the bit from `if_true`, while
            /// where there is a 0 bit in `mask`, the output will put the bit from `if_false`.
            #[inline]
            #[must_use]
            pub fn blend(mask: $t, if_true: Self, if_false: Self) -> Self {
                Self {
                    x: mask.blend(if_true.x, if_false.x),
                    y: mask.blend(if_true.y, if_false.y),
                    z: mask.blend(if_true.z, if_false.z),
                }
            }

            /*
            /// Returns a bitmask with the lowest 3 bits set from the elements of `self`.
            ///
            /// A true element results in a `1` bit and a false element in a `0` bit.  Element `x` goes
            /// into the first lowest bit, element `y` into the second, etc.
            #[inline]
            #[must_use]
            pub fn bitmask(self) -> u32x4 {
                unimplemented!()
            }
            */

            /// Returns true if any of the elements are true, false otherwise.
            #[inline]
            #[must_use]
            pub fn any(self) -> $t {
                self.x | self.y | self.z
            }

            /// Returns true if all the elements are true, false otherwise.
            #[inline]
            #[must_use]
            pub fn all(self) -> $t {
                self.x & self.y & self.z
            }

            /// Tests the value at `index`.
            ///
            /// Panics if `index` is greater than 2.
            #[inline]
            #[must_use]
            pub fn test(&self, index: usize) -> $t {
                match index {
                    0 => self.x,
                    1 => self.y,
                    2 => self.z,
                    _ => panic!("index out of bounds"),
                }
            }

            /// Sets the element at `index`.
            ///
            /// Panics if `index` is greater than 2.
            #[inline]
            pub fn set(&mut self, index: usize, value: $t) {
                match index {
                    0 => self.x = value,
                    1 => self.y = value,
                    2 => self.z = value,
                    _ => panic!("index out of bounds"),
                }
            }

            #[inline]
            #[must_use]
            fn into_bool_array(self) -> [$t; 3] {
                [self.x, self.y, self.z]
            }
        }

        impl Default for $n {
            #[inline]
            fn default() -> Self {
                Self::FALSE
            }
        }

        impl BitAnd for $n {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                Self {
                    x: self.x & rhs.x,
                    y: self.y & rhs.y,
                    z: self.z & rhs.z,
                }
            }
        }

        impl BitAnd<&Self> for $n {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: &Self) -> Self {
                self.bitand(*rhs)
            }
        }

        impl BitAnd<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitand(self, rhs: &$n) -> $n {
                (*self).bitand(*rhs)
            }
        }

        impl BitAnd<$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitand(self, rhs: $n) -> $n {
                (*self).bitand(rhs)
            }
        }

        impl BitAndAssign for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.bitand(rhs);
            }
        }

        impl BitAndAssign<&Self> for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: &Self) {
                self.bitand_assign(*rhs);
            }
        }

        impl BitOr for $n {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                Self {
                    x: self.x | rhs.x,
                    y: self.y | rhs.y,
                    z: self.z | rhs.z,
                }
            }
        }

        impl BitOr<&Self> for $n {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: &Self) -> Self {
                self.bitor(*rhs)
            }
        }

        impl BitOr<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitor(self, rhs: &$n) -> $n {
                (*self).bitor(*rhs)
            }
        }

        impl BitOr<$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitor(self, rhs: $n) -> $n {
                (*self).bitor(rhs)
            }
        }

        impl BitOrAssign for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.bitor(rhs);
            }
        }

        impl BitOrAssign<&Self> for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: &Self) {
                self.bitor_assign(*rhs);
            }
        }

        impl BitXor for $n {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                Self {
                    x: self.x ^ rhs.x,
                    y: self.y ^ rhs.y,
                    z: self.z ^ rhs.z,
                }
            }
        }

        impl BitXor<&Self> for $n {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: &Self) -> Self {
                self.bitxor(*rhs)
            }
        }

        impl BitXor<&$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitxor(self, rhs: &$n) -> $n {
                (*self).bitxor(*rhs)
            }
        }

        impl BitXor<$n> for &$n {
            type Output = $n;
            #[inline]
            fn bitxor(self, rhs: $n) -> $n {
                (*self).bitxor(rhs)
            }
        }

        impl BitXorAssign for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.bitxor(rhs);
            }
        }

        impl BitXorAssign<&Self> for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: &Self) {
                self.bitxor_assign(*rhs);
            }
        }

        impl Not for $n {
            type Output = Self;
            #[inline]
            fn not(self) -> Self {
                Self {
                    x: !self.x,
                    y: !self.y,
                    z: !self.z,
                }
            }
        }

        impl Not for &$n {
            type Output = $n;
            #[inline]
            fn not(self) -> $n {
                (*self).not()
            }
        }

        impl From<[$t; 3]> for $n {
            #[inline]
            fn from(a: [$t; 3]) -> Self {
                Self::new(a[0], a[1], a[2])
            }
        }

        impl From<$n> for [$t; 3] {
            #[inline]
            fn from(mask: $n) -> Self {
                mask.into_bool_array()
            }
        }
        )*
    };
}

#[cfg(feature = "f32")]
wide_bvec3s!(
    (BVec3, BVec3x4) => boolf32x4,
    (BVec3, BVec3x8) => boolf32x8
);

#[cfg(feature = "f64")]
wide_bvec3s!(
    (BVec3, BDVec3x2) => boolf64x2,
    (BVec3, BDVec3x4) => boolf64x4
);
