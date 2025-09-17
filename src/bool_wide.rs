#![expect(non_camel_case_types)]

use core::{
    fmt::{Debug, Display},
    ops::*,
};
use wide::{f32x4, f32x8, f64x2, f64x4};

use crate::SimdLaneCount;

const ALL_ONES_F32: f32 = f32::from_bits(u32::MAX);
const ALL_ONES_F64: f64 = f64::from_bits(u64::MAX);

macro_rules! wide_bools {
    ($(($n:ident, $t:ident, $all_ones:ident; $($ii:expr),+)),*) => {
        $(
        #[doc = concat!("A boolean vector interface for [`", stringify!($t), "`].")]
        #[derive(Clone, Copy)]
        pub struct $n($t);

        impl $n {
            /// False for all lanes.
            pub const FALSE: Self = Self($t::ZERO);

            /// True for all lanes.
            pub const TRUE: Self = Self($t::new([$all_ones; $t::LANES]));

            /// Creates a new boolean vector from a boolean array.
            #[inline(always)]
            #[must_use]
            pub const fn new(array: [bool; $n::LANES]) -> Self {
                Self($t::new([
                    if array[0] { $all_ones } else { 0.0 },
                    $(if array[$ii] { $all_ones } else { 0.0 }),*,
                ]))
            }

            /// Creates a new boolean vector with all lanes set to the given boolean value.
            #[inline]
            #[must_use]
            pub const fn splat(value: bool) -> Self {
                if value {
                    Self::TRUE
                } else {
                    Self::FALSE
                }
            }

            /// Creates a new boolean vector from a raw float vector value.
            #[inline]
            #[must_use]
            pub const fn from_raw(value: $t) -> Self {
                Self(value)
            }

            /// Converts the boolean vector into its raw float vector representation.
            #[inline]
            #[must_use]
            pub const fn to_raw(self) -> $t {
                self.0
            }

            /// Converts the boolean vector into a boolean array.
            #[inline]
            #[must_use]
            pub fn to_array(self) -> [bool; $n::LANES] {
                let array = self.0.to_array();
                [
                    array[0] != 0.0,
                    $(array[$ii] != 0.0),*,
                ]
            }

            /// Converts the boolean vector into a `u32` bitmask with the least significant bits
            /// representing the boolean values of each lane.
            #[inline]
            #[must_use]
            pub fn bitmask(self) -> u32 {
                let array = self.0.to_array();
                (((array[0] != 0.0) as u32) << 0)
                    $(| (((array[$ii] != 0.0) as u32) << $ii))*
            }

            /// Chooses elements from two boolean vectors based on a mask.
            ///
            /// For each element in `self`, the corresponding element from `if_true`
            /// is chosen if the mask element is true, and the corresponding element from
            /// `if_false` is chosen if the mask element is false.
            #[inline]
            #[must_use]
            pub fn blend(self, if_true: Self, if_false: Self) -> Self {
                Self(self.0.blend(if_true.0, if_false.0))
            }

            /// Returns `true` if any lane is `true`, or `false` otherwise.
            #[inline]
            #[must_use]
            pub fn any(self) -> bool {
                self.0.any()
            }

            /// Returns `true` if all lanes are `true`, or `false` otherwise.
            #[inline]
            #[must_use]
            pub fn all(self) -> bool {
                self.0.all()
            }

            /// Returns `true` if no lane is `true`, or `false` otherwise.
            #[inline]
            #[must_use]
            pub fn none(self) -> bool {
                self.0.none()
            }

            /// Tests the value of the specified lane.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than or equal to the number of lanes.
            #[inline]
            #[must_use]
            pub fn test(&self, index: usize) -> bool {
                self.0.to_array()[index] != 0.0
            }

            /// Tests the value of the specified lane, without doing bounds checking.
            ///
            /// # Safety
            ///
            /// Calling this method with an index that is greater than or equal to the number of lanes
            /// is *[undefined behavior]* even if the resulting reference is not used.
            ///
            /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
            #[inline]
            #[must_use]
            pub unsafe fn test_unchecked(&self, index: usize) -> bool {
                unsafe { *self.0.to_array().get_unchecked(index) != 0.0 }
            }

            /// Sets the value of the specified lane.
            ///
            /// # Panics
            ///
            /// Panics if `index` is greater than or equal to the number of lanes.
            #[inline]
            pub fn set(&mut self, index: usize, value: bool) {
                let mut array = self.0.to_array();
                array[index] = if value { -1.0 } else { 0.0 };
                self.0 = $t::new(array);
            }

            /// Sets the value of the specified lane, without doing bounds checking.
            ///
            /// # Safety
            ///
            /// Calling this method with an index that is greater than or equal to the number of lanes
            /// is *[undefined behavior]* even if the resulting reference is not used.
            ///
            /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
            #[inline]
            pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
                unsafe {
                    let mut array = self.0.to_array();
                    *array.get_unchecked_mut(index) = if value { -1.0 } else { 0.0 };
                    self.0 = $t::new(array);
                }
            }
        }

        impl Default for $n {
            #[inline]
            fn default() -> Self {
                Self($t::ZERO)
            }
        }

        impl PartialEq for $n {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                // We don't directly compare the equality of the underlying floats,
                // because `true` values are represented by all bits set, which is NaN.
                self.bitmask() == other.bitmask()
            }
        }

        impl BitAnd for $n {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                Self(self.0 & rhs.0)
            }
        }

        impl BitAnd<bool> for $n {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: bool) -> Self::Output {
                self & Self::splat(rhs)
            }
        }

        impl BitAnd<$n> for bool {
            type Output = $n;

            #[inline]
            fn bitand(self, rhs: $n) -> Self::Output {
                $n::splat(self) & rhs
            }
        }

        impl BitAndAssign for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 = self.0 & rhs.0;
            }
        }

        impl BitAndAssign<bool> for $n {
            #[inline]
            fn bitand_assign(&mut self, rhs: bool) {
                *self &= Self::splat(rhs);
            }
        }

        impl BitOr for $n {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                Self(self.0 | rhs.0)
            }
        }

        impl BitOr<bool> for $n {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: bool) -> Self::Output {
                self | Self::splat(rhs)
            }
        }

        impl BitOr<$n> for bool {
            type Output = $n;

            #[inline]
            fn bitor(self, rhs: $n) -> Self::Output {
                $n::splat(self) | rhs
            }
        }

        impl BitOrAssign for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 = self.0 | rhs.0;
            }
        }

        impl BitOrAssign<bool> for $n {
            #[inline]
            fn bitor_assign(&mut self, rhs: bool) {
                *self |= Self::splat(rhs);
            }
        }

        impl BitXor for $n {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl BitXor<bool> for $n {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: bool) -> Self::Output {
                self ^ Self::splat(rhs)
            }
        }

        impl BitXor<$n> for bool {
            type Output = $n;

            #[inline]
            fn bitxor(self, rhs: $n) -> Self::Output {
                $n::splat(self) ^ rhs
            }
        }

        impl BitXorAssign for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 = self.0 ^ rhs.0;
            }
        }

        impl BitXorAssign<bool> for $n {
            #[inline]
            fn bitxor_assign(&mut self, rhs: bool) {
                *self ^= Self::splat(rhs);
            }
        }

        impl Not for $n {
            type Output = Self;

            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }

        impl From<[bool; $n::LANES]> for $n {
            #[inline]
            fn from(array: [bool; $n::LANES]) -> Self {
                Self::new(array)
            }
        }

        impl From<$n> for [bool; $n::LANES] {
            #[inline]
            fn from(value: $n) -> Self {
                value.to_array()
            }
        }

        impl From<$n> for $t {
            #[inline]
            fn from(value: $n) -> Self {
                value.0
            }
        }

        impl From<$t> for $n {
            #[inline]
            fn from(value: $t) -> Self {
                Self(value)
            }
        }

        impl Debug for $n {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_tuple(stringify!($n))
                    .field(&self.0)
                    .finish()
            }
        }

        impl Display for $n {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                let array: [bool; $n::LANES] = (*self).into();
                f.debug_tuple(stringify!($n))
                    .field(&array)
                    .finish()
            }
        }
        )*
    };
}

wide_bools!(
    (boolf32x4, f32x4, ALL_ONES_F32; 1, 2, 3),
    (boolf32x8, f32x8, ALL_ONES_F32; 1, 2, 3, 4, 5, 6, 7),
    (boolf64x2, f64x2, ALL_ONES_F64; 1),
    (boolf64x4, f64x4, ALL_ONES_F64; 1, 2, 3)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boolf32x4() {
        let a = boolf32x4::new([true, false, true, false]);
        let b = boolf32x4::new([false, false, true, true]);
        let c = a & b;
        assert_eq!(c, boolf32x4::new([false, false, true, false]));
        let d = a | b;
        assert_eq!(d, boolf32x4::new([true, false, true, true]));
        let e = a ^ b;
        assert_eq!(e, boolf32x4::new([true, false, false, true]));
        let f = !a;
        assert_eq!(f, boolf32x4::new([false, true, false, true]));
        assert_eq!(a.bitmask(), 0b0101);
        assert!(a.any());
        assert!(!a.all());
        assert!(!a.none());
        assert!(boolf32x4::splat(true).all());
        assert!(!boolf32x4::splat(false).any());
        assert!(a.test(0));
        assert!(!a.test(1));
        assert!(a.test(2));
        assert!(!a.test(3));
        let mut g = a;
        g.set(1, true);
        assert_eq!(g, boolf32x4::new([true, true, true, false]));
    }

    #[test]
    fn test_boolf64x2() {
        let a = boolf64x2::new([true, false]);
        let b = boolf64x2::new([false, true]);
        let c = a & b;
        assert_eq!(c, boolf64x2::new([false, false]));
        let d = a | b;
        assert_eq!(d, boolf64x2::new([true, true]));
        let e = a ^ b;
        assert_eq!(e, boolf64x2::new([true, true]));
        let f = !a;
        assert_eq!(f, boolf64x2::new([false, true]));
        assert_eq!(a.bitmask(), 0b01);
        assert!(a.any());
        assert!(!a.all());
        assert!(!a.none());
        assert!(boolf64x2::splat(true).all());
        assert!(!boolf64x2::splat(false).any());
        assert!(a.test(0));
        assert!(!a.test(1));
        let mut g = a;
        g.set(1, true);
        assert_eq!(g, boolf64x2::new([true, true]));
    }

    #[test]
    fn test_boolf32x8() {
        let a = boolf32x8::new([true, false, true, false, true, false, true, false]);
        let b = boolf32x8::new([false, false, true, true, false, false, true, true]);
        let c = a & b;
        assert_eq!(
            c,
            boolf32x8::new([false, false, true, false, false, false, true, false])
        );
        let d = a | b;
        assert_eq!(
            d,
            boolf32x8::new([true, false, true, true, true, false, true, true])
        );
        let e = a ^ b;
        assert_eq!(
            e,
            boolf32x8::new([true, false, false, true, true, false, false, true])
        );
        let f = !a;
        assert_eq!(
            f,
            boolf32x8::new([false, true, false, true, false, true, false, true])
        );
        assert_eq!(a.bitmask(), 0b01010101);
        assert!(a.any());
        assert!(!a.all());
        assert!(!a.none());
        assert!(boolf32x8::splat(true).all());
        assert!(!boolf32x8::splat(false).any());
        assert!(a.test(0));
        assert!(!a.test(1));
        assert!(a.test(2));
        assert!(!a.test(3));
        assert!(a.test(4));
        assert!(!a.test(5));
        assert!(a.test(6));
        assert!(!a.test(7));
        let mut g = a;
        g.set(1, true);
        assert_eq!(
            g,
            boolf32x8::new([true, true, true, false, true, false, true, false])
        );
    }

    #[test]
    fn test_boolf64x4() {
        let a = boolf64x4::new([true, false, true, false]);
        let b = boolf64x4::new([false, false, true, true]);
        let c = a & b;
        assert_eq!(c, boolf64x4::new([false, false, true, false]));
        let d = a | b;
        assert_eq!(d, boolf64x4::new([true, false, true, true]));
        let e = a ^ b;
        assert_eq!(e, boolf64x4::new([true, false, false, true]));
        let f = !a;
        assert_eq!(f, boolf64x4::new([false, true, false, true]));
        assert_eq!(a.bitmask(), 0b0101);
        assert!(a.any());
        assert!(!a.all());
        assert!(!a.none());
        assert!(boolf64x4::splat(true).all());
        assert!(!boolf64x4::splat(false).any());
        assert!(a.test(0));
        assert!(!a.test(1));
        assert!(a.test(2));
        assert!(!a.test(3));
        let mut g = a;
        g.set(1, true);
        assert_eq!(g, boolf64x4::new([true, true, true, false]));
    }
}
