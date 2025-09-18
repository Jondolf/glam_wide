// See https://github.com/bitshifter/glam-rs/blob/c1ff830175bcd064f35a5acc60105ec8a278a874/src/swizzles/vec2_impl.rs

use glam::Vec2Swizzles;

#[cfg(feature = "f64")]
use crate::{
    BDVec2x2, BDVec2x4, BDVec3x2, BDVec3x4, BDVec4x2, BDVec4x4, DVec2x2, DVec2x4, DVec3x2, DVec3x4,
    DVec4x2, DVec4x4,
};
#[cfg(feature = "f32")]
use crate::{
    BVec2x4, BVec2x8, BVec3x4, BVec3x8, BVec4x4, BVec4x8, Vec2x4, Vec2x8, Vec3x4, Vec3x8, Vec4x4,
    Vec4x8,
};

macro_rules! wide_vec2_swizzles {
    ($(($n:ident, $v3t:ident, $v4t:ident)),+) => {
        $(
        impl Vec2Swizzles for $n {
            type Vec3 = $v3t;

            type Vec4 = $v4t;

            #[inline]
            fn xx(self) -> Self {
                Self {
                    x: self.x,
                    y: self.x,
                }
            }

            #[inline]
            fn yx(self) -> Self {
                Self {
                    x: self.y,
                    y: self.x,
                }
            }

            #[inline]
            fn yy(self) -> Self {
                Self {
                    x: self.y,
                    y: self.y,
                }
            }

            #[inline]
            fn xxx(self) -> $v3t {
                $v3t::new(self.x, self.x, self.x)
            }

            #[inline]
            fn xxy(self) -> $v3t {
                $v3t::new(self.x, self.x, self.y)
            }

            #[inline]
            fn xyx(self) -> $v3t {
                $v3t::new(self.x, self.y, self.x)
            }

            #[inline]
            fn xyy(self) -> $v3t {
                $v3t::new(self.x, self.y, self.y)
            }

            #[inline]
            fn yxx(self) -> $v3t {
                $v3t::new(self.y, self.x, self.x)
            }

            #[inline]
            fn yxy(self) -> $v3t {
                $v3t::new(self.y, self.x, self.y)
            }

            #[inline]
            fn yyx(self) -> $v3t {
                $v3t::new(self.y, self.y, self.x)
            }

            #[inline]
            fn yyy(self) -> $v3t {
                $v3t::new(self.y, self.y, self.y)
            }

            #[inline]
            fn xxxx(self) -> $v4t {
                $v4t::new(self.x, self.x, self.x, self.x)
            }

            #[inline]
            fn xxxy(self) -> $v4t {
                $v4t::new(self.x, self.x, self.x, self.y)
            }

            #[inline]
            fn xxyx(self) -> $v4t {
                $v4t::new(self.x, self.x, self.y, self.x)
            }

            #[inline]
            fn xxyy(self) -> $v4t {
                $v4t::new(self.x, self.x, self.y, self.y)
            }

            #[inline]
            fn xyxx(self) -> $v4t {
                $v4t::new(self.x, self.y, self.x, self.x)
            }

            #[inline]
            fn xyxy(self) -> $v4t {
                $v4t::new(self.x, self.y, self.x, self.y)
            }

            #[inline]
            fn xyyx(self) -> $v4t {
                $v4t::new(self.x, self.y, self.y, self.x)
            }

            #[inline]
            fn xyyy(self) -> $v4t {
                $v4t::new(self.x, self.y, self.y, self.y)
            }

            #[inline]
            fn yxxx(self) -> $v4t {
                $v4t::new(self.y, self.x, self.x, self.x)
            }

            #[inline]
            fn yxxy(self) -> $v4t {
                $v4t::new(self.y, self.x, self.x, self.y)
            }

            #[inline]
            fn yxyx(self) -> $v4t {
                $v4t::new(self.y, self.x, self.y, self.x)
            }

            #[inline]
            fn yxyy(self) -> $v4t {
                $v4t::new(self.y, self.x, self.y, self.y)
            }

            #[inline]
            fn yyxx(self) -> $v4t {
                $v4t::new(self.y, self.y, self.x, self.x)
            }

            #[inline]
            fn yyxy(self) -> $v4t {
                $v4t::new(self.y, self.y, self.x, self.y)
            }

            #[inline]
            fn yyyx(self) -> $v4t {
                $v4t::new(self.y, self.y, self.y, self.x)
            }

            #[inline]
            fn yyyy(self) -> $v4t {
                $v4t::new(self.y, self.y, self.y, self.y)
            }
        }
        )+
    };
}

#[cfg(feature = "f32")]
wide_vec2_swizzles!(
    (Vec2x4, Vec3x4, Vec4x4),
    (Vec2x8, Vec3x8, Vec4x8),
    (BVec2x4, BVec3x4, BVec4x4),
    (BVec2x8, BVec3x8, BVec4x8)
);

#[cfg(feature = "f64")]
wide_vec2_swizzles!(
    (DVec2x2, DVec3x2, DVec4x2),
    (DVec2x4, DVec3x4, DVec4x4),
    (BDVec2x2, BDVec3x2, BDVec4x2),
    (BDVec2x4, BDVec3x4, BDVec4x4)
);
