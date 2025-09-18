// See https://github.com/bitshifter/glam-rs/blob/c1ff830175bcd064f35a5acc60105ec8a278a874/src/swizzles/vec3_impl.rs

use glam::Vec3Swizzles;

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

macro_rules! wide_vec3_swizzles {
    ($(($v2t:ident, $n:ident, $v4t:ident)),+) => {
        $(
        impl Vec3Swizzles for $n {
            type Vec2 = $v2t;

            type Vec4 = $v4t;

            #[inline]
            fn xx(self) -> $v2t {
                $v2t {
                    x: self.x,
                    y: self.x,
                }
            }

            #[inline]
            fn xy(self) -> $v2t {
                $v2t {
                    x: self.x,
                    y: self.y,
                }
            }

            #[inline]
            fn with_xy(self, rhs: $v2t) -> Self {
                Self::new(rhs.x, rhs.y, self.z)
            }

            #[inline]
            fn xz(self) -> $v2t {
                $v2t {
                    x: self.x,
                    y: self.z,
                }
            }

            #[inline]
            fn with_xz(self, rhs: $v2t) -> Self {
                Self::new(rhs.x, self.y, rhs.y)
            }

            #[inline]
            fn yx(self) -> $v2t {
                $v2t {
                    x: self.y,
                    y: self.x,
                }
            }

            #[inline]
            fn with_yx(self, rhs: $v2t) -> Self {
                Self::new(rhs.y, rhs.x, self.z)
            }

            #[inline]
            fn yy(self) -> $v2t {
                $v2t {
                    x: self.y,
                    y: self.y,
                }
            }

            #[inline]
            fn yz(self) -> $v2t {
                $v2t {
                    x: self.y,
                    y: self.z,
                }
            }

            #[inline]
            fn with_yz(self, rhs: $v2t) -> Self {
                Self::new(self.x, rhs.x, rhs.y)
            }

            #[inline]
            fn zx(self) -> $v2t {
                $v2t {
                    x: self.z,
                    y: self.x,
                }
            }

            #[inline]
            fn with_zx(self, rhs: $v2t) -> Self {
                Self::new(rhs.y, self.y, rhs.x)
            }

            #[inline]
            fn zy(self) -> $v2t {
                $v2t {
                    x: self.z,
                    y: self.y,
                }
            }

            #[inline]
            fn with_zy(self, rhs: $v2t) -> Self {
                Self::new(self.x, rhs.y, rhs.x)
            }

            #[inline]
            fn zz(self) -> $v2t {
                $v2t {
                    x: self.z,
                    y: self.z,
                }
            }

            #[inline]
            fn xxx(self) -> Self {
                Self::new(self.x, self.x, self.x)
            }

            #[inline]
            fn xxy(self) -> Self {
                Self::new(self.x, self.x, self.y)
            }

            #[inline]
            fn xxz(self) -> Self {
                Self::new(self.x, self.x, self.z)
            }

            #[inline]
            fn xyx(self) -> Self {
                Self::new(self.x, self.y, self.x)
            }

            #[inline]
            fn xyy(self) -> Self {
                Self::new(self.x, self.y, self.y)
            }

            #[inline]
            fn xzx(self) -> Self {
                Self::new(self.x, self.z, self.x)
            }

            #[inline]
            fn xzy(self) -> Self {
                Self::new(self.x, self.z, self.y)
            }

            #[inline]
            fn xzz(self) -> Self {
                Self::new(self.x, self.z, self.z)
            }

            #[inline]
            fn yxx(self) -> Self {
                Self::new(self.y, self.x, self.x)
            }

            #[inline]
            fn yxy(self) -> Self {
                Self::new(self.y, self.x, self.y)
            }

            #[inline]
            fn yxz(self) -> Self {
                Self::new(self.y, self.x, self.z)
            }

            #[inline]
            fn yyx(self) -> Self {
                Self::new(self.y, self.y, self.x)
            }

            #[inline]
            fn yyy(self) -> Self {
                Self::new(self.y, self.y, self.y)
            }

            #[inline]
            fn yyz(self) -> Self {
                Self::new(self.y, self.y, self.z)
            }

            #[inline]
            fn yzx(self) -> Self {
                Self::new(self.y, self.z, self.x)
            }

            #[inline]
            fn yzy(self) -> Self {
                Self::new(self.y, self.z, self.y)
            }

            #[inline]
            fn yzz(self) -> Self {
                Self::new(self.y, self.z, self.z)
            }

            #[inline]
            fn zxx(self) -> Self {
                Self::new(self.z, self.x, self.x)
            }

            #[inline]
            fn zxy(self) -> Self {
                Self::new(self.z, self.x, self.y)
            }

            #[inline]
            fn zxz(self) -> Self {
                Self::new(self.z, self.x, self.z)
            }

            #[inline]
            fn zyx(self) -> Self {
                Self::new(self.z, self.y, self.x)
            }

            #[inline]
            fn zyy(self) -> Self {
                Self::new(self.z, self.y, self.y)
            }

            #[inline]
            fn zyz(self) -> Self {
                Self::new(self.z, self.y, self.z)
            }

            #[inline]
            fn zzx(self) -> Self {
                Self::new(self.z, self.z, self.x)
            }

            #[inline]
            fn zzy(self) -> Self {
                Self::new(self.z, self.z, self.y)
            }

            #[inline]
            fn zzz(self) -> Self {
                Self::new(self.z, self.z, self.z)
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
            fn xxxz(self) -> $v4t {
                $v4t::new(self.x, self.x, self.x, self.z)
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
            fn xxyz(self) -> $v4t {
                $v4t::new(self.x, self.x, self.y, self.z)
            }

            #[inline]
            fn xxzx(self) -> $v4t {
                $v4t::new(self.x, self.x, self.z, self.x)
            }

            #[inline]
            fn xxzy(self) -> $v4t {
                $v4t::new(self.x, self.x, self.z, self.y)
            }

            #[inline]
            fn xxzz(self) -> $v4t {
                $v4t::new(self.x, self.x, self.z, self.z)
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
            fn xyxz(self) -> $v4t {
                $v4t::new(self.x, self.y, self.x, self.z)
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
            fn xyyz(self) -> $v4t {
                $v4t::new(self.x, self.y, self.y, self.z)
            }

            #[inline]
            fn xyzx(self) -> $v4t {
                $v4t::new(self.x, self.y, self.z, self.x)
            }

            #[inline]
            fn xyzy(self) -> $v4t {
                $v4t::new(self.x, self.y, self.z, self.y)
            }

            #[inline]
            fn xyzz(self) -> $v4t {
                $v4t::new(self.x, self.y, self.z, self.z)
            }

            #[inline]
            fn xzxx(self) -> $v4t {
                $v4t::new(self.x, self.z, self.x, self.x)
            }

            #[inline]
            fn xzxy(self) -> $v4t {
                $v4t::new(self.x, self.z, self.x, self.y)
            }

            #[inline]
            fn xzxz(self) -> $v4t {
                $v4t::new(self.x, self.z, self.x, self.z)
            }

            #[inline]
            fn xzyx(self) -> $v4t {
                $v4t::new(self.x, self.z, self.y, self.x)
            }

            #[inline]
            fn xzyy(self) -> $v4t {
                $v4t::new(self.x, self.z, self.y, self.y)
            }

            #[inline]
            fn xzyz(self) -> $v4t {
                $v4t::new(self.x, self.z, self.y, self.z)
            }

            #[inline]
            fn xzzx(self) -> $v4t {
                $v4t::new(self.x, self.z, self.z, self.x)
            }

            #[inline]
            fn xzzy(self) -> $v4t {
                $v4t::new(self.x, self.z, self.z, self.y)
            }

            #[inline]
            fn xzzz(self) -> $v4t {
                $v4t::new(self.x, self.z, self.z, self.z)
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
            fn yxxz(self) -> $v4t {
                $v4t::new(self.y, self.x, self.x, self.z)
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
            fn yxyz(self) -> $v4t {
                $v4t::new(self.y, self.x, self.y, self.z)
            }

            #[inline]
            fn yxzx(self) -> $v4t {
                $v4t::new(self.y, self.x, self.z, self.x)
            }

            #[inline]
            fn yxzy(self) -> $v4t {
                $v4t::new(self.y, self.x, self.z, self.y)
            }

            #[inline]
            fn yxzz(self) -> $v4t {
                $v4t::new(self.y, self.x, self.z, self.z)
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
            fn yyxz(self) -> $v4t {
                $v4t::new(self.y, self.y, self.x, self.z)
            }

            #[inline]
            fn yyyx(self) -> $v4t {
                $v4t::new(self.y, self.y, self.y, self.x)
            }

            #[inline]
            fn yyyy(self) -> $v4t {
                $v4t::new(self.y, self.y, self.y, self.y)
            }

            #[inline]
            fn yyyz(self) -> $v4t {
                $v4t::new(self.y, self.y, self.y, self.z)
            }

            #[inline]
            fn yyzx(self) -> $v4t {
                $v4t::new(self.y, self.y, self.z, self.x)
            }

            #[inline]
            fn yyzy(self) -> $v4t {
                $v4t::new(self.y, self.y, self.z, self.y)
            }

            #[inline]
            fn yyzz(self) -> $v4t {
                $v4t::new(self.y, self.y, self.z, self.z)
            }

            #[inline]
            fn yzxx(self) -> $v4t {
                $v4t::new(self.y, self.z, self.x, self.x)
            }

            #[inline]
            fn yzxy(self) -> $v4t {
                $v4t::new(self.y, self.z, self.x, self.y)
            }

            #[inline]
            fn yzxz(self) -> $v4t {
                $v4t::new(self.y, self.z, self.x, self.z)
            }

            #[inline]
            fn yzyx(self) -> $v4t {
                $v4t::new(self.y, self.z, self.y, self.x)
            }

            #[inline]
            fn yzyy(self) -> $v4t {
                $v4t::new(self.y, self.z, self.y, self.y)
            }

            #[inline]
            fn yzyz(self) -> $v4t {
                $v4t::new(self.y, self.z, self.y, self.z)
            }

            #[inline]
            fn yzzx(self) -> $v4t {
                $v4t::new(self.y, self.z, self.z, self.x)
            }

            #[inline]
            fn yzzy(self) -> $v4t {
                $v4t::new(self.y, self.z, self.z, self.y)
            }

            #[inline]
            fn yzzz(self) -> $v4t {
                $v4t::new(self.y, self.z, self.z, self.z)
            }

            #[inline]
            fn zxxx(self) -> $v4t {
                $v4t::new(self.z, self.x, self.x, self.x)
            }

            #[inline]
            fn zxxy(self) -> $v4t {
                $v4t::new(self.z, self.x, self.x, self.y)
            }

            #[inline]
            fn zxxz(self) -> $v4t {
                $v4t::new(self.z, self.x, self.x, self.z)
            }

            #[inline]
            fn zxyx(self) -> $v4t {
                $v4t::new(self.z, self.x, self.y, self.x)
            }

            #[inline]
            fn zxyy(self) -> $v4t {
                $v4t::new(self.z, self.x, self.y, self.y)
            }

            #[inline]
            fn zxyz(self) -> $v4t {
                $v4t::new(self.z, self.x, self.y, self.z)
            }

            #[inline]
            fn zxzx(self) -> $v4t {
                $v4t::new(self.z, self.x, self.z, self.x)
            }

            #[inline]
            fn zxzy(self) -> $v4t {
                $v4t::new(self.z, self.x, self.z, self.y)
            }

            #[inline]
            fn zxzz(self) -> $v4t {
                $v4t::new(self.z, self.x, self.z, self.z)
            }

            #[inline]
            fn zyxx(self) -> $v4t {
                $v4t::new(self.z, self.y, self.x, self.x)
            }

            #[inline]
            fn zyxy(self) -> $v4t {
                $v4t::new(self.z, self.y, self.x, self.y)
            }

            #[inline]
            fn zyxz(self) -> $v4t {
                $v4t::new(self.z, self.y, self.x, self.z)
            }

            #[inline]
            fn zyyx(self) -> $v4t {
                $v4t::new(self.z, self.y, self.y, self.x)
            }

            #[inline]
            fn zyyy(self) -> $v4t {
                $v4t::new(self.z, self.y, self.y, self.y)
            }

            #[inline]
            fn zyyz(self) -> $v4t {
                $v4t::new(self.z, self.y, self.y, self.z)
            }

            #[inline]
            fn zyzx(self) -> $v4t {
                $v4t::new(self.z, self.y, self.z, self.x)
            }

            #[inline]
            fn zyzy(self) -> $v4t {
                $v4t::new(self.z, self.y, self.z, self.y)
            }

            #[inline]
            fn zyzz(self) -> $v4t {
                $v4t::new(self.z, self.y, self.z, self.z)
            }

            #[inline]
            fn zzxx(self) -> $v4t {
                $v4t::new(self.z, self.z, self.x, self.x)
            }

            #[inline]
            fn zzxy(self) -> $v4t {
                $v4t::new(self.z, self.z, self.x, self.y)
            }

            #[inline]
            fn zzxz(self) -> $v4t {
                $v4t::new(self.z, self.z, self.x, self.z)
            }

            #[inline]
            fn zzyx(self) -> $v4t {
                $v4t::new(self.z, self.z, self.y, self.x)
            }

            #[inline]
            fn zzyy(self) -> $v4t {
                $v4t::new(self.z, self.z, self.y, self.y)
            }

            #[inline]
            fn zzyz(self) -> $v4t {
                $v4t::new(self.z, self.z, self.y, self.z)
            }

            #[inline]
            fn zzzx(self) -> $v4t {
                $v4t::new(self.z, self.z, self.z, self.x)
            }

            #[inline]
            fn zzzy(self) -> $v4t {
                $v4t::new(self.z, self.z, self.z, self.y)
            }

            #[inline]
            fn zzzz(self) -> $v4t {
                $v4t::new(self.z, self.z, self.z, self.z)
            }
        }
        )+
    };
}

#[cfg(feature = "f32")]
wide_vec3_swizzles!(
    (Vec2x4, Vec3x4, Vec4x4),
    (Vec2x8, Vec3x8, Vec4x8),
    (BVec2x4, BVec3x4, BVec4x4),
    (BVec2x8, BVec3x8, BVec4x8)
);

#[cfg(feature = "f64")]
wide_vec3_swizzles!(
    (DVec2x2, DVec3x2, DVec4x2),
    (DVec2x4, DVec3x4, DVec4x4),
    (BDVec2x2, BDVec3x2, BDVec4x2),
    (BDVec2x4, BDVec3x4, BDVec4x4)
);
