# `glam_wide`

Wide SIMD types for the [`glam`] ecosystem.

This includes types for [`glam`], [`glam_matrix_extras`], and [`bevy_math`].

[`glam`]: https://docs.rs/glam/latest/glam/
[`glam_matrix_extras`]: https://github.com/Jondolf/glam_matrix_extras
[`bevy_math`]: https://docs.rs/bevy_math/latest/bevy_math/

## What is SIMD?

SIMD stands for [_Single Instruction, Multiple Data_](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
In short, it allows you to perform the same operation on multiple pieces of data at the same time,
which can lead to significant performance improvements in numerical computations.

Broadly speaking, SIMD can be split into "horizontal" and "vertical" SIMD.
Glam itself is focused on horizontal SIMD, in that it does calculations on one
piece of data at a time, but still leverages SIMD instructions to speed up some of its
internal calculations, and provides types such as `Vec3A` that are aligned for SIMD usage.
This type of SIMD is almost entirely transparent to the user.

`glam_wide` on the other hand provides "wide" alternatives for Glam's types, focusing on vertical SIMD.
It operates in an [AoSoA (Array of Structures of Arrays)](https://en.wikipedia.org/wiki/AoS_and_SoA) fashion,
where each vector or matrix type contains multiple "lanes" of data, allowing you to perform operations
on multiple vectors or matrices at once. For example, a `Vec3x4` contains an `f32x4` for the `x`, `y`, and `z` axes,
and operations on it will operate on all four vectors at once.

Wide SIMD can be significantly more performant when used effectively, but it typically requires more effort
and can make algorithm design more challenging. As such, it is generally better to start with Glam's standard types,
and only use wide types for performance-critical code paths that are suitable for vectorization. Examples of where
wide SIMD can be beneficial include physics simulations, audio processing, and ray intersections.

## Features

The following features are provided by `glam_wide`, or are being actively worked on:

- Wide number types provided by `wide`
  - [x] 32-bit floats: `f32x4`, `f32x8`
  - [x] 64-bit floats: `f64x2`, `f64x4`
  - [x] 32-bit masks: `boolf32x4`, `boolf32x8`
  - [x] 64-bit masks: `boolf64x4`, `boolf64x8`
- Wide vectors
  - [x] 2D vectors: `Vec2x4`, `Vec2x8`, `DVec2x2`, `DVec2x4`
  - [x] 3D vectors: `Vec3x4`, `Vec3x8`, `DVec3x2`, `DVec3x4`
  - [x] 4D vectors: `Vec4x4`, `Vec4x8`, `DVec4x2`, `DVec4x4`
- Wide vector masks
  - [x] 2D vector masks: `BVec2x4`, `BVec2x8`, `BDVec2x2`, `BDVec2x4`
  - [x] 3D vector masks: `BVec3x4`, `BVec3x8`, `BDVec3x2`, `BDVec3x4`
  - [x] 4D vector masks: `BVec4x4`, `BVec4x8`, `BDVec4x2`, `BDVec4x4`
- Wide square matrices
  - [x] 2x2 matrices: `Mat2x4`, `Mat2x8`, `DMat2x2`, `DMat2x4`
  - [x] 3x3 matrices: `Mat3x4`, `Mat3x8`, `DMat3x2`, `DMat3x4`
  - [ ] 4x4 matrices: `Mat4x4`, `Mat4x8`, `DMat4x2`, `DMat4x4`
- Wide symmetric matrices
  - [x] Symmetric 2x2 matrices: `SymmetricMat2x4`, `SymmetricMat2x8`, `SymmetricDMat2x2`, `SymmetricDMat2x4`
  - [x] Symmetric 3x3 matrices: `SymmetricMat3x4`, `SymmetricMat3x8`, `SymmetricDMat3x2`, `SymmetricDMat3x4`
  - [ ] Symmetric 4x4 matrices: `SymmetricMat4x4`, `SymmetricMat4x8`, `SymmetricDMat4x2`, `SymmetricDMat4x4`
  - [ ] Symmetric 5x5 matrices: `SymmetricMat5x4`, `SymmetricMat5x8`, `SymmetricDMat5x2`, `SymmetricDMat5x4`
  - [x] Symmetric 6x6 matrices: `SymmetricMat6x4`, `SymmetricMat6x8`, `SymmetricDMat6x2`, `SymmetricDMat6x4`
- Wide rectangular matrices
  - [x] 2x3 matrices: `Mat23x4`, `Mat23x8`, `DMat23x2`, `DMat23x4`
  - [x] 3x2 matrices: `Mat32x4`, `Mat32x8`, `DMat32x2`, `DMat32x4`
- Wide rotations
  - [x] 2D rotations: `Rot2x4`, `Rot2x8`, `DRot2x2`, `DRot2x4`
  - [x] 3D rotations: `Quatx4`, `Quatx8`, `DQuatx2`, `DQuatx4`

In its current state, `glam_wide` is primarily focused on providing floating-point SIMD vector and matrix types.
The following are not an active focus, but are open to contributions:

- Wide integer vectors
- Wide affine transformations
- Wide isometries (`bevy_math`)
- Full feature parity

## License

`glam_wide` is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
