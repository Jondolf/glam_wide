# `bevy_math_extensions`

Extensions to [`bevy_math`] and [`glam`], the math libraries used by the [Bevy game engine].

[`bevy_math`]: https://docs.rs/bevy_math/latest/bevy_math/
[`glam`]: https://docs.rs/glam/latest/glam/
[Bevy game engine]: https://bevyengine.org/

## Warning ⚠️

This is experimental, very unfinished, and focused primarily on my own needs for [Avian] and related projects.

Feel free to copy-paste any types you may need in your own project, or even depend on the Git repository,
but don't expect stability or long-term support at this stage!

[Avian]: https://github.com/Jondolf/avian

## Planned Features

Some planned features include:

- Double-precision versions of math types
  - [x] 2D rotations: `DRot2`
- Rectangular matrices
  - [ ] 2x3 matrices: `Mat23`, `DMat23`
- Symmetric matrices
  - [ ] Symmetric 3x3 matrices: `SymmetricMat3`, `DSymmetricMat3`
  - [ ] Symmetric 4x4 matrices: `SymmetricMat4`, `DSymmetricMat4`
  - [ ] Symmetric 5x5 matrices: `SymmetricMat5`, `DSymmetricMat5`
  - [ ] Symmetric 6x6 matrices: `SymmetricMat6`, `DSymmetricMat6`
- Eigen decompositions of symmetric matrices (I have these locally already)
  - [ ] 2D: `SymmetricEigen2`
  - [ ] 3D: `SymmetricEigen3`
- Wide SIMD types
  - [x] 2D vectors: `Vec2x4`, `Vec2x8`, `DVec2x2`, `DVec2x4`
  - [x] 3D vectors: `Vec3x4`, `Vec3x8`, `DVec3x2`, `DVec3x4`
  - [x] 2D rotations: `Rot2x4`, `Rot2x8`, `DRot2x2`, `DRot2x4`
  - [x] 3D rotations: `Quatx4`, `Quatx8`, `DQuatx2`, `DQuatx4`
  - [x] 2x2 matrices: `Mat2x4`, `Mat2x8`, `DMat2x2`, `DMat2x4`
  - [x] 3x3 matrices: `Mat3x4`, `Mat3x8`, `DMat3x2`, `DMat3x4`
  - [ ] 2x3 matrices: `Mat23x4`, `Mat23x8`, `DMat23x2`, `DMat23x4`
  - [ ] Symmetric 3x3 matrices: `SymmetricMat3x4`, `SymmetricMat3x8`, `DSymmetricMat3x2`, `DSymmetricMat3x4`
  - [ ] Symmetric 4x4 matrices: `SymmetricMat4x4`, `SymmetricMat4x8`, `DSymmetricMat4x2`, `DSymmetricMat4x4`
  - [ ] Symmetric 5x5 matrices: `SymmetricMat5x4`, `SymmetricMat5x8`, `DSymmetricMat5x2`, `DSymmetricMat5x4`
  - [ ] Symmetric 6x6 matrices: `SymmetricMat&x4`, `SymmetricMat6x8`, `DSymmetricMat6x2`, `DSymmetricMat6x4`

These features are primarily driven by my needs for physics simulation.
More features may be added over time as requirements change.

## Out-Of-Scope Features

The following are *not* planned for now:

- Full feature parity with `bevy_math` or `glam` types
- Wide integer types

However, if there is interest, PRs are welcome!

## License

`bevy_math_extensions` is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
