# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.32] - 2025-07-19

### Removed

- **(rust)** Remove unused db for pyxis cli by @Glatzel

## [0.0.31] - 2025-07-19

### Changed

- **(rust)** Update clerk by @Glatzel

### Fixed

- Release ci by @Glatzel
- Install cpp headers by @Glatzel

## [0.0.30] - 2025-07-18

### Removed

- Remove unused clone by @Glatzel

## [0.0.29] - 2025-07-18

### Changed

- **(rust)** Arc proj by @Glatzel

## [0.0.28] - 2025-07-18

### Fixed

- Rattler build by @Glatzel

## [0.0.27] - 2025-07-13

### Fixed

- Clippy by @Glatzel

### Removed

- Remove cuda test report by @Glatzel

## [0.0.26] - 2025-06-11

### Documentation

- **(rust)** Proj doc by @Glatzel

### Fixed

- Dangling pointers by @Glatzel

## [0.0.25] - 2025-06-04

### Added

- Add cuda runner by @Glatzel

### Changed

- **(rust)** Proj iso19111 advanced functions by @Glatzel
- **(rust)** Proj network by @Glatzel

### Refactor

- **(rust)** Change iso19111 structure by @Glatzel

### Testing

- **(rust)** Add iso19111 advanced functions test by @Glatzel
- **(rust)** Add iso19111 advanced functions test by @Glatzel

## [0.0.24] - 2025-05-31

### Changed

- **(rust)** Impl some iso19111 functions
- **(rust)** Proj pj_obj_list by @Glatzel
- **(rust)** Use envoy in proj by @Glatzel
- **(rust)** Iso19111 by @Glatzel

### Refactor

- **(rust)** Simplify string enum
- Simplify some string enum
- Ffi string by @Glatzel
- **(rust)** Simplify *mut i8 to string by @Glatzel

### Testing

- **(rust)** Add some test for iso19111
- **(rust)** Proj iso19111 pj_obj_list by @Glatzel
- **(rust)** Proj 4d by @Glatzel
- **(rust)** Iso19111 by @Glatzel

## [0.0.23] - 2025-05-29

### Documentation

- **(rust)** Add some proj doc
- **(rust)** Add iso19111 type doc

### Refactor

- Proj from_raw
- Rename some objects

## [0.0.22] - 2025-05-28

### Changed

- **(rust)** Proj create_cs
- **(rust)** Proj 2d cs
- **(rust)** Proj create_geographic_crs
- **(rust)** Proj crs
- **(rust)** Proj database
- Simplify loglevel

### Documentation

- Move doc to module file

### Testing

- **(rust)** Add some test

## [0.0.21] - 2025-05-16

### Added

- Add guess_wkt_dialect

### Changed

- **(rust)** Proj iso19111 pj
- **(rust)** Proj option

### Fixed

- **(rust)** Clerk null macro
- **(rust)** Proj option macro

### Refactor

- **(python,rust,rust-cuda)** Datum compensate
- **(rust)** Pj

## [0.0.20] - 2025-05-12

### Added

- **(rust)** Add proj iso19111 types
- **(rust)** Add proj native log_level and log_func
- **(cpp,rust)** Support macos arm64
- Add fmt and clippy to pre-commit
- Add pre-commit to ignore
- Add csharp hook

### Changed

- **(rust)** Proj get_last_used_operation
- **(rust)** Inital proj network

### Fixed

- **(rust)** Proj-sys build

### Removed

- Remove whl asset
- Delete pre-commit/pixi.toml

### Testing

- **(rust)** Proj init info

## [0.0.19] - 2025-05-06

### Added

- Add some proj features
- Add proj logging

### Changed

- Replace some functions by proj
- Proj convert array
- Simplify proj trait
- Omit unsafe PJ_COORD struct
- Centralize proj creation
- Optimize IPyCoord trait

### Fixed

- Proj bindings c char type

### Removed

- Remove pkg config lite in python ci
- Remove some useless crates

### Testing

- Proj extension conversion
- Proj function distance
- Proj function coordinate_transformation
- Proj transformation_setup
- Proj grid info

## [0.0.18] - 2025-05-02

### Added

- Initial renovate bot
- Initial builtin proj

### Removed

- Remove cli dynamic version
- Remove some useless packages

## [0.0.17] - 2025-03-26

### Added

- Add CODEOWNERS
- Add some filter
- Add permission

### Changed

- Manually set block size in rust

## [0.0.16] - 2025-03-18

### Added

- Add new clone parameter to python
- Add python cuda crypto
- Add kernel setting to python cuda
- Add python bench

### Changed

- Simplify build scripts
- Use ref in some functions

### Documentation

- Add dependency graph

### Fixed

- Crypto cuda problem
- Vscode workspace rust config

## [0.0.15] - 2025-03-15

### Added

- Initial csharp
- Add nupkg build

### Fixed

- Build scripts

### Removed

- Remove nupkg

## [0.0.14] - 2025-03-14

### Added

- Initial python cuda
- Initial opencl

### Changed

- Cpp generics
- Generics rust cuda

## [0.0.13] - 2025-03-13

### Added

- Add check condition
- Add update vcpkg baseline

### Documentation

- Add feature doc

### Fixed

- Build problem

## [0.0.12] - 2025-03-11

### Added

- Add ptx struct
- Add fn to set `size_limit` and `count_limit`
- Add conda packages

### Changed

- Use option to control cuda module load
- Singleton cuda context

### Documentation

- Add some comments and log

### Fixed

- Build pyxis-py conda package

## [0.0.11] - 2025-03-08

### Added

- Initial pyxis-cuda
- Add some check and log to datum_compense cuda
- Add util fn to calculate block and grid size
- Add gcj bd crypto cuda
- Add .pc file in cmake project

### Changed

- Complete crypto cuda
- New cuda context manager

### Performance

- Use newton method to accelerate crypto exact

### Refactor

- Reconstruct folers

### Removed

- Remove python build need

## [0.0.10] - 2025-03-05

### Fixed

- Doc test
- Use wgs84 ellipsoid to calculate distance

### Performance

- Improve const ellipsoid performance
- Optimize datum_compense and crypto_exact

### Refactor

- Initial num_trait for algorithm generics
- Simplify space python api
- Generics migrate
- Use assign trait

## [0.0.9] - 2025-03-01

### Added

- Add python crypto exact mode

### Refactor

- Simplify python crypto interface

## [0.0.8] - 2025-02-28

### Added

- Add vcpkg baseline and specify proj version
- Add linux build

### Changed

- Embed proj.db into cli
- New build type

## [0.0.7] - 2025-02-26

### Added

- Initial bench

### Changed

- Complete bench for crypto
- Modify crypto exact recursion exit

### Performance

- Raise crypto precision
- Revert to original crypto algorithm
- Improve crypto exact precision

### Refactor

- Use closures in crypto exact

### Testing

- Optimize crypto bench
- Revert to use threshold to bench crypto exact
- Add new thershold to crypto bench
- Modify some test precision assert
- Optimize crypto exact bench input value

## [0.0.6] - 2025-02-20

### Added

- Add cargo-clean script
- Add example
- Add ellipsoid to cli record
- Add cli example
- Add log feature to recursion algorithm
- Add some cli log message

### Changed

- Implement coordinate syetem migrate 2d

### Refactor

- Rename file, parameters

### Testing

- Add cli tests
- Add cli scale test
- Add algorithm rotate 2d test
- Add cli rotate test

## [0.0.5] - 2025-02-15

### Documentation

- Replace parameters with arguments in rust doc

### Performance

- Raise precision of `gcj02_to_wgs84`
- Raise precision of `bd09_to_gcj02`

### Refactor

- Migrate log to independent repo

### Removed

- Delete slug parameter in codecov

## [0.0.4] - 2025-02-12

### Changed

- Prettify cli output

### Documentation

- Add python doc
- Fix some python docs

## [0.0.3] - 2025-02-11

### Added

- Add cli transform subcommands
- Add `from_semi_axis` to ellipsoid

### Fixed

- Cli args parser log
- RotateUnit from str
- `from_semi_axis` doctest
- Cargo test never fail

### Performance

- Optimize test and coverage

### Refactor

- Use `Ellipsoid` in some functions

### Testing

- Optimize assert float

## [0.0.2] - 2025-02-07

### Added

- Add rust cov
- Add doctest
- Add react to comment in changelog ci

### Documentation

- Complete doc for crypto

### Fixed

- Python function return type for scalar array

### Removed

- Remove changelog ci

## [0.0.1] - 2025-02-05

[0.0.32]: https://github.com/Glatzel/pyxis/compare/v0.0.31..v0.0.32
[0.0.31]: https://github.com/Glatzel/pyxis/compare/v0.0.30..v0.0.31
[0.0.30]: https://github.com/Glatzel/pyxis/compare/v0.0.29..v0.0.30
[0.0.29]: https://github.com/Glatzel/pyxis/compare/v0.0.28..v0.0.29
[0.0.28]: https://github.com/Glatzel/pyxis/compare/v0.0.27..v0.0.28
[0.0.27]: https://github.com/Glatzel/pyxis/compare/v0.0.26..v0.0.27
[0.0.26]: https://github.com/Glatzel/pyxis/compare/v0.0.25..v0.0.26
[0.0.25]: https://github.com/Glatzel/pyxis/compare/v0.0.24..v0.0.25
[0.0.24]: https://github.com/Glatzel/pyxis/compare/v0.0.23..v0.0.24
[0.0.23]: https://github.com/Glatzel/pyxis/compare/v0.0.22..v0.0.23
[0.0.22]: https://github.com/Glatzel/pyxis/compare/v0.0.21..v0.0.22
[0.0.21]: https://github.com/Glatzel/pyxis/compare/v0.0.20..v0.0.21
[0.0.20]: https://github.com/Glatzel/pyxis/compare/v0.0.19..v0.0.20
[0.0.19]: https://github.com/Glatzel/pyxis/compare/v0.0.18..v0.0.19
[0.0.18]: https://github.com/Glatzel/pyxis/compare/v0.0.17..v0.0.18
[0.0.17]: https://github.com/Glatzel/pyxis/compare/v0.0.16..v0.0.17
[0.0.16]: https://github.com/Glatzel/pyxis/compare/v0.0.15..v0.0.16
[0.0.15]: https://github.com/Glatzel/pyxis/compare/v0.0.14..v0.0.15
[0.0.14]: https://github.com/Glatzel/pyxis/compare/v0.0.13..v0.0.14
[0.0.13]: https://github.com/Glatzel/pyxis/compare/v0.0.12..v0.0.13
[0.0.12]: https://github.com/Glatzel/pyxis/compare/v0.0.11..v0.0.12
[0.0.11]: https://github.com/Glatzel/pyxis/compare/v0.0.10..v0.0.11
[0.0.10]: https://github.com/Glatzel/pyxis/compare/v0.0.9..v0.0.10
[0.0.9]: https://github.com/Glatzel/pyxis/compare/v0.0.8..v0.0.9
[0.0.8]: https://github.com/Glatzel/pyxis/compare/v0.0.7..v0.0.8
[0.0.7]: https://github.com/Glatzel/pyxis/compare/v0.0.6..v0.0.7
[0.0.6]: https://github.com/Glatzel/pyxis/compare/v0.0.5..v0.0.6
[0.0.5]: https://github.com/Glatzel/pyxis/compare/v0.0.4..v0.0.5
[0.0.4]: https://github.com/Glatzel/pyxis/compare/v0.0.3..v0.0.4
[0.0.3]: https://github.com/Glatzel/pyxis/compare/v0.0.2..v0.0.3
[0.0.2]: https://github.com/Glatzel/pyxis/compare/v0.0.1..v0.0.2

<!-- generated by git-cliff -->
