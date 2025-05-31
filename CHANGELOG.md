# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.24] - 2025-05-31

### Changed

- **(rust)** Impl some iso19111 functions by @Glatzel in [#425](https://github.com/Glatzel/pyxis/pull/425)
- **(rust)** Proj pj_obj_list by @Glatzel in [#440](https://github.com/Glatzel/pyxis/pull/440)
- **(rust)** Use envoy in proj by @Glatzel in [#445](https://github.com/Glatzel/pyxis/pull/445)
- **(rust)** Iso19111 by @Glatzel in [#447](https://github.com/Glatzel/pyxis/pull/447)

### Refactor

- **(rust)** Simplify string enum by @Glatzel in [#436](https://github.com/Glatzel/pyxis/pull/436)
- Simplify some string enum by @Glatzel in [#437](https://github.com/Glatzel/pyxis/pull/437)
- Ffi string by @Glatzel in [#444](https://github.com/Glatzel/pyxis/pull/444)
- **(rust)** Simplify *mut i8 to string by @Glatzel in [#449](https://github.com/Glatzel/pyxis/pull/449)

### Testing

- **(rust)** Add some test for iso19111 by @Glatzel in [#435](https://github.com/Glatzel/pyxis/pull/435)
- **(rust)** Proj iso19111 pj_obj_list by @Glatzel in [#442](https://github.com/Glatzel/pyxis/pull/442)
- **(rust)** Proj 4d by @Glatzel in [#443](https://github.com/Glatzel/pyxis/pull/443)
- **(rust)** Iso19111 by @Glatzel in [#448](https://github.com/Glatzel/pyxis/pull/448)

## [0.0.23] - 2025-05-29

### Documentation

- **(rust)** Add some proj doc by @Glatzel in [#420](https://github.com/Glatzel/pyxis/pull/420)
- **(rust)** Add iso19111 type doc by @Glatzel in [#421](https://github.com/Glatzel/pyxis/pull/421)

### Refactor

- Proj from_raw by @Glatzel in [#417](https://github.com/Glatzel/pyxis/pull/417)
- Rename some objects by @Glatzel in [#422](https://github.com/Glatzel/pyxis/pull/422)

## [0.0.22] - 2025-05-28

### Changed

- **(rust)** Proj create_cs by @Glatzel in [#402](https://github.com/Glatzel/pyxis/pull/402)
- **(rust)** Proj 2d cs by @Glatzel in [#404](https://github.com/Glatzel/pyxis/pull/404)
- **(rust)** Proj create_geographic_crs by @Glatzel in [#406](https://github.com/Glatzel/pyxis/pull/406)
- **(rust)** Proj crs by @Glatzel in [#407](https://github.com/Glatzel/pyxis/pull/407)
- **(rust)** Proj database by @Glatzel in [#411](https://github.com/Glatzel/pyxis/pull/411)
- Simplify loglevel by @Glatzel in [#412](https://github.com/Glatzel/pyxis/pull/412)

### Documentation

- Move doc to module file by @Glatzel in [#405](https://github.com/Glatzel/pyxis/pull/405)

### Testing

- **(rust)** Add some test by @Glatzel in [#409](https://github.com/Glatzel/pyxis/pull/409)

## [0.0.21] - 2025-05-16

### Added

- Add guess_wkt_dialect by @Glatzel in [#388](https://github.com/Glatzel/pyxis/pull/388)

### Changed

- **(rust)** Proj iso19111 pj by @Glatzel in [#377](https://github.com/Glatzel/pyxis/pull/377)
- **(rust)** Proj option by @Glatzel in [#385](https://github.com/Glatzel/pyxis/pull/385)

### Fixed

- **(rust)** Clerk null macro by @Glatzel in [#371](https://github.com/Glatzel/pyxis/pull/371)
- **(rust)** Proj option macro by @Glatzel in [#391](https://github.com/Glatzel/pyxis/pull/391)

### Refactor

- **(python,rust,rust-cuda)** Datum compensate by @Glatzel in [#372](https://github.com/Glatzel/pyxis/pull/372)
- **(rust)** Pj by @Glatzel in [#380](https://github.com/Glatzel/pyxis/pull/380)

## [0.0.20] - 2025-05-12

### Added

- **(rust)** Add proj iso19111 types by @Glatzel in [#319](https://github.com/Glatzel/pyxis/pull/319)
- **(rust)** Add proj native log_level and log_func by @Glatzel in [#326](https://github.com/Glatzel/pyxis/pull/326)
- **(cpp,rust)** Support macos arm64 by @Glatzel in [#327](https://github.com/Glatzel/pyxis/pull/327)
- Add fmt and clippy to pre-commit by @Glatzel in [#337](https://github.com/Glatzel/pyxis/pull/337)
- Add pre-commit to ignore by @Glatzel in [#343](https://github.com/Glatzel/pyxis/pull/343)
- Add csharp hook by @Glatzel in [#344](https://github.com/Glatzel/pyxis/pull/344)

### Changed

- **(rust)** Proj get_last_used_operation by @Glatzel in [#320](https://github.com/Glatzel/pyxis/pull/320)
- **(rust)** Inital proj network by @Glatzel in [#364](https://github.com/Glatzel/pyxis/pull/364)

### Fixed

- **(rust)** Proj-sys build by @Glatzel in [#324](https://github.com/Glatzel/pyxis/pull/324)

### Removed

- Remove whl asset by @Glatzel in [#318](https://github.com/Glatzel/pyxis/pull/318)
- Delete pre-commit/pixi.toml by @Glatzel in [#342](https://github.com/Glatzel/pyxis/pull/342)

### Testing

- **(rust)** Proj init info by @Glatzel in [#317](https://github.com/Glatzel/pyxis/pull/317)

## [0.0.19] - 2025-05-06

### Added

- Add some proj features by @Glatzel in [#305](https://github.com/Glatzel/pyxis/pull/305)
- Add proj logging by @Glatzel in [#306](https://github.com/Glatzel/pyxis/pull/306)

### Changed

- Replace some functions by proj by @Glatzel in [#290](https://github.com/Glatzel/pyxis/pull/290)
- Proj convert array by @Glatzel in [#291](https://github.com/Glatzel/pyxis/pull/291)
- Simplify proj trait by @Glatzel in [#296](https://github.com/Glatzel/pyxis/pull/296)
- Omit unsafe PJ_COORD struct by @Glatzel in [#299](https://github.com/Glatzel/pyxis/pull/299)
- Centralize proj creation by @Glatzel in [#312](https://github.com/Glatzel/pyxis/pull/312)
- Optimize IPyCoord trait by @Glatzel in [#314](https://github.com/Glatzel/pyxis/pull/314)

### Fixed

- Proj bindings c char type by @Glatzel in [#307](https://github.com/Glatzel/pyxis/pull/307)

### Removed

- Remove pkg config lite in python ci by @Glatzel in [#292](https://github.com/Glatzel/pyxis/pull/292)
- Remove some useless crates by @Glatzel in [#304](https://github.com/Glatzel/pyxis/pull/304)

### Testing

- Proj extension conversion by @Glatzel in [#293](https://github.com/Glatzel/pyxis/pull/293)
- Proj function distance by @Glatzel in [#295](https://github.com/Glatzel/pyxis/pull/295)
- Proj function coordinate_transformation by @Glatzel in [#298](https://github.com/Glatzel/pyxis/pull/298)
- Proj transformation_setup by @Glatzel in [#308](https://github.com/Glatzel/pyxis/pull/308)
- Proj grid info by @Glatzel in [#310](https://github.com/Glatzel/pyxis/pull/310)

## [0.0.18] - 2025-05-02

### Added

- Initial renovate bot by @Glatzel in [#264](https://github.com/Glatzel/pyxis/pull/264)
- Initial builtin proj by @Glatzel in [#282](https://github.com/Glatzel/pyxis/pull/282)

### Removed

- Remove cli dynamic version by @Glatzel in [#258](https://github.com/Glatzel/pyxis/pull/258)
- Remove some useless packages by @Glatzel in [#269](https://github.com/Glatzel/pyxis/pull/269)

## [0.0.17] - 2025-03-26

### Added

- Add CODEOWNERS by @Glatzel in [#246](https://github.com/Glatzel/pyxis/pull/246)
- Add some filter by @Glatzel in [#249](https://github.com/Glatzel/pyxis/pull/249)
- Add permission by @Glatzel in [#251](https://github.com/Glatzel/pyxis/pull/251)

### Changed

- Manually set block size in rust by @Glatzel in [#245](https://github.com/Glatzel/pyxis/pull/245)

## [0.0.16] - 2025-03-18

### Added

- Add new clone parameter to python by @Glatzel in [#229](https://github.com/Glatzel/pyxis/pull/229)
- Add python cuda crypto by @Glatzel in [#230](https://github.com/Glatzel/pyxis/pull/230)
- Add kernel setting to python cuda by @Glatzel in [#231](https://github.com/Glatzel/pyxis/pull/231)
- Add python bench by @Glatzel in [#236](https://github.com/Glatzel/pyxis/pull/236)

### Changed

- Simplify build scripts by @Glatzel in [#234](https://github.com/Glatzel/pyxis/pull/234)
- Use ref in some functions by @Glatzel in [#237](https://github.com/Glatzel/pyxis/pull/237)

### Documentation

- Add dependency graph by @Glatzel in [#235](https://github.com/Glatzel/pyxis/pull/235)

### Fixed

- Crypto cuda problem by @Glatzel in [#233](https://github.com/Glatzel/pyxis/pull/233)
- Vscode workspace rust config by @Glatzel in [#239](https://github.com/Glatzel/pyxis/pull/239)

## [0.0.15] - 2025-03-15

### Added

- Initial csharp by @Glatzel in [#216](https://github.com/Glatzel/pyxis/pull/216)
- Add nupkg build by @Glatzel in [#222](https://github.com/Glatzel/pyxis/pull/222)

### Fixed

- Build scripts by @Glatzel in [#221](https://github.com/Glatzel/pyxis/pull/221)

### Removed

- Remove nupkg by @Glatzel in [#227](https://github.com/Glatzel/pyxis/pull/227)

## [0.0.14] - 2025-03-14

### Added

- Initial python cuda by @Glatzel in [#206](https://github.com/Glatzel/pyxis/pull/206)
- Initial opencl by @Glatzel in [#209](https://github.com/Glatzel/pyxis/pull/209)

### Changed

- Cpp generics by @Glatzel in [#204](https://github.com/Glatzel/pyxis/pull/204)
- Generics rust cuda by @Glatzel in [#205](https://github.com/Glatzel/pyxis/pull/205)

## [0.0.13] - 2025-03-13

### Added

- Add check condition by @Glatzel in [#193](https://github.com/Glatzel/pyxis/pull/193)
- Add update vcpkg baseline by @Glatzel in [#194](https://github.com/Glatzel/pyxis/pull/194)

### Documentation

- Add feature doc by @Glatzel in [#200](https://github.com/Glatzel/pyxis/pull/200)

### Fixed

- Build problem by @Glatzel in [#201](https://github.com/Glatzel/pyxis/pull/201)

## [0.0.12] - 2025-03-11

### Added

- Add ptx struct by @Glatzel in [#170](https://github.com/Glatzel/pyxis/pull/170)
- Add fn to set `size_limit` and `count_limit` by @Glatzel in [#173](https://github.com/Glatzel/pyxis/pull/173)
- Add conda packages by @Glatzel in [#184](https://github.com/Glatzel/pyxis/pull/184)

### Changed

- Use option to control cuda module load by @Glatzel in [#171](https://github.com/Glatzel/pyxis/pull/171)
- Singleton cuda context by @Glatzel in [#172](https://github.com/Glatzel/pyxis/pull/172)

### Documentation

- Add some comments and log by @Glatzel in [#177](https://github.com/Glatzel/pyxis/pull/177)

### Fixed

- Build pyxis-py conda package by @Glatzel in [#186](https://github.com/Glatzel/pyxis/pull/186)

## [0.0.11] - 2025-03-08

### Added

- Initial pyxis-cuda by @Glatzel in [#150](https://github.com/Glatzel/pyxis/pull/150)
- Add some check and log to datum_compense cuda by @Glatzel in [#151](https://github.com/Glatzel/pyxis/pull/151)
- Add util fn to calculate block and grid size by @Glatzel in [#152](https://github.com/Glatzel/pyxis/pull/152)
- Add gcj bd crypto cuda by @Glatzel in [#153](https://github.com/Glatzel/pyxis/pull/153)
- Add .pc file in cmake project by @Glatzel in [#167](https://github.com/Glatzel/pyxis/pull/167)

### Changed

- Complete crypto cuda by @Glatzel in [#155](https://github.com/Glatzel/pyxis/pull/155)
- New cuda context manager by @Glatzel in [#165](https://github.com/Glatzel/pyxis/pull/165)

### Performance

- Use newton method to accelerate crypto exact by @Glatzel in [#154](https://github.com/Glatzel/pyxis/pull/154)

### Refactor

- Reconstruct folers by @Glatzel in [#157](https://github.com/Glatzel/pyxis/pull/157)

### Removed

- Remove python build need by @Glatzel in [#168](https://github.com/Glatzel/pyxis/pull/168)

## [0.0.10] - 2025-03-05

### Fixed

- Doc test by @Glatzel in [#132](https://github.com/Glatzel/pyxis/pull/132)
- Use wgs84 ellipsoid to calculate distance by @Glatzel in [#134](https://github.com/Glatzel/pyxis/pull/134)

### Performance

- Improve const ellipsoid performance by @Glatzel in [#129](https://github.com/Glatzel/pyxis/pull/129)
- Optimize datum_compense and crypto_exact by @Glatzel in [#142](https://github.com/Glatzel/pyxis/pull/142)

### Refactor

- Initial num_trait for algorithm generics by @Glatzel in [#128](https://github.com/Glatzel/pyxis/pull/128)
- Simplify space python api by @Glatzel in [#135](https://github.com/Glatzel/pyxis/pull/135)
- Generics migrate by @Glatzel in [#136](https://github.com/Glatzel/pyxis/pull/136)
- Use assign trait by @Glatzel in [#137](https://github.com/Glatzel/pyxis/pull/137)

## [0.0.9] - 2025-03-01

### Added

- Add python crypto exact mode by @Glatzel in [#124](https://github.com/Glatzel/pyxis/pull/124)

### Refactor

- Simplify python crypto interface by @Glatzel in [#123](https://github.com/Glatzel/pyxis/pull/123)

## [0.0.8] - 2025-02-28

### Added

- Add vcpkg baseline and specify proj version by @Glatzel in [#107](https://github.com/Glatzel/pyxis/pull/107)
- Add linux build by @Glatzel in [#118](https://github.com/Glatzel/pyxis/pull/118)

### Changed

- Embed proj.db into cli by @Glatzel in [#109](https://github.com/Glatzel/pyxis/pull/109)
- New build type by @Glatzel in [#116](https://github.com/Glatzel/pyxis/pull/116)

## [0.0.7] - 2025-02-26

### Added

- Initial bench by @Glatzel in [#83](https://github.com/Glatzel/pyxis/pull/83)

### Changed

- Complete bench for crypto by @Glatzel in [#85](https://github.com/Glatzel/pyxis/pull/85)
- Modify crypto exact recursion exit by @Glatzel in [#94](https://github.com/Glatzel/pyxis/pull/94)

### Performance

- Raise crypto precision by @Glatzel in [#89](https://github.com/Glatzel/pyxis/pull/89)
- Revert to original crypto algorithm by @Glatzel in [#98](https://github.com/Glatzel/pyxis/pull/98)
- Improve crypto exact precision by @Glatzel in [#97](https://github.com/Glatzel/pyxis/pull/97)

### Refactor

- Use closures in crypto exact by @Glatzel in [#93](https://github.com/Glatzel/pyxis/pull/93)

### Testing

- Optimize crypto bench by @Glatzel in [#87](https://github.com/Glatzel/pyxis/pull/87)
- Revert to use threshold to bench crypto exact by @Glatzel in [#88](https://github.com/Glatzel/pyxis/pull/88)
- Add new thershold to crypto bench by @Glatzel in [#90](https://github.com/Glatzel/pyxis/pull/90)
- Modify some test precision assert by @Glatzel in [#91](https://github.com/Glatzel/pyxis/pull/91)
- Optimize crypto exact bench input value by @Glatzel in [#100](https://github.com/Glatzel/pyxis/pull/100)

## [0.0.6] - 2025-02-20

### Added

- Add cargo-clean script by @Glatzel in [#64](https://github.com/Glatzel/pyxis/pull/64)
- Add example by @Glatzel in [#68](https://github.com/Glatzel/pyxis/pull/68)
- Add ellipsoid to cli record by @Glatzel in [#69](https://github.com/Glatzel/pyxis/pull/69)
- Add cli example by @Glatzel in [#70](https://github.com/Glatzel/pyxis/pull/70)
- Add log feature to recursion algorithm by @Glatzel in [#78](https://github.com/Glatzel/pyxis/pull/78)
- Add some cli log message by @Glatzel in [#79](https://github.com/Glatzel/pyxis/pull/79)

### Changed

- Implement coordinate syetem migrate 2d by @Glatzel in [#71](https://github.com/Glatzel/pyxis/pull/71)

### Refactor

- Rename file, parameters by @Glatzel in [#74](https://github.com/Glatzel/pyxis/pull/74)

### Testing

- Add cli tests by @Glatzel in [#73](https://github.com/Glatzel/pyxis/pull/73)
- Add cli scale test by @Glatzel in [#75](https://github.com/Glatzel/pyxis/pull/75)
- Add algorithm rotate 2d test by @Glatzel in [#76](https://github.com/Glatzel/pyxis/pull/76)
- Add cli rotate test by @Glatzel in [#77](https://github.com/Glatzel/pyxis/pull/77)

## [0.0.5] - 2025-02-15

### Documentation

- Replace parameters with arguments in rust doc by @Glatzel in [#57](https://github.com/Glatzel/pyxis/pull/57)

### Performance

- Raise precision of `gcj02_to_wgs84` by @Glatzel in [#59](https://github.com/Glatzel/pyxis/pull/59)
- Raise precision of `bd09_to_gcj02` by @Glatzel in [#62](https://github.com/Glatzel/pyxis/pull/62)

### Refactor

- Migrate log to independent repo by @Glatzel in [#54](https://github.com/Glatzel/pyxis/pull/54)

### Removed

- Delete slug parameter in codecov by @Glatzel in [#55](https://github.com/Glatzel/pyxis/pull/55)

## [0.0.4] - 2025-02-12

### Changed

- Prettify cli output by @Glatzel in [#47](https://github.com/Glatzel/pyxis/pull/47)

### Documentation

- Add python doc by @Glatzel in [#50](https://github.com/Glatzel/pyxis/pull/50)
- Fix some python docs by @Glatzel in [#51](https://github.com/Glatzel/pyxis/pull/51)

## [0.0.3] - 2025-02-11

### Added

- Add cli transform subcommands by @Glatzel in [#31](https://github.com/Glatzel/pyxis/pull/31)
- Add `from_semi_axis` to ellipsoid by @Glatzel in [#40](https://github.com/Glatzel/pyxis/pull/40)

### Fixed

- Cli args parser log by @Glatzel in [#30](https://github.com/Glatzel/pyxis/pull/30)
- RotateUnit from str by @Glatzel in [#32](https://github.com/Glatzel/pyxis/pull/32)
- `from_semi_axis` doctest by @Glatzel in [#41](https://github.com/Glatzel/pyxis/pull/41)
- Cargo test never fail by @Glatzel in [#42](https://github.com/Glatzel/pyxis/pull/42)

### Performance

- Optimize test and coverage by @Glatzel in [#35](https://github.com/Glatzel/pyxis/pull/35)

### Refactor

- Use `Ellipsoid` in some functions by @Glatzel in [#36](https://github.com/Glatzel/pyxis/pull/36)

### Testing

- Optimize assert float by @Glatzel in [#39](https://github.com/Glatzel/pyxis/pull/39)

## [0.0.2] - 2025-02-07

### Added

- Add rust cov by @Glatzel in [#2](https://github.com/Glatzel/pyxis/pull/2)
- Add doctest by @Glatzel in [#8](https://github.com/Glatzel/pyxis/pull/8)
- Add react to comment in changelog ci by @Glatzel in [#12](https://github.com/Glatzel/pyxis/pull/12)

### Documentation

- Complete doc for crypto by @Glatzel in [#15](https://github.com/Glatzel/pyxis/pull/15)

### Fixed

- Python function return type for scalar array by @Glatzel in [#7](https://github.com/Glatzel/pyxis/pull/7)

### Removed

- Remove changelog ci by @Glatzel in [#14](https://github.com/Glatzel/pyxis/pull/14)

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
