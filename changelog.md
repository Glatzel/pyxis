# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.7] - 2025-02-26

### Changed

- Initial bench by @Glatzel in [#83](https://github.com/Glatzel/rs-geotool/pull/83)
- Complete bench for crypto by @Glatzel in [#85](https://github.com/Glatzel/rs-geotool/pull/85)
- Modify crypto exact recursion exit by @Glatzel in [#94](https://github.com/Glatzel/rs-geotool/pull/94)

### Performance

- Raise crypto precision by @Glatzel in [#89](https://github.com/Glatzel/rs-geotool/pull/89)
- Revert to original crypto algorithm by @Glatzel in [#98](https://github.com/Glatzel/rs-geotool/pull/98)
- Improve crypto exact precision by @Glatzel in [#97](https://github.com/Glatzel/rs-geotool/pull/97)

### Refactor

- Use closures in crypto exact by @Glatzel in [#93](https://github.com/Glatzel/rs-geotool/pull/93)

### Testing

- Optimize crypto bench by @Glatzel in [#87](https://github.com/Glatzel/rs-geotool/pull/87)
- Revert to use threshold to bench crypto exact by @Glatzel in [#88](https://github.com/Glatzel/rs-geotool/pull/88)
- Add new thershold to crypto bench by @Glatzel in [#90](https://github.com/Glatzel/rs-geotool/pull/90)
- Modify some test precision assert by @Glatzel in [#91](https://github.com/Glatzel/rs-geotool/pull/91)
- Optimize crypto exact bench input value by @Glatzel in [#100](https://github.com/Glatzel/rs-geotool/pull/100)

## [0.0.6] - 2025-02-20

### Added

- Add cargo-clean script by @Glatzel in [#64](https://github.com/Glatzel/rs-geotool/pull/64)
- Add example by @Glatzel in [#68](https://github.com/Glatzel/rs-geotool/pull/68)
- Add ellipsoid to cli record by @Glatzel in [#69](https://github.com/Glatzel/rs-geotool/pull/69)
- Add cli example by @Glatzel in [#70](https://github.com/Glatzel/rs-geotool/pull/70)
- Add log feature to recursion algorithm by @Glatzel in [#78](https://github.com/Glatzel/rs-geotool/pull/78)
- Add some cli log message by @Glatzel in [#79](https://github.com/Glatzel/rs-geotool/pull/79)

### Changed

- Implement coordinate syetem migrate 2d by @Glatzel in [#71](https://github.com/Glatzel/rs-geotool/pull/71)

### Refactor

- Rename file, parameters by @Glatzel in [#74](https://github.com/Glatzel/rs-geotool/pull/74)

### Testing

- Add cli tests by @Glatzel in [#73](https://github.com/Glatzel/rs-geotool/pull/73)
- Add cli scale test by @Glatzel in [#75](https://github.com/Glatzel/rs-geotool/pull/75)
- Add algorithm rotate 2d test by @Glatzel in [#76](https://github.com/Glatzel/rs-geotool/pull/76)
- Add cli rotate test by @Glatzel in [#77](https://github.com/Glatzel/rs-geotool/pull/77)

## [0.0.5] - 2025-02-15

### Documentation

- Replace parameters with arguments in rust doc by @Glatzel in [#57](https://github.com/Glatzel/rs-geotool/pull/57)

### Performance

- Raise precision of `gcj02_to_wgs84` by @Glatzel in [#59](https://github.com/Glatzel/rs-geotool/pull/59)
- Raise precision of `bd09_to_gcj02` by @Glatzel in [#62](https://github.com/Glatzel/rs-geotool/pull/62)

### Refactor

- Migrate log to independent repo by @Glatzel in [#54](https://github.com/Glatzel/rs-geotool/pull/54)

### Removed

- Delete slug parameter in codecov by @Glatzel in [#55](https://github.com/Glatzel/rs-geotool/pull/55)

## [0.0.4] - 2025-02-12

### Changed

- Prettify cli output by @Glatzel in [#47](https://github.com/Glatzel/rs-geotool/pull/47)

### Documentation

- Add python doc by @Glatzel in [#50](https://github.com/Glatzel/rs-geotool/pull/50)
- Fix some python docs by @Glatzel in [#51](https://github.com/Glatzel/rs-geotool/pull/51)

## [0.0.3] - 2025-02-11

### Added

- Add cli transform subcommands by @Glatzel in [#31](https://github.com/Glatzel/rs-geotool/pull/31)
- Add `from_semi_axis` to ellipsoid by @Glatzel in [#40](https://github.com/Glatzel/rs-geotool/pull/40)

### Fixed

- Cli args parser log by @Glatzel in [#30](https://github.com/Glatzel/rs-geotool/pull/30)
- RotateUnit from str by @Glatzel in [#32](https://github.com/Glatzel/rs-geotool/pull/32)
- `from_semi_axis` doctest by @Glatzel in [#41](https://github.com/Glatzel/rs-geotool/pull/41)
- Cargo test never fail by @Glatzel in [#42](https://github.com/Glatzel/rs-geotool/pull/42)

### Performance

- Optimize test and coverage by @Glatzel in [#35](https://github.com/Glatzel/rs-geotool/pull/35)

### Refactor

- Use `Ellipsoid` in some functions by @Glatzel in [#36](https://github.com/Glatzel/rs-geotool/pull/36)

### Testing

- Optimize assert float by @Glatzel in [#39](https://github.com/Glatzel/rs-geotool/pull/39)

## [0.0.2] - 2025-02-07

### Added

- Add rust cov by @Glatzel in [#2](https://github.com/Glatzel/rs-geotool/pull/2)
- Add doctest by @Glatzel in [#8](https://github.com/Glatzel/rs-geotool/pull/8)
- Add react to comment in changelog ci by @Glatzel in [#12](https://github.com/Glatzel/rs-geotool/pull/12)

### Documentation

- Complete doc for crypto by @Glatzel in [#15](https://github.com/Glatzel/rs-geotool/pull/15)

### Fixed

- Python function return type for scalar array by @Glatzel in [#7](https://github.com/Glatzel/rs-geotool/pull/7)

### Removed

- Remove changelog ci by @Glatzel in [#14](https://github.com/Glatzel/rs-geotool/pull/14)

[0.0.7]: https://github.com/Glatzel/rs-geotool/compare/v0.0.6..v0.0.7
[0.0.6]: https://github.com/Glatzel/rs-geotool/compare/v0.0.5..v0.0.6
[0.0.5]: https://github.com/Glatzel/rs-geotool/compare/v0.0.4..v0.0.5
[0.0.4]: https://github.com/Glatzel/rs-geotool/compare/v0.0.3..v0.0.4
[0.0.3]: https://github.com/Glatzel/rs-geotool/compare/v0.0.2..v0.0.3
[0.0.2]: https://github.com/Glatzel/rs-geotool/compare/v0.0.1..v0.0.2

<!-- generated by git-cliff -->
