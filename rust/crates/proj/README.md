# proj-rs

![PROJ](https://img.shields.io/badge/Proj-9.7.0-blue?logo=rust)

**Safe Rust bindings for [PROJ](https://proj.org/)** – the definitive library for cartographic projections and coordinate transformations.

## Features

* **Safe & idiomatic Rust API**
* **Coordinate transformation in 2D, 3D, and 4D**
* **Leverages native PROJ capabilities**
* **Extended utilities**
* **Customizable coordinate type**

## Goals

This crate aims to:

* Provide a **safe and ergonomic API** for working with PROJ in Rust.
* Preserve the **native behavior and power** of PROJ.
* **Extend coordinate conversion** capabilities to 3D and 4D use cases.
* Offer **additional helper functions** that go beyond core PROJ usage.

## Installation

### Method 1 – Git

1. Add this to your `Cargo.toml`:

   ```toml
   [dependencies]
   proj = { git = "https://github.com/Glatzel/clerk", tag = "version" }
   ```

2. Set the `PKG_CONFIG_PATH` environment variable to the directory that contains `proj.pc`.
    Windows users must install `pkg-config` and add the binary path to their `PATH` environment variable.

### Method 2 – Pixi

Use [pixi](https://github.com/prefix-dev/pixi/?tab=readme-ov-file#installation) to install all dependencies.

```toml
[dependencies]
pkg-config = "==0.29.2"
proj = { version = "==9.6.2", channel = "https://prefix.dev/glatzel" }
```

## Usage

Check examples folder.

## Development

### Dependencies

* [pixi](https://github.com/prefix-dev/pixi/?tab=readme-ov-file#installation)

### Folder Structure of src

```yaml
- data_type: Native Proj data types.
- extension: Extra utilities or native Proj function override.
- functions: Native Proj functions.
```

### General Rules

* `data_type`, `functions` folder only contains native Proj objects and implementation for `std`.
* To add new tools or APIs, put them in `extension` folder.
* If extended API cover the native function, consider to set the native function private and add link to extended API in native function doc.

## Unimplemented

### Data Types

| Module                                    | Name     | Reason                       | References                                                                        |
| ----------------------------------------- | -------- | ---------------------------- | --------------------------------------------------------------------------------- |
| 2 dimensional coordinates                 | PJ_LP    | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LP)    |
| 2 dimensional coordinates                 | PJ_UV    | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UV)    |
| 2 dimensional coordinates                 | PJ_XY    | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_XY)    |
| 3 dimensional coordinates                 | PJ_LPZ   | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LPZ)   |
| 3 dimensional coordinates                 | PJ_UVW   | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UVW)   |
| 3 dimensional coordinates                 | PJ_XYZ   | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_XYZ)   |
| Ancillary types for geodetic computations | PJ_COORD | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COORD) |
| Ancillary types for geodetic computations | PJ_ENU   | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ENU)   |
| Ancillary types for geodetic computations | PJ_ENU   | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ENU)   |
| Ancillary types for geodetic computations | PJ_OPK   | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPK)   |
| Spatiotemporal coordinate types           | PJ_LPZT  | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LPZT)  |
| Spatiotemporal coordinate types           | PJ_UVWT  | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UVWT)  |
| Spatiotemporal coordinate types           | PJ_XYZT  | Use `impl ICoord` to instead | [doc](https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_XYZT)  |

### Functions

| Module                            | Name                                 | Reason                                                         | References                                                                                                  |
| --------------------------------- | ------------------------------------ | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| C API for ISO-19111 functionality | proj_context_set_autoclose_database  | Deprecated.                                                    | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_context_set_autoclose_database)  |
| C API for ISO-19111 functionality | proj_get_crs_list_parameters_create  | Directly called in other fn.                                   | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_get_crs_list_parameters_create)  |
| C API for ISO-19111 functionality | proj_get_crs_list_parameters_destroy | Directly called in other fn.                                   | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_get_crs_list_parameters_destroy) |
| C API for ISO-19111 functionality | proj_int_list_destroy                | Directly called in other fn.                                   | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_int_list_destroy)                |
| C API for ISO-19111 functionality | proj_is_derived_crs                  | Use `Proj::crs_is_derived` to instead.                         | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_is_derived_crs)                  |
| C API for ISO-19111 functionality | proj_string_destroy                  | Useless.                                                       | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_string_destroy)                  |
| C API for ISO-19111 functionality | proj_string_list_destroy             | Directly called in other fn.                                   | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_string_list_destroy)             |
| C API for ISO-19111 functionality | proj_unit_list_destroy               | Directly called in other fn.                                   | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_unit_list_destroy)               |
| Cleanup                           | proj_cleanup                         | Useless.                                                       | [doc](https://proj.org/en/stable/development/reference/functions.html#cleanup)                              |
| Coordinate transformation         | proj_trans                           | Use `Proj::convert` or `Proj::project` to instead.             | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_trans)                           |
| Coordinate transformation         | proj_trans_array                     | Use `Proj::convert_array` or `Proj::project_array` to instead. | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_trans_array)                     |
| Logging                           | proj_log_func                        | Use `Context::set_log_fn` to instead.                          | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_log_func)                        |
| Logging                           | proj_log_level                       | Use `Context::set_log_level` to instead.                       | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_log_level)                       |
| Various                           | proj_coord                           | Use `impl ICoord` to instead.                                  | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_coord)                           |
| Various                           | proj_rtodms                          | Deprecated.                                                    | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_rtodms)                          |
| Various                           | proj_todeg                           | Use `f64::to_degrees` to instead.                              | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_todeg)                           |
| Various                           | proj_torad                           | Use `f64::to_radians` to instead.                              | [doc](https://proj.org/en/stable/development/reference/functions.html#proj_torad)                           |

## References

* [Proj](https://proj.org/en/stable/)
* [georust/proj](https://github.com/georust/proj)
* [pyproj](https://pyproj4.github.io/pyproj/stable/)
