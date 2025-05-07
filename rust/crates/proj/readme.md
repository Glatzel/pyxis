# proj-rs

![PROJ](https://img.shields.io/badge/Proj-9.6.0-blue?logo=rust)

**Safe Rust bindings for [PROJ](https://proj.org/)** – the definitive library for cartographic projections and coordinate transformations.

## ✨ Features

- ✅ **Safe & idiomatic Rust API**
- 🌍 **Coordinate transformation in 2D, 3D, and 4D**
- ⚙️ **Leverages native PROJ capabilities**
- ➕ **Extended utilities**

## 🎯 Goals

This crate aims to:

- Provide a **safe and ergonomic API** for working with PROJ in Rust.
- Preserve the **native behavior and power** of PROJ.
- **Extend coordinate conversion** capabilities to 3D and 4D use cases.
- Offer **additional helper functions** that go beyond core PROJ usage.

## 📦 Installation

### Method1

1. Add this to your `Cargo.toml`:

    ```toml
    [dependencies]
    proj = { git = "https://github.com/Glatzel/clerk", tag = "version" }
    ```

2. Set `PKG_CONFIG_PATH` environment variable to the directory that contains `proj.pc`.Windows users need to install pkg-config and add the binary path to `PATH` environment variable.

### Method2

Use [pixi](https://github.com/prefix-dev/pixi/?tab=readme-ov-file#installation) to install all dependencies.

```toml
[dependencies]
pkg-config = "==0.29.2"
proj = { version = "==9.6.0", channel = "https://repo.prefix.dev/glatzel" }
```

## Development

### Dependencies

- [pixi](https://github.com/prefix-dev/pixi/?tab=readme-ov-file#installation)

### Naming Schema

#### Structs

| Proj      | This crate |
| --------- | ---------- |
| PJ        | Pj         |
| Pj_*      | Pj*        |
| PROJ_*    | Pj*        |
| Pj_PROJ_* | Pj*        |

#### Functions

| Proj | This crate |
| ---- | ---------- |
| PJ_* | *          |

### Folder Structure of src

```yaml
- data_type: Native Proj data types.
- extension: Extra utilities or native Proj function override.
- functions: Native Proj functions.
```

### General Rules

- `data_type`, `functions` folder only contains native Proj objects and implemention for `std`.
- To add new tools or APIs, put them in `extension` folder.
- If extended API cover the native function, consider to set the native function private and add link to extended API in native function doc.

## References

- [Proj](https://proj.org/en/stable/)
- [georust/proj](https://github.com/georust/proj)
- [pyproj](https://pyproj4.github.io/pyproj/stable/)
