# pyxis

[![Release](https://img.shields.io/github/v/release/Glatzel/pyxis)](https://github.com/Glatzel/pyxis/releases/latest)
![CI](https://github.com/Glatzel/pyxis/actions/workflows/ci.yml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/Glatzel/pyxis/graph/badge.svg?token=I6L8Y698AR)](https://codecov.io/gh/Glatzel/pyxis)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/Glatzel/pyxis)
![CUDA Version](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia&logoColor=white)

**Pyxis** is a multi-language library and toolkit for coordinate conversion and processing.  
It currently contains implementations in **Rust, Python, C, CUDA, C#, and OpenCL**.

The repository provides a central framework for coordinate-related computation, transformations, and experimentation.  
Details for each language, including API, tools, and target platforms, are maintained in their respective subfolders.

```mermaid
flowchart TD

cuda --> c++
rust-cuda --> cuda

rust-cuda ---> rust
python --> rust
python-cuda --> cuda

c#
opencl

style c# fill         : #e884e8, stroke: #e884e8, stroke-width: 4px
style c++ fill        : #6da9ff, stroke: #6da9ff, stroke-width: 4px
style cuda fill       : #a8e64d, stroke: #a8e64d, stroke-width: 4px
style opencl fill     : #ff6f61, stroke: #ff6f61, stroke-width: 4px
style python fill     : #ffd343, stroke: #ffd343, stroke-width: 4px
style python-cuda fill: #ffd343, stroke: #a8e64d, stroke-width: 4px
style rust fill       : #f7c9a8, stroke: #f7c9a8, stroke-width: 4px
style rust-cuda fill  : #f7c9a8, stroke: #a8e64d, stroke-width: 4px

```
