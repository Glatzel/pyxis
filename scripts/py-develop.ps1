param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel

# build-cuda
& "$ROOT/scripts/build-cuda.ps1"
Remove-Item $ROOT/src/python/pyxis/cuda/ptx -Recurse -Force -ErrorAction SilentlyContinue
New-Item $ROOT/src/python/pyxis/cuda/ptx -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item $ROOT/dist/pyxis-cuda/ptx/* $ROOT/src/python/pyxis/cuda/ptx/

Set-Location $PSScriptRoot/..
Set-Location src/python
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

pixi run cargo build --profile  $config -p pyxis-py
pixi run maturin develop --profile $config

Set-Location $ROOT
