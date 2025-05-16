$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel

& "$ROOT/cpp/scripts/build-cpp.ps1" -config release
& "$ROOT/cuda/scripts/build-cuda.ps1"
& "$ROOT/rust/scripts/build-rust-cli.ps1" -config release

Set-Location $PSScriptRoot
pixi run rattler-build build
New-Item $PSScriptRoot/../dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $PSScriptRoot/../dist
Set-Location $ROOT
