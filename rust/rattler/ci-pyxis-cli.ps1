$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& ./scripts/setup.ps1
cargo build --bin pyxis --release
Set-Location $PSScriptRoot
pixi run rattler-build build
Set-Location $ROOT
