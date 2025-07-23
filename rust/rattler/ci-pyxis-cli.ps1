$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
& $PSScriptRoot/../scripts/build-rust-cli.ps1 -config release
Set-Location $PSScriptRoot
pixi run rattler-build build
Set-Location $ROOT
