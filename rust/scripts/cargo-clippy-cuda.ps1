$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
pixi run -e gpu cargo +stable clippy --all-features -p pyxis-cuda
Set-Location $ROOT
