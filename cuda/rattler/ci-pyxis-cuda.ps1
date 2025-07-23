$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
& "$ROOT/cuda/scripts/build-cuda.ps1"
Set-Location $PSScriptRoot
pixi run rattler-build build
New-Item $PSScriptRoot/../dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $PSScriptRoot/../dist
Set-Location $ROOT
