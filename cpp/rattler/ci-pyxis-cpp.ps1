$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
& "$ROOT/cpp/scripts/build-cpp.ps1" -config Release
Set-Location $PSScriptRoot
pixi run rattler-build build
New-Item $PSScriptRoot/../dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item ./output/**.conda $PSScriptRoot/../dist
Set-Location $ROOT
