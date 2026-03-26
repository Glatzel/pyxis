$ROOT = git rev-parse --show-toplevel
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

& $PSScriptRoot/maturin-develop.ps1
Set-Location $PSScriptRoot/..

Remove-Item -Path "./doc/source/reference/api" -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "./doc/build" -Recurse -ErrorAction SilentlyContinue
pixi run sphinx-build -M html ./doc/source ./doc/build --fail-on-warning
Set-Location $ROOT
