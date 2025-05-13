param (
    [ValidateSet("develop", "release")]
    $config = "develop"
)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel

Set-Location $PSScriptRoot
& "$PSScriptRoot/../scripts/build-cuda.ps1"

Set-Location $PSScriptRoot
pixi run rattler-build build
Set-Location $ROOT
