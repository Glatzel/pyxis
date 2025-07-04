param (
    [ValidateSet("develop", "release")]
    $config = "develop"
)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot

if ($config -eq 'develop') {
    & "$PSScriptRoot/../scripts/maturin-develop.ps1" -config $config
    & "$PSScriptRoot/../scripts/pytest.ps1"
}
& "$PSScriptRoot/../scripts/build-python-whl.ps1" -config $config

Set-Location $PSScriptRoot
pixi run rattler-build build
Set-Location $ROOT
