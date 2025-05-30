$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
if (Test-Path $PSScriptRoot/setup.ps1) {
    &$PSScriptRoot/setup.ps1
}
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
if (-not $env:CI) {
    cargo +stable clippy --fix
}
cargo +stable clippy -- -Dwarnings
Set-Location $ROOT
