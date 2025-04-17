$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

if ($env:CI) {
    cargo +stable clippy --all-features --all-targets
}
else {
    pixi run cargo clippy --fix --all-targets --all-features
}

Set-Location $ROOT
