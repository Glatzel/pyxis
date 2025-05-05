$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo fmt --all -- --check --unstable-features
}
else {
    cargo fmt --all -- --unstable-features
}

Set-Location $ROOT
