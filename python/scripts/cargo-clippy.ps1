$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
cargo clippy --fix --all-targets --all-features -- -D warnings
Set-Location $ROOT
