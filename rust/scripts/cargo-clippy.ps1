$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo clippy --fix --all-features -- -D warnings -p pyxis -p pyxis-cli -p proj
}
else {
    cargo clippy --fix --all-targets --all-features -- -D warnings
}
Set-Location $ROOT
