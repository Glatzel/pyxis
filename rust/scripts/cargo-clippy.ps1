$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

if ($env:CI) {
    cargo +stable clippy --fix --all-features -p pyxis -p pyxis-cli -p proj -- -D warnings
    cargo +stable clippy --all-features -p pyxis -p pyxis-cli -p proj -- -D warnings
}
else {
    cargo clippy --fix --all-features -- -D warnings
    cargo clippy --all-targets --all-features -- -D warnings
}
Write-Error "fas"
Set-Location $ROOT
