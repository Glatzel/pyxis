$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1
$env:RUST_BACKTRACE=1
cargo +stable clippy --fix --all -- -D warnings
Set-Location $ROOT
