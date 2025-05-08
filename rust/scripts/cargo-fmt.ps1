$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1
cargo +nightly fmt --all -- --config-path $ROOT
Set-Location $ROOT
