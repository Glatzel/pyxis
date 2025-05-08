$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1
cargo +nightly fmt --all --check -- --config-path $ROOT
Set-Location $ROOT
