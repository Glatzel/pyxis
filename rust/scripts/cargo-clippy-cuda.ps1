$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
cargo +stable clippy --all-features -p pyxis-cuda
Set-Location $ROOT
