$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

cargo +stable clippy --all-features -p pyxis-cuda
Set-Location $ROOT
