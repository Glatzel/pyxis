$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot
Set-Location ..
cargo clean
Set-Location $ROOT
