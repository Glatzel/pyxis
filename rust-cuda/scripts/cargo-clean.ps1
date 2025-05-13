$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
&$PSScriptRoot/setup.ps1
cargo clean
Set-Location $ROOT
