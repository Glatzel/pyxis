param($filter)

$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1
cargo bench --all -- $filter
Set-Location $PSScriptRoot
Set-Location $ROOT
