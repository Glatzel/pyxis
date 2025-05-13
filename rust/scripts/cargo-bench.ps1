param($filter)

$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
cargo bench --all -- $filter
Set-Location $PSScriptRoot
Set-Location $ROOT
