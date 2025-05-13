param($filter)

$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
&$PSScriptRoot/setup.ps1
cargo bench --all -- $filter
Set-Location $PSScriptRoot
Set-Location $ROOT
