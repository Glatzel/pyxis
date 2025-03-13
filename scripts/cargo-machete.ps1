$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

cargo machete
Set-Location $PSScriptRoot
Set-Location $ROOT
