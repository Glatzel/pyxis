$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot
Set-Location ..

cargo machete
Set-Location $PSScriptRoot
Set-Location $ROOT
