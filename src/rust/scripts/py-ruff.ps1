Set-Location $PSScriptRoot
Set-Location ..
pixi run ruff format
pixi run ruff check --fix
Set-Location $PSScriptRoot
Set-Location ../../../
