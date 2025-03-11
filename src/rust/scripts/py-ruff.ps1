$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
pixi run ruff format
pixi run ruff check --fix
Set-Location $ROOT
