$ROOT = git rev-parse --show-toplevel

Set-Location $PSScriptRoot/..
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue

pixi run maturin develop

Set-Location $ROOT
