Set-Location $PSScriptRoot/..
Remove-Item pyxis/pyxis.pyd -ErrorAction SilentlyContinue
pixi run maturin develop
