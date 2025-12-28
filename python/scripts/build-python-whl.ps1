Set-Location $PSScriptRoot/..
Remove-Item ./dist/pyxis*.whl -ErrorAction SilentlyContinue
Remove-Item ./pyxis/pyxis_py.pyd -ErrorAction SilentlyContinue
Remove-Item ./pyxis/**__pycache__ -Recurse -ErrorAction SilentlyContinue
pixi run maturin build --out ./dist --profile release
