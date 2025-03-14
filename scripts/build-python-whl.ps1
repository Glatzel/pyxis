param (
    [ValidateSet("develop","release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel

Set-Location $PSScriptRoot/..
Remove-Item $ROOT/dist/pyxis*.whl -ErrorAction SilentlyContinue
Remove-Item src/python/pyxis/pyxis_py.pyd -ErrorAction SilentlyContinue
Remove-Item src/python/pyxis/**__pycache__ -Recurse -ErrorAction SilentlyContinue
pixi run cargo build --profile $config -p pyxis-py
Set-Location src/python
pixi run maturin build --out $ROOT/dist --profile $config

Set-Location $ROOT
