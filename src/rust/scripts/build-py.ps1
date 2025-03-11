param (
    [ValidateSet("develop","release")]
    $config = "develop"
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item ../../../../dist/pyxis*.whl -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/pyxis_py.pyd -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/**__pycache__ -Recurse -ErrorAction SilentlyContinue
pixi run cargo build --profile $config -p pyxis-py
Set-Location crates/pyxis-py
pixi run maturin build --out ../../../../dist --profile $config

Set-Location $PSScriptRoot
Set-Location ../../../
