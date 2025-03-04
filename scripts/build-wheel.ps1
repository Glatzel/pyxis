param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item dist/pyxis*.whl -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/py_pyxis.pyd -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/**__pycache__ -Recurse -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config --lib
    Set-Location crates/py-pyxis
    pixi run maturin build --out ../../dist --profile $config
}
else {
    pixi run cargo build --lib
    Set-Location crates/py-pyxis
    pixi run maturin build --out ../../dist
}
