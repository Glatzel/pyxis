param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item dist/pyxis*.whl -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/pyxis_py.pyd -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/**__pycache__ -Recurse -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config -p pyxis-py
    Set-Location crates/pyxis-py
    pixi run maturin build --out ../../dist --profile $config
}
else {
    pixi run cargo build -p pyxis-py
    Set-Location crates/pyxis-py
    pixi run maturin build --out ../../dist
}
