param (
    [ValidateSet($null,"-r")]
    $config = $null
)



Set-Location $PSScriptRoot
Set-Location ..
Remove-Item ../../../../dist/pyxis*.whl -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/pyxis_py.pyd -ErrorAction SilentlyContinue
Remove-Item crates/pyxis-py/pyxis/**__pycache__ -Recurse -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
pixi run cargo build $config -p pyxis-py
Set-Location crates/pyxis-py
pixi run maturin build --out ../../../../dist $config


Set-Location $PSScriptRoot
Set-Location ../../../
