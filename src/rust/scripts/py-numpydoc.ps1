Set-Location -Path "$PSScriptRoot"
Set-Location -Path ".."
$files = Get-ChildItem -Path "./crates/pyxis-py/pyxis/*.py" -Recurse
pixi run numpydoc lint $files
