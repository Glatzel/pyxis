Set-Location -Path "$PSScriptRoot"
Set-Location -Path ".."
$files = Get-ChildItem -Path ".\src/rust\pyxis-py\pyxis\*.py" -Recurse
pixi run numpydoc lint $files
