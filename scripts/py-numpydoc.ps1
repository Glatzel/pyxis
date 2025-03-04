Set-Location -Path "$PSScriptRoot"
Set-Location -Path ".."
$files = Get-ChildItem -Path ".\crates\py-pyxis\pyxis\*.py" -Recurse
pixi run numpydoc lint $files
