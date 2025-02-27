Set-Location -Path "$PSScriptRoot"
Set-Location -Path ".."
$files = Get-ChildItem -Path ".\crates\py-geotool\geotool\*.py" -Recurse
pixi run numpydoc lint $files
