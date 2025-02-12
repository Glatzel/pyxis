& $PSScriptRoot/develop.ps1
Set-Location -Path $PSScriptRoot
Set-Location -Path ..
Set-Location -Path crates/py-geotool

Remove-Item -Path "./doc/source/reference/api" -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "./doc/build" -Recurse -ErrorAction SilentlyContinue
pixi run sphinx-build -M html ./doc/source ./doc/build --fail-on-warning

Remove-Item $PSScriptRoot/../dist/python-doc.zip -Force -ErrorAction SilentlyContinue
New-Item $PSScriptRoot/../dist -ItemType Directory -ErrorAction SilentlyContinue
Compress-Archive ./target/doc "$PSScriptRoot/../dist/python-doc.zip"
