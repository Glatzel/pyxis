$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

& $PSScriptRoot/py-develop.ps1
Set-Location -Path $PSScriptRoot
Set-Location -Path ..
Set-Location -Path crates/pyxis-py

Remove-Item -Path "./doc/source/reference/api" -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "./doc/build" -Recurse -ErrorAction SilentlyContinue
pixi run sphinx-build -M html ./doc/source ./doc/build --fail-on-warning

Remove-Item $PSScriptRoot/../../../dist/python-doc.zip -Force -ErrorAction SilentlyContinue
New-Item $PSScriptRoot/../../../dist -ItemType Directory -ErrorAction SilentlyContinue
7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on -sccUTF-8 -bb0 -bse0 -bsp2 "-wdoc/build" -mtc=on -mta=on "$PSScriptRoot/../../../dist/python-doc.7z" "./doc/build/*"
Set-Location $PSScriptRoot
Set-Location ../../../