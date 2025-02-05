param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug",
    [switch]$clean
)

Set-Location $PSScriptRoot
Set-Location ..
Remove-Item dist/geotool*.whl -ErrorAction SilentlyContinue
Remove-Item crates/py-geotool/geotool/py_geotool.pyd -ErrorAction SilentlyContinue
Remove-Item crates/py-geotool/geotool/**__pycache__ -Recurse -ErrorAction SilentlyContinue

if ($clean) { cargo clean }
Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config --lib
    Set-Location crates/py-geotool
    pixi run -e dev maturin build --out ../../dist --profile $config
}
else {
    pixi run cargo build --lib
    Set-Location crates/py-geotool
    pixi run -e dev maturin build --out ../../dist
}
Set-Location $PSScriptRoot
Set-Location ..
Remove-Item ./dist/geotool -Recurse -Force -ErrorAction SilentlyContinue
New-Item ./dist/geotool -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item "target/$config/geotool.exe" ./dist/geotool
Copy-Item "vcpkg_installed/x64-windows/share/proj/proj.db" ./dist/geotool
Copy-Item "vcpkg_installed/x64-windows/bin/*.dll" ./dist/geotool
