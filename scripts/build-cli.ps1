param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug",
    [switch]$clean
)

Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1
Remove-Item dist/geotool -Recurse -ErrorAction SilentlyContinue

if ($clean) { cargo clean }
Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config --bin geotool
}
else {
    pixi run cargo build --bin geotool
}
Set-Location $PSScriptRoot
Set-Location ..
Remove-Item ./dist/geotool -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item ./dist/geotool*.zip -Recurse -Force -ErrorAction SilentlyContinue
New-Item ./dist/geotool -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item "target/$config/geotool.exe" ./dist/geotool
Copy-Item "vcpkg_deps/vcpkg_installed/x64-windows/share/proj/proj.db" ./dist/geotool
Copy-Item "vcpkg_deps/vcpkg_installed/x64-windows/bin/*.dll" ./dist/geotool
$version=cargo metadata --format-version=1 --no-deps | jq '.packages[0].version'
$version="$version".Replace("""","")
Compress-Archive ./dist/geotool "./dist/geotool-$version-win.zip"
