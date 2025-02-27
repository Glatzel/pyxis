param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1
Remove-Item dist/geotool -Recurse -ErrorAction SilentlyContinue

Write-Host "Build in $config mode."
if ($config -ne "debug") {
    pixi run cargo build --profile $config --bin geotool
}
else {
    pixi run cargo build --bin geotool
}
Set-Location $PSScriptRoot
Set-Location ..
Remove-Item ./dist/geotool*.zip -Recurse -Force -ErrorAction SilentlyContinue
New-Item ./dist -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item "target/$config/geotool.exe" ./dist/geotool.exe
Compress-Archive ./dist/geotool.exe "./dist/geotool-windows-x64.zip"
