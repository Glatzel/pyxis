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
7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on -sccUTF-8 -bb0 -bse0 -bsp2 "-wdist" -mtc=on -mta=on "dist\geotool.7z" "dist\geotool.exe"
