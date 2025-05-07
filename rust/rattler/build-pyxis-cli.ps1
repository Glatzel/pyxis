New-Item $env:PREFIX/bin/pyxis-cli -ItemType Directory
if ($IsWindows) {
    Copy-Item "$env:RECIPE_DIR/../dist/cli/pyxis.exe" "$env:PREFIX/bin/pyxis-cli/pyxis.exe"
}
if ($IsLinux -or $IsMacOS) {
    Copy-Item "$env:RECIPE_DIR/../dist/cli/pyxis" "$env:PREFIX/bin/pyxis-cli/pyxis"
}
