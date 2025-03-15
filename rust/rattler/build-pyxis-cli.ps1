New-Item $env:PREFIX/bin/pyxis-cli -ItemType Directory
if ($IsWindows) {
    Copy-Item "$env:RECIPE_DIR/../dist/cli/static/pyxis.exe" "$env:PREFIX/bin/pyxis-cli/pyxis.exe"
}
if ($IsLinux) {
    Copy-Item "$env:RECIPE_DIR/../dist/cli/static/pyxis" "$env:PREFIX/bin/pyxis-cli/pyxis"
}
