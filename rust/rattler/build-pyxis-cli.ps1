New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
New-Item $env:PREFIX/pyxis/bin -ItemType Directory -ErrorAction SilentlyContinue
if ($IsWindows) {
    Copy-Item $PSScriptRoot/../dist/cli/pyxis.exe "$env:PREFIX/pyxis/bin/pyxis.exe"
}
else {
    Copy-Item $PSScriptRoot/../dist/cli/pyxis "$env:PREFIX/pyxis/bin/pyxis"
}
