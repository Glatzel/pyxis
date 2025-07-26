New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
New-Item $env:PREFIX/pyxis/bin -ItemType Directory -ErrorAction SilentlyContinue
if ($IsWindows) {
    Copy-Item $PSScriptRoot/../target/release/pyxis.exe "$env:PREFIX/pyxis/bin/"
    Copy-Item $PSScriptRoot/../target/release/pyxis.exe "$env:PREFIX/pyxis/bin/"
}
else {
    Copy-Item $PSScriptRoot/../target/release/pyxis "$env:PREFIX/pyxis/bin/"
    Copy-Item $PSScriptRoot/../target/release/pyxis "$env:PREFIX/pyxis/bin/"
}
