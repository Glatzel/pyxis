New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
New-Item $env:PREFIX/pyxis/bin -ItemType Directory -ErrorAction SilentlyContinue
if ($IsWindows) {
    Copy-Item $PSScriptRoot/../target/release/pyxis-abacus.exe "$env:PREFIX/pyxis/bin/"
    Copy-Item $PSScriptRoot/../target/release/pyxis-trail.exe "$env:PREFIX/pyxis/bin/"
}
else {
    Copy-Item $PSScriptRoot/../target/release/pyxis-abacus "$env:PREFIX/pyxis/bin/"
    Copy-Item $PSScriptRoot/../target/release/pyxis-trail "$env:PREFIX/pyxis/bin/"
}
