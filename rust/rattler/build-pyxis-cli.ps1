New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
New-Item $env:PREFIX/pyxis/bin -ItemType Directory -ErrorAction SilentlyContinue
if ($IsWindows) {
    Copy-Item $PSScriptRoot/../dist/cli/pyxis.exe "$env:PREFIX/pyxis/bin/pyxis.exe"
}
if ($IsMacOS) {
    Copy-Item $PSScriptRoot/../dist/cli/pyxis "$env:PREFIX/pyxis/bin/pyxis"
}
if ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    Copy-Item $PSScriptRoot/../target/x86_64-unknown-linux-musl/release/pyxis "$env:PREFIX/pyxis/bin/pyxis"
}
if ($IsLinux -and ($(uname -m) -eq 'aarch64' )) {
    Copy-Item $PSScriptRoot/../target/aarch64-unknown-linux-musl/release/pyxis "$env:PREFIX/pyxis/bin/pyxis"
}