$ROOT = git rev-parse --show-toplevel
New-Item $env:PREFIX/bin/pyxis-cli -ItemType Directory
if ($IsWindows) {
    Copy-Item "$ROOT/dist/cli/static/pyxis.exe" "$env:PREFIX/bin/pyxis-cli/pyxis.exe"
}
if ($IsLinux) {
    Copy-Item "$ROOT/dist/cli/static/pyxis" "$env:PREFIX/bin/pyxis-cli/pyxis"
}
