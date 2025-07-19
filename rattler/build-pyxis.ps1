New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
New-Item $env:PREFIX/pyxis/bin -ItemType Directory -ErrorAction SilentlyContinue
$ROOT = git rev-parse --show-toplevel
if ($IsWindows) {
    Copy-Item "$ROOT/rust/dist/cli/pyxis.exe" "$env:PREFIX/pyxis/bin/pyxis.exe"
}
if ($IsLinux -or $IsMacOS) {
    Copy-Item "$ROOT/rust/dist/cli/pyxis/" "$env:PREFIX/pyxis/bin/pyxis"
}
Copy-Item "$ROOT/cpp/dist/*" "$env:PREFIX/pyxis/" -Recurse
if (-not $IsMacOS) {
    Copy-Item "$ROOT/cuda/dist/*" "$env:PREFIX/pyxis/" -Recurse
}
