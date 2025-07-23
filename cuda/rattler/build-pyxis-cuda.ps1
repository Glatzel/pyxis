New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
$ROOT = git rev-parse --show-toplevel
if ($IsLinux -or $IsMacOS) {
    Copy-Item "$ROOT/rust/dist/cli/pyxis" "$env:PREFIX/pyxis/bin/pyxis"
}
