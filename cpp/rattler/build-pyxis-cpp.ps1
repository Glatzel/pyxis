New-Item $env:PREFIX/pyxis -ItemType Directory -ErrorAction SilentlyContinue
$ROOT = git rev-parse --show-toplevel
Copy-Item "$ROOT/cpp/dist/*" "$env:PREFIX/pyxis/" -Recurse
