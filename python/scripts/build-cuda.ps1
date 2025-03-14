$ROOT = git rev-parse --show-toplevel
& $ROOT/cuda/scripts/build-cuda.ps1
Remove-Item $ROOT/python/pyxis/pyxis_cuda/ptx -Recurse -Force -ErrorAction SilentlyContinue
New-Item $ROOT/python/pyxis/pyxis_cuda/ptx -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item $ROOT/dist/cuda/ptx/* $ROOT/python/pyxis/pyxis_cuda/ptx/

Set-Location $ROOT