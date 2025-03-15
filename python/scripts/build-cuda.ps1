$ROOT = git rev-parse --show-toplevel
& $ROOT/cuda/scripts/build-cuda.ps1
Set-Location $PSScriptRoot/..
Remove-Item $ROOT/python/pyxis/pyxis_cuda/ptx -Recurse -Force -ErrorAction SilentlyContinue
New-Item $ROOT/python/pyxis/pyxis_cuda/ptx -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item $ROOT/cuda/dist/ptx/* ./python/pyxis/pyxis_cuda/ptx/

Set-Location $ROOT
