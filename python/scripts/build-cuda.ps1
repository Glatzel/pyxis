# install to python src
Remove-Item $ROOT/src/python/pyxis/cuda/ptx -Recurse -Force -ErrorAction SilentlyContinue
New-Item $ROOT/src/python/pyxis/cuda/ptx -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item $ROOT/dist/pyxis-cuda/ptx/* $ROOT/src/python/pyxis/cuda/ptx/

Set-Location $ROOT