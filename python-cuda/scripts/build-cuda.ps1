$ROOT = git rev-parse --show-toplevel
$install=Resolve-Path  $PSScriptRoot/../src/pyxis_cuda
& $ROOT/cuda/scripts/build-cuda.ps1 -install $install
Set-Location $PSScriptRoot/..
Remove-Item ./src/pyxis_cuda/ptx -Recurse -Force -ErrorAction SilentlyContinue
New-Item ./src/pyxis_cuda/ptx -ItemType Directory -ErrorAction SilentlyContinue
Copy-Item $ROOT/cuda/dist/ptx/* ./src/pyxis_cuda/ptx/
Set-Location $ROOT

