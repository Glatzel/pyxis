Set-Location $PSScriptRoot
pixi install

if ($IsWindows) {
    $env:CUDA_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library
    $env:LIB = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/lib
}
if ($IsLinux) {
    $lib = Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib
    $env:Path = "$lib" + ":" + "$env:Path"
    $env:CUDA_LIBRARY_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default
}
