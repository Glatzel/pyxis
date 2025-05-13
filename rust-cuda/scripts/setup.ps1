pixi install

if ($IsWindows) {
    $env:CUDA_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library
}
if ($IsLinux) {
    $env:CUDA_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default
}
