$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
if ($env:CI) {
    pixi install --no-progress
}
else {
    pixi install
}

if ($IsWindows) {
    # find visual studio
    $path = pixi run vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $cl_path = join-path $path 'VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64'
    $env:INCLUDE = join-path $path  'VC\Tools\MSVC\14.43.34808\include'

    # find cuda
    $nvcc_path = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $lib_path = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/lib
    $env:CUDA_ROOT = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library

    $env:PATH = "$cl_path;$lib_path;$nvcc_path;$env:PATH"
}

if ($IsLinux) {
    $nvcc_path = Resolve-Path $PSScriptRoot/../.pixi/envs/default/bin
    $env:CUDA_LIBRARY_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default
    $env:PATH = "$nvcc_path" + ":" + "$env:PATH"
}

Set-Location $current_dir
