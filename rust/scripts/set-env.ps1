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

    # pkg-config
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $env:PKG_CONFIG_PATH = Resolve-Path "./.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig"

    $env:PATH = "$lib_path;$cl_path;$nvcc_path;$pkg_config_exe;$env:PATH"
    Copy-Item ./.pixi/envs/default/proj/x64-windows-static/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsMacOS) {
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/bin
    $env:Path = "$pkg_config_exe" + ":" + "$env:Path"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/lib/pkgconfig
    Copy-Item ./.pixi/envs/default/proj/arm64-osx-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/bin
    $nvcc_path = Resolve-Path $PSScriptRoot/../.pixi/envs/default/bin
    $env:CUDA_LIBRARY_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default
    $env:PATH = "$nvcc_path" + ":" + "$pkg_config_exe" + ":" + "$env:PATH"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
    Copy-Item ./.pixi/envs/default/proj/x64-linux-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}

Set-Location $current_dir
