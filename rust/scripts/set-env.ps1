$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot/..
if ($env:CI) {
    pixi install --all --no-progress
}
else {
    pixi install --all
}

if ($IsWindows) {
    $pkg_config_exe = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $nvcc_path = Resolve-Path $PSScriptRoot/../.pixi/envs/gpu/Library/bin
    $env:CUDA_ROOT = Resolve-Path $PSScriptRoot/../.pixi/envs/gpu/Library
    $env:PATH = "$nvcc_path;$pkg_config_exe;$env:PATH"
    $env:PKG_CONFIG_PATH = Resolve-Path "./.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig"
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
    $nvcc_path = Resolve-Path $PSScriptRoot/../.pixi/envs/gpu/bin
    $env:CUDA_LIBRARY_PATH=Resolve-Path $PSScriptRoot/../.pixi/envs/gpu
    $env:PATH = "$nvcc_path" + ":" + "$pkg_config_exe" + ":" + "$env:PATH"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
    Copy-Item ./.pixi/envs/default/proj/x64-linux-release/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}

Set-Location $current_dir
