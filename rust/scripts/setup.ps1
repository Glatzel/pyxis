Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $env:PROJ_LIB_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/lib
    $env:PROJ_INCLUDE_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/include
}
if ($IsMacOS) {
    $env:PROJ_LIB_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/lib
    $env:PROJ_INCLUDE_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/include
}
if ($IsLinux) {
    $env:PROJ_LIB_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib
    $env:PROJ_INCLUDE_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/include
}
