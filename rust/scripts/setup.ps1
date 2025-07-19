Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $env:LIB_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/lib
    $env:INCLUDE_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/include
}
if ($IsMacOS) {
    $env:LIB_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/lib
    $env:INCLUDE_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/include
}
if ($IsLinux) {
    $env:LIB_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib
    $env:INCLUDE_DIR = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/include
}
