Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $bin = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $env:Path = "$bin" + ";" + "$env:Path"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig
}
if ($IsMacOS) {
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/lib/pkgconfig
}
Write-Output $(uname -m)
if ($IsLinux -and ($(uname -m) -contains 'x64' )) {
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
}
if ($IsLinux -and ($(uname -m) -contains 'ARM64' )) {
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-linux-release/lib/pkgconfig
}
