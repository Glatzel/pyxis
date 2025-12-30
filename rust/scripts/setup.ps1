Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $bin = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $env:Path = "$bin" + ";" + "$env:Path"
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig
}
elseif ($IsMacOS) {
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-osx-release/lib/pkgconfig
}
elseif ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
    $env:PKG_CONFIG_PATH = "/usr/lib/x86_64-linux-gnu/pkgconfig`:${env:PKG_CONFIG_PATH}"
}
elseif ($IsLinux -and ($(uname -m) -eq 'aarch64' )) {
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-linux-release/lib/pkgconfig
    $env:PKG_CONFIG_PATH = "/usr/lib/aarch64-linux-gnu/pkgconfig`:${env:PKG_CONFIG_PATH}"
}
