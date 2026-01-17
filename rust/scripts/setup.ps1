Set-Location $PSScriptRoot/..
pixi install
if ($IsWindows) {
    $bin = Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/bin
    $env:Path = "$bin" + ";" + "$env:Path"
    $env:PKG_CONFIG_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/Library/lib/pkgconfig);${env:PKG_CONFIG_PATH}"
}
if ($IsMacOS) {
    $env:PKG_CONFIG_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib/pkgconfig)`:${env:PKG_CONFIG_PATH}"
}
if ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    $env:PKG_CONFIG_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib/pkgconfig)`:/usr/lib/x86_64-linux-gnu/pkgconfig`:${env:PKG_CONFIG_PATH}"
}
if ($IsLinux -and ($(uname -m) -eq 'aarch64' )) {
    $env:PKG_CONFIG_PATH = "$(Resolve-Path $PSScriptRoot/../.pixi/envs/default/lib/pkgconfig)`:/usr/lib/aarch64-linux-gnu/pkgconfig`:${env:PKG_CONFIG_PATH}"
}
