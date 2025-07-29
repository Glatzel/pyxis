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
    sudo apt install -y libudev-dev libc6-dev
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig
}
elseif ($IsLinux -and ($(uname -m) -eq 'aarch64' )) {
    sudo apt install -y libudev-dev libc6-dev
    $env:PKG_CONFIG_PATH = Resolve-Path $PSScriptRoot/../.pixi/envs/default/proj/arm64-linux-release/lib/pkgconfig
}
