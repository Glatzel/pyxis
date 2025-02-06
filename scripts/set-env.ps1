$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..
$pkg_config = Resolve-Path .pixi/envs/dev/Library/bin
$env:PATH=$env:PATH+";$pkg_config"
$env:PKG_CONFIG_PATH=Resolve-Path vcpkg_deps/vcpkg_installed/x64-windows/lib/pkgconfig
Set-Location $current_dir
