$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..

$env:PKG_CONFIG_PATH=Resolve-Path vcpkg_deps/vcpkg_installed/static/x64-windows/lib/pkgconfig
Set-Location $current_dir
