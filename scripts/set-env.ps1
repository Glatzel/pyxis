$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..

$env:PKG_CONFIG_PATH=Resolve-Path vcpkg_deps/vcpkg_installed/x64-windows-static/lib/pkgconfig
Set-Location $current_dir
