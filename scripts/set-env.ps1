param (
    [ValidateSet("static", "dynamic")]
    [string]$link = "static"
)
if ($IsWindows) {
    $current_dir = Resolve-Path $PWD
    Set-Location $PSScriptRoot
    Set-Location ..
    if ($link -eq 'static') {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/static/x64-windows-static/lib/pkgconfig
    }
    else {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/dynamic/x64-windows/lib/pkgconfig
    }
    Set-Location $current_dir
    Copy-Item ./vcpkg_deps/vcpkg_installed/static/x64-windows-static/share/proj/proj.db ./crates/geotool-cli/src
}
if ($IsLinux) {
    $current_dir = Resolve-Path $PWD
    Set-Location $PSScriptRoot
    Set-Location ..

    if ($link -eq 'static') {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/static/x64-linux-static/lib/pkgconfig
    }
    else {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/dynamic/x64-linux/lib/pkgconfig
    }
    Set-Location $current_dir
    Copy-Item ./vcpkg_deps/vcpkg_installed/static/x64-linux-static/share/proj/proj.db ./crates/geotool-cli/src
}