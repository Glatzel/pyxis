param (
    [ValidateSet("static", "dynamic")]
    [string]$link = "static"
)
$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..
if ($IsWindows) {
    if ($link -eq 'static') {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/static/x64-windows-static/lib/pkgconfig
    }
    else {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/dynamic/x64-windows/lib/pkgconfig
    }
    Copy-Item ./vcpkg_deps/vcpkg_installed/static/x64-windows-static/share/proj/proj.db ./crates/geotool-cli/src
}
if ($IsLinux) {
    Copy-Item .pixi/envs/default/Lib/share/proj/proj.db ./crates/geotool-cli/src
}
Set-Location $current_dir