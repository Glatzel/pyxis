param (
    [ValidateSet("static", "dynamic")]
    [string]$link = "static"
)
$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..
pixi install
if ($IsWindows) {
    if ($link -eq 'static') {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/static/x64-windows-static/lib/pkgconfig
    }
    else {
        $env:PKG_CONFIG_PATH = Resolve-Path vcpkg_deps/vcpkg_installed/dynamic/x64-windows/lib/pkgconfig
    }
    Copy-Item .pixi/envs/default/Library/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
if ($IsLinux) {
    pixi install
    Copy-Item .pixi/envs/default/share/proj/proj.db ./crates/pyxis-cli/src/proj.db
}
Set-Location $current_dir
