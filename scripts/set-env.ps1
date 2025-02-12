$current_dir = Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..
try{
    $pkg_config = Resolve-Path .pixi/envs/default/Library/bin
}catch{
    try {
        $pkg_config = Resolve-Path .pixi/envs/ci-cli/Library/bin
    }
    catch {
        $pkg_config = Resolve-Path .pixi/envs/ci-py/Library/bin
    }
}
Write-Output "pkg_config_execute_path: $pkg_config"
$dll_path = Resolve-Path vcpkg_deps/vcpkg_installed/x64-windows/bin
$env:PATH = $env:PATH + ";$pkg_config;$dll_path"
$env:PKG_CONFIG_PATH=Resolve-Path vcpkg_deps/vcpkg_installed/x64-windows/lib/pkgconfig
Set-Location $current_dir
