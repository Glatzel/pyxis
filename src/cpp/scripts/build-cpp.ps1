param($config)
Set-Location $PSScriptRoot
Set-Location ..
if ($config) {
    $config = "-DCMAKE_BUILD_TYPE=Release"
}
# create install dir
$install = "../../dist/pyxis-cpp"
Remove-Item $install -Recurse -ErrorAction SilentlyContinue
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = Resolve-Path $install
$install = "$install".Replace('\', '/')
$install = "-DCMAKE_INSTALL_PREFIX=$install"

cmake . -B build $install $config
if ($install) {
    cmake --build build --target install
}
else {
    cmake --build build
}
if ($IsWindows) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cpp-windows-x64.7z" "../../dist/pyxis-cpp/"
}if ($IsLinux) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cpp-linux-x64.7z" "../../dist/pyxis-cpp/"
}
Set-Location $PSScriptRoot
Set-Location ../../../