param(
    $config,
    $install = "../../dist/pyxis-cpp"
)

if ($config) {
    $config = "-DCMAKE_BUILD_TYPE=Release"
}
if ($install) {
    if (-not (Test-Path $install)) {
        New-Item $install -ItemType Directory
    }
    $install = Resolve-Path $install
    $install = "$install".Replace('\', '/')
    $install = "-DCMAKE_INSTALL_PREFIX=$install"
}
Set-Location $PSScriptRoot
Set-Location ..

cmake . -B build $install $config
if ($install) {
    cmake --build build --target install
}
else {
    cmake --build build
}
Set-Location $PSScriptRoot
Set-Location ..
if ($IsWindows) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cpp-windows-x64.7z" "../../dist/pyxis-cpp/"
}if ($IsLinux) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cpp-linux-x64.7z" "../../dist/pyxis-cpp/"
}
