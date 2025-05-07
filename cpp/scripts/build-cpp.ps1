param($config)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

# set cmake taget config
if ($config) { $config = "-DCMAKE_BUILD_TYPE=$config" }

# create install dir
$install = "./dist"
Remove-Item $install -Recurse -ErrorAction SilentlyContinue
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = "-DCMAKE_INSTALL_PREFIX=$install"

# build
cmake -B build $install $config -DBUILD_CPP=ON
cmake --build build --target install

# pack output files
if ($IsWindows) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cpp-windows-x64.7z" "./dist/"
}
if ($IsLinux) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cpp-linux-x64.7z" "./dist/"
}
if ($IsMacOS) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cpp-macos-arm64.7z" "./dist/"
}
Set-Location $ROOT
