param($config)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

# set cmake taget config
if ($config) { $config = "-DCMAKE_BUILD_TYPE=$config" }

# create install dir
$install = "$ROOT/dist/pyxis-cpp"
Remove-Item $install -Recurse -ErrorAction SilentlyContinue
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = "-DCMAKE_INSTALL_PREFIX=$install"

# build
cmake . -B build $install $config
cmake --build build --target install

# pack output files
if ($IsWindows) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "$ROOT/dist/pyxis-cpp-windows-x64.7z" "$ROOT/dist/pyxis-cpp/"
}if ($IsLinux) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "$ROOT/dist/pyxis-cpp-linux-x64.7z" "$ROOT/dist/pyxis-cpp/"
}

Set-Location $ROOT
