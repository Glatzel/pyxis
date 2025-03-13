param($config)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $ROOT
if ($config) { $config = "-DCMAKE_BUILD_TYPE=Release" }

# create install dir
$install = "$ROOT/dist/pyxis-cuda"
Remove-Item $install -Recurse -ErrorAction SilentlyContinue
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = Resolve-Path $install
$install = "$install".Replace('\', '/')
$install = "-DCMAKE_INSTALL_PREFIX=$install"

New-Item $ROOT/build/ptx -ItemType Directory -ErrorAction SilentlyContinue

# build
cmake -B build $install $config -DBUILD_CUDA=ON
cmake --build build --target install

# pack output files
if ($IsWindows) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "$ROOT/dist/pyxis-cuda-windows-x64.7z" "$ROOT/dist/pyxis-cuda/"
}if ($IsLinux) {
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "$ROOT/dist/pyxis-cuda-linux-x64.7z" "$ROOT/dist/pyxis-cuda/"
}

Set-Location $ROOT
