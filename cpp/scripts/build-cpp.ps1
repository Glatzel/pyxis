param($config)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

# set cmake target config
if ($config) { $config = "-DCMAKE_BUILD_TYPE=$config" }

# create install dir
$install = "./dist"
Remove-Item $install -Recurse -ErrorAction SilentlyContinue
New-Item $install -ItemType Directory -ErrorAction SilentlyContinue
$install = "-DCMAKE_INSTALL_PREFIX=$install"

# build
cmake -B build $install $config -DBUILD_CPP=ON
cmake --build build --target install

Set-Location $ROOT