$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
rustup target add aarch64-unknown-linux-gnu
Set-Location $PSScriptRoot/..
sudo apt-get update
sudo apt-get install --assume-yes curl git build-essential cmake pkg-config zip unzip tar qemu-user-static g++-aarch64-linux-gnu cmake ninja-build

# Install vcpkg
git clone https://github.com/microsoft/vcpkg.git
Set-Location ./vcpkg && ./bootstrap-vcpkg.sh

# Use vcpkg to install proj for aarch64
./vcpkg install proj --triplet arm64-linux-release

# Set PKG_CONFIG_PATH to vcpkg's pkgconfig directory
$p = resolve-path ./installed/arm64-linux-release/lib/pkgconfig
$env:PKG_CONFIG_PATH = "$p" + ":" + "$env:PKG_CONFIG_PATH"
$env:PKG_CONFIG_ALLOW_CROSS = 1
Set-Location ..
cargo build --target aarch64-unknown-linux-gnu --all-features
