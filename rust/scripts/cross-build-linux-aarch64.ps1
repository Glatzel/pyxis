$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
Set-Location $PSScriptRoot/..
  sudo apt-get update
 sudo  apt-get install --assume-yes curl git build-essential cmake pkg-config zip unzip tar qemu-user-static g++-aarch64-linux-gnu cmake ninja-build

  # Install vcpkg
  git clone https://github.com/microsoft/vcpkg.git /vcpkg
  cd /vcpkg && ./bootstrap-vcpkg.sh

  # Use vcpkg to install proj for aarch64
  cd /vcpkg && ./vcpkg install proj --triplet arm64-linux-release

  # Set PKG_CONFIG_PATH to vcpkg's pkgconfig directory
  export PKG_CONFIG_PATH=/vcpkg/installed/arm64-linux-release/lib/pkgconfig:$PKG_CONFIG_PATH

cargo build --target aarch64-unknown-linux-gnu --all-features
