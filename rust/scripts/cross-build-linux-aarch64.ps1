$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
if ($IsLinux) {
  Set-Location $PSScriptRoot/..
  sudo apt-get update
  sudo apt-get install -y g++-aarch64-linux-gnu
  rustup target add aarch64-unknown-linux-gnu
  pixi global install proj

  # Set PKG_CONFIG_PATH to vcpkg's pkgconfig directory
  $p = resolve-path ~/.pixi/envs/proj/proj/arm64-linux-release/lib/pkgconfig
  $env:PKG_CONFIG_PATH = "$p" + ":" + "$env:PKG_CONFIG_PATH"
  $env:PKG_CONFIG_ALLOW_CROSS = 1
  cargo build --target aarch64-unknown-linux-gnu --all-features
}
else { exit 1 }
