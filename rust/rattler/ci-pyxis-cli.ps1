$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ROOT = git rev-parse --show-toplevel
& $PSScriptRoot/../scripts/build-rust-cli.ps1 -config release
Set-Location $PSScriptRoot/..
&./scripts/setup.ps1

if ($IsWindows -or $IsMacOS) {
    cargo build --release -p pyxis-cli
}
if ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    sudo apt install musl-tools
    rustup target add x86_64-unknown-linux-musl
    cargo build --release -p pyxis-cli --target x86_64-unknown-linux-musl
}
if ($IsLinux -and ($(uname -m) -eq 'x86_64' )) {
    sudo apt install musl-tools
    rustup target add aarch64-unknown-linux-musl
    cargo build --release -p pyxis-cli --target aarch64-unknown-linux-musl
}

Set-Location $PSScriptRoot..
Set-Location $ROOT
