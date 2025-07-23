$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
&$PSScriptRoot/setup.ps1

$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
$env:RUSTDOCFLAGS = "--html-in-header katex.html -Dwarnings"
cargo doc --no-deps --all-features -p pyxis -p proj

Remove-Item ./dist/rust-doc.7z -Force -ErrorAction SilentlyContinue
New-Item ./dist -ItemType Directory -ErrorAction SilentlyContinue
7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
    "./dist/rust-doc.7z" "./target/doc/*"
Set-Location $ROOT
