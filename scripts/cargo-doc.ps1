Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1

pixi run cargo doc --no-deps --all

Remove-Item ./dist/rust-doc-*.zip -Recurse -Force -ErrorAction SilentlyContinue
New-Item ./dist -ItemType Directory -ErrorAction SilentlyContinue
$version=cargo metadata --format-version=1 --no-deps | jq '.packages[0].version'
$version="$version".Replace("""","")
Compress-Archive ./target/doc "./dist/rust-doc-$version.zip"