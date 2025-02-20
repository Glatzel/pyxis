Set-Location $PSScriptRoot
Set-Location ..
if($env:CI){
    cargo doc --no-deps --all-features --package geotool-algorithm
}
else{
    & $PSScriptRoot/set-env.ps1
    pixi run cargo doc --no-deps --all-features --package geotool-algorithm
}

Remove-Item ./dist/rust-doc.zip -Force -ErrorAction SilentlyContinue
New-Item ./dist -ItemType Directory -ErrorAction SilentlyContinue
Compress-Archive ./target/doc "./dist/rust-doc.zip"
