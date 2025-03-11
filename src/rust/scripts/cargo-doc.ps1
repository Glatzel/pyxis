$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1
if($env:CI){
    cargo doc --no-deps --all-features --package pyxis
}
else{
    cargo doc --no-deps --all-features --package pyxis
}

Remove-Item $ROOT/dist/rust-doc.7z -Force -ErrorAction SilentlyContinue
New-Item $ROOT/dist -ItemType Directory -ErrorAction SilentlyContinue
7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on -sccUTF-8 -bb0 -bse0 -bsp2 "-wtarget/doc" -mtc=on -mta=on "$ROOT/dist/rust-doc.7z" "./target/doc/*"
Set-Location $PSScriptRoot
Set-Location $ROOT
