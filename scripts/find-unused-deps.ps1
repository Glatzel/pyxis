$current_dir=Resolve-Path $PWD
Set-Location $PSScriptRoot
Set-Location ..
$ErrorActionPreference = 'Stop'

try{cargo machete --with-metadata}
catch{
    cargo install cargo-machete
    cargo machete --with-metadata
}

Set-Location $current_dir