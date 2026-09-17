Set-Location $PSScriptRoot/..
$version = "0.0.46"

# rust
$cargoTomlPath = "./rust/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated rust version to $version"
Set-Location rust
cargo update
pixi update
Set-Location $PSScriptRoot/..

# python
$cargoTomlPath = "./python/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated python version to $version"
Set-Location python
cargo update
pixi update
Set-Location $PSScriptRoot/..

# gnss-receiver
$cargoTomlPath = "./tools/gnss-receiver/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated gnss-receiver version to $version"
Set-Location tools/gnss-receiver
cargo update
pixi update
Set-Location $PSScriptRoot/..

# pyxis-cli
$cargoTomlPath = "./tools/pyxis-cli/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated pyxis-cli version to $version"
Set-Location tools/pyxis-cli
cargo update
pixi update
Set-Location $PSScriptRoot/..
