Set-Location $PSScriptRoot/..
$version = "0.0.43"

# rust
$cargoTomlPath = "./rust/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated rust version to $version"
Set-Location rust
cargo update
Set-Location $PSScriptRoot/..

# python
$cargoTomlPath = "./python/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated python version to $version"
Set-Location python
cargo update
Set-Location $PSScriptRoot/..
