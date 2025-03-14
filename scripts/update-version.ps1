Set-Location $PSScriptRoot/..
$version= "0.0.14"

# Update cpp version
$cmakeListsPath = "./cpp/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCppProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated CPP version to $version"

# Update cuda version
$cmakeListsPath = "./cuda/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCppProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated cuda version to $version"

# Update python version
$cargoTomlPath = "./python/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated python version to $version"

# Update the version in Cargo.toml
$cargoTomlPath = "./rust/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated Rust version to $version"
