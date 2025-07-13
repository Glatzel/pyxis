Set-Location $PSScriptRoot/..
$version = "0.0.27"

# cpp
$cmakeListsPath = "./cpp/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCppProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated cpp version to $version"


# csharp
$csconfig = "./csharp/Directory.Build.props"
(Get-Content -Path $csconfig) -replace '<Version>.*</Version>', "<Version>$version</Version>" | Set-Content -Path $csconfig
Write-Host "Updated csharp version to $version"

# cuda
$cmakeListsPath = "./cuda/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCppProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated cuda version to $version"

# python
$cargoTomlPath = "./python/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated python version to $version"

$recipe_path = "./python/rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated python rattler version to $version"

# python-cuda
$recipe_path = "./python-cuda/rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated python-cuda rattler version to $version"

# rust
$cargoTomlPath = "./rust/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated rust version to $version"

# rust-cuda
$cargoTomlPath = "./rust-cuda/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated rust-cuda version to $version"

# rattler
$recipe_path = "./rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated rust ratter version to $version"
