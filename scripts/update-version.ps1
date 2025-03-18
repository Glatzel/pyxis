Set-Location $PSScriptRoot/..
$version= "0.0.16"

# Update cpp version
$cmakeListsPath = "./cpp/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCppProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated CPP version to $version"


# Update csharp version
$csconfig = "./csharp/Directory.Build.props"
(Get-Content -Path $csconfig) -replace '<Version>.*</Version>', "<Version>$version</Version>" | Set-Content -Path $csconfig
Write-Host "Updated csharp version to $version"

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

# Update python rattler version
$recipe_path = "./python/rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated ratter cli version to $version"

# Update rust rattler version
$recipe_path = "./rust/rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated ratter py version to $version"
