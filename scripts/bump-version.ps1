Set-Location $PSScriptRoot/..
$version = "0.0.39"

# python
$cargoTomlPath = "./python/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated python version to $version"

$recipe_path = "./python/rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated python rattler version to $version"

# rust
$cargoTomlPath = "./rust/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated rust version to $version"

$recipe_path = "./rust/rattler/recipe.yaml"
(Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
Write-Host "Updated rust ratter version to $version"
