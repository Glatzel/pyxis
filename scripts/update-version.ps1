Set-Location $PSScriptRoot
Set-Location ..
$version = (Get-Content -Path "VERSION").Trim()

# Update the version in CPP
$cmakeListsPath = "./src/cpp/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCppProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated CPP version to $version"

# Update the version in CPP
$cmakeListsPath = "./src/cuda/CMakeLists.txt"
(Get-Content -Path $cmakeListsPath) -replace 'project\(.* VERSION .*', "project(PyxisCudaProject VERSION $version)" | Set-Content -Path $cmakeListsPath
Write-Host "Updated CUDA version to $version"

# Update the version in Cargo.toml
$cargoTomlPath = "./src/rust/Cargo.toml"
(Get-Content -Path $cargoTomlPath) -replace '^version = .*', "version = `"$version`"" | Set-Content -Path $cargoTomlPath
Write-Host "Updated Rust version to $version"

# Update the version in rattler
foreach ($pkg in ("pyxis-cli","pyxis-py")) {
    $recipe_path = "./src/rattler/$pkg/recipe.yaml"
    (Get-Content -Path $recipe_path) -replace '^  version: .*', "  version: $version" | Set-Content -Path $recipe_path
    Write-Host "Updated ratter $pkg version to $version"
}
