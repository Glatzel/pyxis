Set-Location $PSScriptRoot
Set-Location ..

$tag = Read-Host "Please enter tag"
pixi run -e dev git-cliff --tag $tag > changelog.md