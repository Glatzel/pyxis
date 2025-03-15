$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
Remove-Item ./bin -Recurse -ErrorAction SilentlyContinue
dotnet build --configuration Release
Set-Location $ROOT
