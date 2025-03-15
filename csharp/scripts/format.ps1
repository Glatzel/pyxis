$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
dotnet csharpier .
dotnet csharpier . --check
Set-Location $ROOT
