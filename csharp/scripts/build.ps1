param (
    [ValidateSet("Debug", "Release")]
    $config = "Debug"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
Remove-Item ./bin -Recurse -ErrorAction SilentlyContinue
dotnet build --configuration $config
Set-Location $ROOT
