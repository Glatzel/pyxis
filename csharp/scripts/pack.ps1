param (
    [ValidateSet("Debug", "Release")]
    $config = "Debug"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
dotnet pack --configuration $config -o dist
Set-Location $ROOT
