param (
    [ValidateSet("develop", "release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..

# clean dist folder
Remove-Item ./dist/cli -Recurse -ErrorAction SilentlyContinue
Remove-Item ./dist/pyxis-cli*.7z -Recurse -Force -ErrorAction SilentlyContinue

if ($IsWindows) {
    Write-Output "::group::Build static"
    # build
    cargo build --profile $config -p pyxis-cli

    # copy build file to dist
    New-Item ./dist/cli -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis.exe" ./dist/cli
    Write-Output "::endgroup::"
}
elseif ($IsLinux) {
    # build
    Write-Output "::group::Build static"
    cargo build --profile $config -p pyxis-cli

    #copy to dist
    New-Item ./dist/cli -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis" ./dist/cli
    Write-Output "::endgroup::"
}
elseif ($IsMacOS) {
    # build
    Write-Output "::group::Build static"
    cargo build --profile $config -p pyxis-cli

    #copy to dist
    New-Item ./dist/cli -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis" ./dist/cli
    Write-Output "::endgroup::"
}
else {
    Write-Error "Unsupported system $os"
    exit 1
}

Set-Location $ROOT
