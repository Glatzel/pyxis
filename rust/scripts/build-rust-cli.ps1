param (
    [ValidateSet("develop", "release")]
    $config = "develop"
)
$ROOT = git rev-parse --show-toplevel
Set-Location $PSScriptRoot/..
& $PSScriptRoot/set-env.ps1

# clean dist folder
Remove-Item ./dist/cli -Recurse -ErrorAction SilentlyContinue
Remove-Item ./dist/pyxis-cli*.7z -Recurse -Force -ErrorAction SilentlyContinue

if ($IsWindows) {
    Write-Output "::group::Build static"
    # build
    & $PSScriptRoot/set-env.ps1
    cargo build --profile $config -p pyxis-cli

    # copy build file to dist
    New-Item ./dist/cli -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis.exe" ./dist/cli/pyxis.exe
    Write-Output "::endgroup::"

    # pack
    Write-Output "::group::Pack pyxis-windows-x64-self-contained.7z"
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cli-windows-x64.7z" "./dist/cli/*"
    Write-Output "::endgroup::"
}
elseif ($IsLinux) {
    # build
    Write-Output "::group::Build static"
    & $PSScriptRoot/set-env.ps1
    cargo build --profile $config -p pyxis-cli

    #copy to dist
    New-Item ./dist/cli -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis" ./dist/cli
    Write-Output "::endgroup::"

    # pack
    Write-Output "::group::Pack pyxis-linux-x64.7z"
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cli-linux-x64.7z" "./dist/cli/*"
    Write-Output "::endgroup::"
}
elseif ($IsMacOS) {
    # build
    Write-Output "::group::Build static"
    & $PSScriptRoot/set-env.ps1
    cargo build --profile $config -p pyxis-cli

    #copy to dist
    New-Item ./dist/cli -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis" ./dist/cli
    Write-Output "::endgroup::"

    # pack
    Write-Output "::group::Pack pyxis-cli-macos-arm64.7z"
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=512m -mfb=256 -ms=on -mqs=on `
        "./dist/pyxis-cli-macos-arm64.7z" "./dist/cli/*"
    Write-Output "::endgroup::"
}
else {
    Write-Error "Unsupported system $os"
    exit 1
}

Set-Location $ROOT
