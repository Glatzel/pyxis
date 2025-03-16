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
    # build static##############################################################
    Write-Output "::group::Build static"
    # build
    & $PSScriptRoot/set-env.ps1 -link static
    cargo install --profile $config -p pyxis-cli --features static --root ./dist/cli/static

    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cli-windows-x64-self-contained.7z" "./dist/cli/static/pyxis.exe"
    Write-Output "::endgroup::"

    # build dynamic#############################################################
    # build
    Write-Output "::group::Build dynamic"
    & $PSScriptRoot/set-env.ps1 -link dynamic
    cargo install --profile $config -p pyxis-cli --root ./dist/cli/dynamic/pyxis.exe

    # pack dynamic without dependency dll
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cli-windows-x64.7z" "./dist/cli/dynamic/pyxis.exe"

    # copy dependency dll to dist
    Copy-Item ./vcpkg/installed/dynamic/x64-windows/bin/*.dll ./dist/cli/dynamic
    Copy-Item ./vcpkg/installed/dynamic/x64-windows/share/proj/proj.db ./dist/cli/dynamic

    # pack dynamic with dependency dll
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cli-windows-x64-proj.7z" "./dist/cli/dynamic/*"
    Write-Output "::endgroup::"
}
elseif ($IsLinux) {
    # build static##############################################################
    # build
    Write-Output "::group::Build static"
    & $PSScriptRoot/set-env.ps1 -link static
    cargo install --profile $config -p pyxis-cli --features static --root ./dist/cli/static

    # pack
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "./dist/pyxis-cli-linux-x64-self-contained.7z" "./dist/cli/static/*"
    Write-Output "::endgroup::"
}
else {
    Write-Error "Unsupported system $os"
    exit 1
}

Set-Location $ROOT
