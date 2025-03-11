param (
    [ValidateSet("debug","--profile dist")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
& $PSScriptRoot/set-env.ps1
Write-Host "Build in $config mode."

# clean dist folder
Remove-Item ../../dist/cli -Recurse -ErrorAction SilentlyContinue
Remove-Item ../../dist/pyxis*.7z -Recurse -Force -ErrorAction SilentlyContinue

if ($IsWindows) {
    # build static##############################################################
    Write-Output "::group::Build static"
    # build
    & $PSScriptRoot/set-env.ps1 -link static
    cargo build $config -p pyxis-cli --features static


    # copy build file to dist
    New-Item ../../dist/cli/static -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis.exe" ../../dist/cli/static/pyxis.exe
    Write-Output "::endgroup::"

    # pack
    Write-Output "::group::Pack pyxis-windows-x64-self-contained.7z"
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cli-windows-x64-self-contained.7z" "../../dist/cli/static/pyxis.exe"
    Write-Output "::endgroup::"

    # build dynamic#############################################################
    # build
    Write-Output "::group::Build dynamic"
    & $PSScriptRoot/set-env.ps1 -link dynamic
    cargo build $config -p pyxis-cli

    # copy build file to dist
    New-Item ../../dist/cli/dynamic -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis.exe" ../../dist/cli/dynamic/pyxis.exe
    Write-Output "::endgroup::"

    # pack dynamic without dependency dll
    Write-Output "::group::Pack pyxis-windows-x64.7z"
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cli-windows-x64.7z" "../../dist/cli/dynamic/pyxis.exe"
    Write-Output "::endgroup::"

    # copy dependency dll to dist
    Write-Output "::group::Pack pyxis-windows-x64-proj.7z.7z"
    Copy-Item ../../vcpkg_deps/vcpkg_installed/dynamic/x64-windows/bin/*.dll ../../dist/cli/dynamic
    Copy-Item ../../vcpkg_deps/vcpkg_installed/dynamic/x64-windows/share/proj/proj.db ../../dist/cli/dynamic

    # pack dynamic with dependency dll
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cli-windows-x64-proj.7z" "../../dist/cli/dynamic/*"
    Write-Output "::endgroup::"
}
elseif ($IsLinux) {
    # build static##############################################################
    # build
    Write-Output "::group::Build static"
    & $PSScriptRoot/set-env.ps1 -link static
    cargo build $config -p pyxis-cli --features static

    #copy to dist
    New-Item ../../dist/cli/static -ItemType Directory -ErrorAction SilentlyContinue
    Copy-Item "target/$config/pyxis" ../../dist/cli/static
    Write-Output "::endgroup::"

    # pack
    Write-Output "::group::Pack pyxis-linux-x64-self-contained.7z"
    7z a -t7z -m0=LZMA2 -mmt=on -mx9 -md=4096m -mfb=273 -ms=on -mqs=on `
        "../../dist/pyxis-cli-linux-x64-self-contained.7z" "../../dist/cli/static/*"
    Write-Output "::endgroup::"
}
else {
    Write-Error "Unsupported system $os"
    exit 1
}
Set-Location $PSScriptRoot
Set-Location ../../../
