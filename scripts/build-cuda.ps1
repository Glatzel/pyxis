param (
    [ValidateSet("dist", "release", "debug")]
    [string]$config = "debug"
)

Set-Location $PSScriptRoot
Set-Location ..
if ($config -ne "debug") {
    pixi run cargo build --profile $config -p pyxis-cuda
}
else {
    pixi run cargo build -p pyxis-cuda
}
