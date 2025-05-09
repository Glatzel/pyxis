param ([switch]$update)
Set-Location $PSScriptRoot/..
if ($update) {
    aria2c -c -x16 -s16 -Z -d ./pre-commit `
        "https://raw.githubusercontent.com/Glatzel/template/main/pre-commit/pixi.toml" `
        "https://raw.githubusercontent.com/Glatzel/template/main/pre-commit/pixi.lock"
}
pixi install --manifest-path ./pre-commit/pixi.toml
