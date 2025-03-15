Set-Location $PSScriptRoot/..
foreach ($lang in ("cpp", "csharp", "cuda", "python", "rust")) {
    copy-Item "./$lang/.github/workflows/ci.yml" "./.github/workflows/$lang-ci.yml"
}
