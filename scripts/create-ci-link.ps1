Set-Location $PSScriptRoot/..
foreach ($lang in ("cpp", "csharp", "cuda", "python", "rust")) {
    New-Item -ItemType SymbolicLink -Path "./.github/workflows/$lang-ci.yml" -Target "../../$lang/.github/workflows/ci.yml"
}