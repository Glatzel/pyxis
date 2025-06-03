$env:CI = $true
&$PSScriptRoot/cargo-test.ps1
Set-Location $PSScriptRoot/..
Copy-Item ./target/nextest/default/nextest.junit.xml ./
