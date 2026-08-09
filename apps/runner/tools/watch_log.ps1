param(
    [Parameter(Mandatory, Position = 0)]
    [string]$Path,

    [Parameter(Mandatory, Position = 1)]
    [ValidateNotNullOrEmpty()]
    [string]$Keyword
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "Log file does not exist: $Path"
}

Get-Content -LiteralPath $Path -Encoding UTF8 -Wait |
    Select-String -SimpleMatch -Pattern $Keyword |
    ForEach-Object {
        [Console]::Beep(1000, 300)
        $_.Line
    }
