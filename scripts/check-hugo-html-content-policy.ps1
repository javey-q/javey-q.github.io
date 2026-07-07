param(
  [string]$ConfigPath = "hugo.toml",
  [string]$ContentDir = "content"
)

$ErrorActionPreference = "Stop"

$htmlContentFiles = Get-ChildItem -LiteralPath $ContentDir -Recurse -File -Filter "*.html" -ErrorAction SilentlyContinue
if (-not $htmlContentFiles) {
  Write-Host "No HTML content files found."
  exit 0
}

$config = Get-Content -Raw -LiteralPath $ConfigPath
if ($config -notmatch '(?s)\[security\].*allowContent\s*=\s*\[[^\]]*(\^?text/html\$?)[^\]]*\]') {
  $files = ($htmlContentFiles | ForEach-Object { $_.FullName }) -join ", "
  throw "HTML content files require Hugo security.allowContent to include text/html. Files: $files"
}

Write-Host "Hugo HTML content policy check passed."
