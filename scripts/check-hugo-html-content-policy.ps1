param(
  [string]$ConfigPath = "hugo.toml",
  [string]$ContentDir = "content"
)

$ErrorActionPreference = "Stop"

$contentFiles = Get-ChildItem -LiteralPath $ContentDir -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Extension -in @(".html", ".htm", ".md", ".markdown", ".mdown") }

if (-not $contentFiles) {
  Write-Host "No checked content files found."
  exit 0
}

$config = Get-Content -Raw -LiteralPath $ConfigPath
$requiredMediaTypes = [ordered]@{}
foreach ($file in $contentFiles) {
  if ($file.Extension -in @(".html", ".htm")) {
    $requiredMediaTypes["text/html"] = $true
  }
  if ($file.Extension -in @(".md", ".markdown", ".mdown")) {
    $requiredMediaTypes["text/markdown"] = $true
  }
}

foreach ($mediaType in $requiredMediaTypes.Keys) {
  $escaped = [regex]::Escape($mediaType)
  if ($config -notmatch "(?s)\[security\].*allowContent\s*=\s*\[[^\]]*$escaped[^\]]*\]") {
    $files = ($contentFiles | ForEach-Object { $_.FullName }) -join ", "
    throw "Content files require Hugo security.allowContent to include $mediaType. Files: $files"
  }
}

Write-Host "Hugo content policy check passed."
