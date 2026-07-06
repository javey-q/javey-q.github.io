param(
  [string]$PublicDir = "public"
)

$ErrorActionPreference = "Stop"

$homePath = Join-Path $PublicDir "index.html"
if (-not (Test-Path -LiteralPath $homePath)) {
  throw "Missing generated home page: $homePath"
}

$html = Get-Content -Raw -LiteralPath $homePath

$requiredText = @(
  "Javey 的技术博客",
  "AI Infra · 模型推理优化 · 算法竞赛实践",
  "博客",
  "比赛",
  "关于",
  "GitHub"
)

foreach ($text in $requiredText) {
  if ($html -notlike "*$text*") {
    throw "Missing expected home/navigation text: $text"
  }
}

$requiredLinks = @(
  'href="/posts"',
  'href="/competitions/"',
  'href="/about"',
  'href="https://github.com/javey-q"'
)

foreach ($link in $requiredLinks) {
  if ($html -notlike "*$link*") {
    throw "Missing expected home/navigation link: $link"
  }
}

Write-Host "Home navigation smoke test passed."
