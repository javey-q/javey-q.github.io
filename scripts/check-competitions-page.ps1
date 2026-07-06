param(
  [string]$PublicDir = "public"
)

$ErrorActionPreference = "Stop"

$page = Join-Path $PublicDir "competitions\index.html"
if (-not (Test-Path -LiteralPath $page)) {
  throw "Missing generated competitions page: $page"
}

$html = Get-Content -Raw -LiteralPath $page
$requiredText = @(
  "比赛记录",
  "昇腾 AI 创新大赛",
  "航天宏图杯",
  "国丰东方慧眼杯",
  "CCF 大数据与计算智能大赛",
  "华为软件精英挑战赛",
  "源码 / 经验分享",
  "待补充"
)

foreach ($text in $requiredText) {
  if ($html -notlike "*$text*") {
    throw "Missing expected text in competitions page: $text"
  }
}

$requiredImages = @(
  "2024-ascend-mindspore-gold.png",
  "2024-ascend-mindspore-silver.png",
  "2022-rsipac-road.jpg",
  "2023-rsipac-decloud.png",
  "2024-huawei-codecraft.png",
  "2023-mindspore-ai-painter.png",
  "2020-future-cup.jpg",
  "2020-shandong-ai-mask.jpg"
)

foreach ($image in $requiredImages) {
  $path = Join-Path "static\images\competitions" $image
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Missing competition image asset: $path"
  }
}

Write-Host "Competitions page smoke test passed."

