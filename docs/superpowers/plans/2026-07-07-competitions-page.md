# Competitions Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Hugo standalone competition timeline page from `比赛记录.md`, with year-grouped cards, copied certificate images, public context supplements, and future link placeholders.

**Architecture:** The feature is a static Hugo page at `content/competitions.html`, styled by scoped rules in `assets/css/extended/custom.css`. Images are copied into `static/images/competitions/` and referenced by stable site paths. A PowerShell smoke test validates generated HTML after `hugo` builds the site.

**Tech Stack:** Hugo, PaperMod, Markdown with inline HTML, PowerShell smoke test, static image assets.

---

### Task 1: Add A Failing Smoke Test

**Files:**
- Create: `scripts/check-competitions-page.ps1`
- Test: `scripts/check-competitions-page.ps1`

- [ ] **Step 1: Create the smoke test script**

```powershell
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `powershell -ExecutionPolicy Bypass -File scripts/check-competitions-page.ps1`

Expected: FAIL with `Missing generated competitions page`.

### Task 2: Copy Competition Images

**Files:**
- Create directory: `static/images/competitions/`
- Create image files under `static/images/competitions/`

- [ ] **Step 1: Copy available source images**

Copy these exact files:

```text
D:\Data\ME\材料\比赛\5700EB786BFE51A7CC195541425DD620.png -> static/images/competitions/2024-ascend-mindspore-gold.png
D:\Data\ME\材料\比赛\5F777E1C4B6FE63799EB4AB08CCD0B46.png -> static/images/competitions/2024-ascend-mindspore-silver.png
D:\Data\ME\材料\微信图片_20230713191036.jpg -> static/images/competitions/2022-rsipac-road.jpg
D:\Data\ME\材料\比赛\5803BD75B4250B1BE1326D22F6EDF278.png -> static/images/competitions/2023-rsipac-decloud.png
D:\Data\ME\材料\比赛\082655D8DFF4BC115BC545283B8FFF64.png -> static/images/competitions/2024-huawei-codecraft.png
D:\Data\ME\材料\比赛\98EF6AADB84B8820B86FB50820B29A85.png -> static/images/competitions/2023-mindspore-ai-painter.png
D:\Data\ME\材料\5.jpg -> static/images/competitions/2020-future-cup.jpg
D:\Data\ME\材料\4.jpg -> static/images/competitions/2020-shandong-ai-mask.jpg
```

- [ ] **Step 2: Report missing files**

Run: `Get-ChildItem -LiteralPath static/images/competitions`

Expected: all eight destination files are present.

### Task 3: Add The Hugo Page

**Files:**
- Create: `content/competitions.html`

- [ ] **Step 1: Create the page**

Create a standalone HTML content page with TOML front matter:

```toml
+++
date = '2026-07-07T00:00:00+08:00'
draft = false
title = '比赛记录'
description = '算法、AI Infra、遥感与模型部署相关比赛经历。'
[menu.main]
  name = '比赛'
  weight = 6
+++
```

Content requirements:

- Wrap the page body in `<div class="competition-page">`.
- Group entries by year in this order: 2024, 2023, 2022, 2020.
- Use `<section class="competition-year" id="competitions-YYYY">` for each year.
- Use `<article class="competition-card">` for each competition.
- Include image galleries using `<figure>` and site paths under `/images/competitions/`.
- Include a reserved link block containing `源码 / 经验分享` and `待补充`.
- Include public context source links in a small source list for entries that have reliable public pages.

### Task 4: Add Scoped Styling

**Files:**
- Modify: `assets/css/extended/custom.css`

- [ ] **Step 1: Append competition page styles**

Add scoped CSS for:

- `.competition-page`
- `.competition-intro`
- `.competition-year`
- `.competition-year-heading`
- `.competition-timeline`
- `.competition-card`
- `.competition-card-header`
- `.competition-award`
- `.competition-meta`
- `.competition-tags`
- `.competition-gallery`
- `.competition-links`
- `.competition-sources`

Expected styling:

- Cards have `border-radius: 8px`.
- Timeline uses a left border on desktop.
- Gallery uses responsive grid columns.
- Dark theme uses PaperMod variables and avoids hard-coded low-contrast colors.
- Mobile layout removes excessive left padding.

### Task 5: Build And Verify

**Files:**
- Generated: `public/`

- [ ] **Step 1: Build the site**

Run: `hugo`

Expected: build succeeds without errors.

- [ ] **Step 2: Run the smoke test**

Run: `powershell -ExecutionPolicy Bypass -File scripts/check-competitions-page.ps1`

Expected: PASS with `Competitions page smoke test passed.`

- [ ] **Step 3: Inspect changed files**

Run: `git diff -- content/competitions.md assets/css/extended/custom.css scripts/check-competitions-page.ps1`

Expected: diff only contains the competitions page, scoped styles, and smoke test.

### Task 6: Optional Browser Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Start Hugo server**

Run: `hugo server --bind 127.0.0.1 --port 1313`

Expected: server starts and `/competitions/` is reachable.

- [ ] **Step 2: Check desktop and mobile viewports**

Open `/competitions/` and verify:

- Main menu contains `比赛`.
- Cards are readable.
- Images render.
- No card text overlaps.
- Mobile layout remains single-column and readable.
