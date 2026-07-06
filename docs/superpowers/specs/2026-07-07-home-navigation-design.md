# Home Navigation Design

## Goal

Optimize the current PaperMod profile homepage so the central action buttons better match the blog's content, while keeping the competitions page visible in the top navigation.

## Approved Design

Homepage profile buttons become:

- `博客` -> `/posts`
- `比赛` -> `/competitions/`
- `关于` -> `/about`
- `GitHub` -> `https://github.com/javey-q`

Top navigation keeps `比赛` and adds a new `博客` item. The intended order is:

- `归档`
- `博客`
- `比赛`
- `分类`
- `搜索`
- `标签`

The homepage profile copy changes from generic welcome text to:

- Title: `Javey 的技术博客`
- Subtitle: `AI Infra · 模型推理优化 · 算法竞赛实践`

## Files

- Modify `hugo.toml` for PaperMod profileMode buttons, title/subtitle, and menu entries.
- Add a small smoke test script to verify generated homepage/navigation HTML.

## Constraints

Do not modify `content/competitions.html`; it currently has local uncommitted changes that are outside this request.

## Verification

- Run `hugo`.
- Run homepage/navigation smoke test.
- Confirm generated homepage contains central buttons for `博客`, `比赛`, `关于`, `GitHub`.
- Confirm generated navigation contains `博客` and `比赛`.
