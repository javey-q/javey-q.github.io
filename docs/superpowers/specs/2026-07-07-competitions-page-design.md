# Competitions Page Design

## Goal

Add a new blog page that organizes the competition records from `比赛记录.md` into a polished Hugo/PaperMod page. The page should present the author's competition history as a chronological timeline, enrich each entry with concise public context where available, include certificate or result images, and reserve a future link for source code or experience-sharing posts.

## Chosen Approach

Use a year-grouped timeline with compact competition cards.

This matches the selected visual option B. The structure keeps the timeline readable when multiple competitions happen in the same year, avoids an oversized landing-page style, and works well on mobile. Each year becomes a visible timeline section. Each competition is a card with a stable internal layout.

## Page Structure

The new page will live as a Hugo standalone content page at `content/competitions.md`, with front matter that adds it to the main menu.

Top-level content:

- Page title: `比赛记录`
- Short intro: one sentence summarizing the page as a record of algorithm, AI infrastructure, remote sensing, and deployment competitions.
- Timeline content grouped by year, sorted newest to oldest unless the source record strongly implies another order.

Each competition card will include:

- Competition name
- Award or rank
- Track or problem title
- Technical direction tags, such as remote sensing, image segmentation, deployment optimization, scheduling, MindSpore, LoRA
- Key strategy summary from the original record
- Public context supplement gathered from official or reliable public pages where available
- Certificate/result images copied into the site
- A reserved link area labeled as source code or experience sharing, initially disabled or marked `待补充`

## Assets

Images referenced by `比赛记录.md` will be copied into `static/images/competitions/` using descriptive ASCII filenames. The page will use site paths such as `/images/competitions/2024-ascend-mindspore-gold.png`.

If an image path in the source record is unavailable, the entry will still render without the image and the missing file will be reported after implementation.

## Styling

Add scoped CSS to `assets/css/extended/custom.css` using competition-specific class names. The design should remain compatible with PaperMod light and dark themes.

Expected components:

- `.competition-page`
- `.competition-intro`
- `.competition-year`
- `.competition-timeline`
- `.competition-card`
- `.competition-meta`
- `.competition-tags`
- `.competition-gallery`
- `.competition-links`

The palette should use the site's existing teal accent sparingly, with neutral backgrounds and borders. Cards should use small radius and restrained spacing to match PaperMod rather than a marketing layout.

## Data Flow

The source of truth for the initial content is `比赛记录.md`. The implementation will manually translate it into structured Markdown/HTML blocks in the Hugo page. Public details found online will be summarized, not copied verbatim.

Future source-code or write-up links can be added by editing each card's reserved link block.

## Error Handling

Unavailable image files will not block page creation. The implementation will identify missing images and omit or leave a non-rendering placeholder only if needed.

Unverified public details will not be stated as facts. If no reliable source is found for a competition, the page will use the original record's wording and avoid extra claims.

## Testing

Verification will include:

- Build the Hugo site successfully.
- Confirm the new page is reachable through the main menu.
- Check that image paths resolve in the generated site.
- Inspect the page in a browser at desktop and mobile widths for readable spacing, no text overlap, and acceptable dark/light theme behavior.

## Out of Scope

- Creating full experience-sharing articles for each competition.
- Uploading competition source code.
- Rewriting existing posts or unrelated pages.
- Changing the theme internals unless necessary.
