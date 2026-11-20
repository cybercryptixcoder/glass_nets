# The Glass Networks theme (v2): how the lab should look, and how to keep it that way

*Version 2, Nov 20, 2026 (A50). It replaces v1 (A49, the "smoked glass" theme), which Shreyas judged too incremental. Code: `lab/static/theme.css`, `shell.html` and `shell.js`. Research and reasons: `UI_RESEARCH.md` (§7 covers v2).*

This file is for anyone (person or agent) who adds or changes a page. Follow it, and the lab stays one product. If a rule here is wrong, change it here first, then in the code, then run the sync.

## 0. The idea in one paragraph

A dark product dashboard in the Linear / Raycast / Vercel family, modelled on Shreyas's reference image (a "SalesOps" dashboard). Every page sits in one **app shell**:
- a left sidebar with the brand and the three pages;
- a header with the page title, its context and its tools;
- content made of **cards**, **KPI cards** (label, icon box, big number, delta) and **progress rows**.

Surfaces are zinc near-black with hairline borders. Type is Geist. **Green** is the one accent, used for the primary action, the selected item and anything good. **Cyan** is its partner for data and information. Orange, amber and red appear only to mean something. Real frosted glass is kept for things that float over content: the header, drawers, dialogs and tooltips.

## 1. How the theme is wired

The server serves only the three pages and is not to be touched, so shared code is **inlined**:

| Source (edit this) | Copied into each page between |
|---|---|
| `lab/static/theme.css` | `/* @theme:begin … */` and `/* @theme:end */` (in `<style>`) |
| `lab/static/shell.html` (the sidebar) | `<!-- @shell:begin … -->` and `<!-- @shell:end -->` (in `<body>`) |
| `lab/static/shell.js` (icons, sidebar rail) | `/* @shell-js:begin … */` and `/* @shell-js:end */` (top of `<script>`) |

- Run `python3 lab/static/sync_theme.py` after editing a source. `--check` exits 1 if any page is stale.
- **Never edit the copies.**
- **A page opts into the shell** with this structure:
  ```html
  <body data-page="catalog|lab|constructor" data-rail="1400">
  <div class="shell"><!-- @shell:begin --><!-- @shell:end -->
    <div class="main"><header class="hdr">…</header> …content… </div></div>
  ```
  - `data-page` lights up the sidebar item.
  - `data-rail` is the width below which the sidebar becomes a 68px icon rail: 1400 by default, 1680 for the lab and 1520 for the constructor, because they have three columns.
  - Users can also collapse the sidebar by hand. The choice is remembered (`gn-rail` in localStorage).
- **Fonts:**
  ```html
  <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400..600&display=swap" rel="stylesheet">
  ```
  Also set `<meta name="theme-color" content="#09090B">`.
- **Icons:** Lucide (ISC licence). They live as path data in `shell.js`.
  - In markup, write `<i data-icon="name"></i>`; it is replaced with the SVG on load. Never put an `id` on it.
  - In script templates, call `ic('name')`.
  - To add an icon, copy its inner SVG from `lucide-static` into `ICONS` in `shell.js`.
- **Old variable names** that page scripts still read are aliased at the top of each page's CSS: `--neg`, `--pos`, `--field`, `--ink`, `--serif`, `--mono`, `--thin`. The heatmap code reads `--neg`, `--pos` and `--field` as **hex**, so `--div-*` must stay hex.

## 2. Colour

**Surfaces** (zinc; far is darker, near is lighter):

| Token | Dark | Use |
|---|---|---|
| `--bg` | #09090B | page, sidebar, header |
| `--card` | #0F0F11 | cards, KPI cards, pane contents |
| `--card-2` | #141416 | inputs, icon boxes, hover |
| `--card-3` | #1A1A1D | selected rows, progress tracks |
| `--card-4` | #222226 | the raised thumb of a segmented control |
| `--sunk` | #0B0B0D | code, logs, the heatmap well |
| `--glass` / `--glass-strong` | 72% / 90% of #0F0F11 + blur 24 | header, drawer, dialog, tooltip |

**Lines:**
- `--border` (white 7.5%): every card and divider;
- `--border-2` (11%): controls;
- `--border-3` (18%): hover;
- `--edge`: the 1px inner top light on cards.

**Text:** `--text-1` #F4F4F5, `--text-2` #A1A1AA, `--text-3` #71717A, `--text-4` #52525B. Their roles: primary, secondary, labels and meta, disabled.

**Accents:**

| Token | Value | Job |
|---|---|---|
| `--accent` | #22C55E (`--accent-hi` #4ADE80) | the one primary button per view; the selected sidebar item, chip, tensor or tab; good status; positive deltas |
| `--cyan` | #06B6D4 | data series 1, information, "running", the second KPI tone |
| `--violet` | #8B5CF6 | a third data series only |
| `--orange` / `--amber` | #F97316 / #F59E0B | "lookup" labels, partial status, warnings |
| `--red` | #EF4444 | failure, negative deltas |
| `*-soft` | 10–12% | tinted backgrounds of badges, icon boxes and table cells |
| `--brand-grad` | green → cyan | the brand chip only |
| `--div-neg` ↔ `--div-mid` ↔ `--div-pos` | #E8601C ↔ #18181B ↔ #0E9FBF | heatmaps of weights; validated (colour-blind ΔE 21, both poles in the dark band) |

**Rules:**
- **Selection is green outline + green tint, everywhere** (sidebar, chips, tensors, the selected list row's 2px bar, the underline tab). Never invert to white.
- **Status always pairs colour with a word** (`.badge`, `.agree`, `.verdict`, `.stage`).
- **No other hues.** If a new kind of data needs a third or fourth series colour, take it from a validated categorical palette; never improvise one.

## 3. Type (Geist)

| Role | Size / weight | Notes |
|---|---|---|
| Page title (`.hdr-title`) | 20 / 600, −0.025em | one per page |
| Detail title (`.d-title`) | 24 / 600, −0.03em | the selected item |
| Card title (`.card-t`) | 16 / 600, −0.015em | with a 13px `--text-2` subtitle (`.card-s`) |
| KPI value (`.kpi-v`) | 32 / 600, −0.035em, tabular | 26–28 in compact KPI rows |
| KPI label / body | 14 / 400 | `--text-2` for labels |
| Meta, hints | 12–13 / 400 | `--text-3` |
| Numbers, ids, shapes, code | Geist Mono 12–13 | tabular |

**Rules:**
- Use weights 400, 500 and 600 only.
- Use sentence case; never uppercase labels.
- `text-wrap: balance` on headings and `pretty` on text.

## 4. Shape, space, depth

- **Radii:** 4 (tiny), 6 (badges, tags), 8 (controls, rows, icon boxes), 12 (cards), 16 (drawers, dialogs), pill (chips, switches, tracks).
- **Spacing:** a 4px grid.
  - Content padding 24px; cards 20–22px.
  - Gaps 16px between cards (12px in compact rows).
- **Sizes:**
  - header 64px; pane headers 56px;
  - sidebar items 42px; controls 36px (30px small).
- **Depth** comes from surface steps and hairlines.
  - Cards carry `--shadow-card`: a 1px inner top light and a 1–2px drop.
  - Floating layers carry `--shadow-float`: a 0.5px black ring, two soft drops and an inner top light.
  - Nothing has a big glow except the primary button and the selected sidebar item, which have a faint green one.

## 5. Layout patterns

- **App shell:** sidebar | main (header + content). The header holds:
  - the title;
  - a context line with an icon (what you are looking at);
  - tools on the right (search, wallet, actions, theme).
- **List and detail** (catalogue, lab specimens): a 280–320px **list pane** separated by a hairline, not a floating card.
  - Groups have small `--text-3` headings.
  - Rows are 14px/500 names with 12px meta.
  - The selected row is `--card-3` with a 2px green bar.
- **Panes** (lab's reader, the constructor's three columns): full-height columns separated by hairlines, each with a 56px header (an icon, a 14px/600 title, meta on the right).
- **Detail:**
  - a breadcrumb (`.crumb`), the title, a subtitle and mono tags, with the status badge on the right;
  - then a **KPI row**;
  - then **underline tabs**;
  - then cards.
- **KPI card** (`.kpi`):
  - the label top left and an icon box top right (`.kpi-ic`; `.g` green or `.c` cyan for the headline one);
  - the value, with a delta chip (`.kpi-d`, green up, red down);
  - a short sub-line. Keep sub-lines to about six words.
  - One row of KPIs: set `grid-template-columns: repeat(n, minmax(0,1fr))`, and compact them on narrow screens.
- **Progress rows** (`.prog`, `.prog-row`, `.track > i`): a label, a value on the right and a 6px bar. Use them for any list of shares; colour by meaning (green = the headline measure, cyan or violet = alternatives, grey = a baseline).
- **Empty states** (`.t-empty`): an icon box, a bold line and one sentence that says what to do.

## 6. Components (in theme.css)

- **`.btn`:** secondary by default (`--card-2`, `--border-2`).
  - `.primary` (legacy `.orange`): green with dark text and a soft green glow.
  - `.yellow`: green-tinted secondary.
  - `.ghost`; `.on`; `.sm`.
  - Icons go first: `ic('play') + 'Read it'`.
  - Every button scales to 0.97 when pressed.
- **`.chip`:** 28px pill filters and toggles; `.on` is a green outline and tint.
- **`.pill`:** a passive label with an icon (wallet, usage).
- **`.badge`:** status (`.ok`, `.warn`/`.part`, `.bad`/`.no`, `.info`/`.run`), optionally with `<span class="dot">`.
- **`.seg`:** a segmented control with a raised thumb.
- **`.utabs`:** underline tabs with a 2px green underline.
- **`.toggle`:** a switch, green when on.
- **Fields:** 36px, `--card-2`, focus = a green ring. Checkboxes are drawn (green when checked).
- **`.field-ic`:** a field with a leading icon (search).
- **`.icon-btn`:** a 36px square icon button.
- **`.card`, `.card-h`, `.card-t`, `.card-s`, `.card-b`, `.card-x`:** a card, its header, title, subtitle, body, and actions slot on the right.
- **`.kpis` / `.kpi`**, **`.prog`**, **`.t-empty`**, **`.glass`**.
- **Tables:** hairlines between rows, 12px `--text-3` headers, tabular numbers. Wide tables go in `.tw` (a bordered scroll box).

## 7. Motion

- **Durations:** 120ms for colour, 180ms for small moves, 280ms for entrances and the drawer.
- **Curves:** `--ease-out` `cubic-bezier(.23,1,.32,1)` for everything that enters; `--ease-drawer` for the drawer; never ease-in.
- **Entrances:** `t-rise` for panels and tab content; `t-pop` for dialogs and reveals; `t-pulse` for a running dot.
- Animate `transform` and `opacity` only, and list the properties (never `transition: all`).
- Reduced motion turns it all to 1ms.

## 8. Accessibility floor

- Focus is a green ring on `:focus-visible`.
- Status is never colour alone.
- Hover, active and focus each have more contrast than rest.
- Native controls follow `color-scheme`.
- `prefers-reduced-transparency` makes the header and glass solid.
- Light mode is supported through `data-theme="light"`, with its own steps. Test it, but design dark first.

## 9. Checklist before you ship a UI change

1. Tokens only, and classes from theme.css. A new value becomes a token with its job written next to it, and a row here.
2. Run `python3 lab/static/sync_theme.py`; `--check` must pass.
3. Screenshot every page you touched:
   - at 1600, 1440 and 1280 wide;
   - in dark, then in light;
   - in every state: empty, loaded, selected, long content, drawer and dialog open, errors.
   Use Playwright with `/usr/bin/google-chrome`. Look at each screenshot and at 2× crops.
4. No page errors or console errors.
5. The catalogue tests pass (`scripts/glass/atlas_catalog_test.py`, which also runs t5 and t6; `scripts/glass/catalog_atlasonly_test.py`).
6. Compare with the references: Linear's density, Raycast's edges, Vercel's type, and the dashboard image's shell, KPI cards and accents. If a screen would look out of place next to them, it isn't done.

## 10. Planned, not done

- Render the maths in the lab's answers (KaTeX, as on the main branch).
- A command palette (⌘K), as in Linear and Raycast, to jump between networks and runs.
- Real charts where the data is a series. An example is a line chart of fidelity across networks in a family, in the style of the reference's revenue chart. It needs a question to answer first.
- If accents ever need theming, generate the scales in OKLCH from base, accent and contrast, as Linear does.
