# The Glass theme: how the lab should look, and how to keep it that way

*Version 1, Nov 20, 2026 (A49). Code: `lab/static/theme.css`. Why each rule exists: `UI_RESEARCH.md`.*

This file is for anyone (person or agent) who adds or changes a page. Follow it, and the lab stays one coherent product. If a rule here is wrong, change it here first, then in `theme.css`, then run the sync.

## 0. The idea in one paragraph

A dark, quiet instrument. The page is near-black. Content sits on **smoked-glass panels**: almost opaque, with an edge that catches light from above. True **frosted glass** (real blur) is kept for the things that float over content: top-bar capsules, the drawer, dialogs and tooltips. There is no hue in the chrome, so colour always means something:
- a status (a dot and a word);
- data (the heatmap's blue/red);
- the one primary action, which is drawn in inverse white.

Type is small, even and calm. One editorial serif moment per page.

## 1. How the theme is wired

- **`lab/static/theme.css`** is the only source.
  - Tokens are in §1, base styles in §2, surfaces in §3, controls in §4 and motion in §5.
  - The server serves only the three pages, so the theme is **inlined**: `python3 lab/static/sync_theme.py` copies it into each page between the `@theme:begin` / `@theme:end` markers.
  - `python3 lab/static/sync_theme.py --check` exits 1 if any page is out of date.
  - **Never edit the copies inside the pages.**
- **Page-specific CSS** comes after the `@theme:end` marker in each page's `<style>`. It may use only tokens (`var(--…)`), never raw colours. The exceptions are the alpha tints of status colours, which are written as rgb() to match the tokens.
- **Every page loads the same fonts:**
  ```html
  <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300..700&family=Geist+Mono:wght@400..600&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
  ```
  It also sets `<meta name="theme-color" content="#09090A">`.
- **Dark is the default.** Light exists through `data-theme="light"` on `<html>`; it is supported but not the designed-for case.
- **Old variable names some scripts still read** are aliased at the top of each page's CSS: `--neg`, `--pos`, `--field`, `--ink`, `--serif`, `--mono`, `--thin`. The heatmap code reads `--neg`, `--pos` and `--field` as **hex**, so the data tokens must stay hex.

## 2. Colour

**Surfaces.** Far is darker, near is lighter.

| Token | Dark | Use |
|---|---|---|
| `--bg` | #09090A | the page |
| `--well` | #0C0C0E | opaque sunken areas: inputs, the heatmap well |
| `--sunk` | black 26% | sunken areas on any surface: code, outputs, logs |
| `--surface-1` | #111113 | the base of a panel |
| `--surface-2` / `-3` / `-4` | #161618 / #1B1B1E / #242428 | raised inside a panel; the selected segment or chip |
| `--panel-bg` | gradient, 78–84% opaque, faint top sheen | panels (via `.t-panel`) |
| `--glass` / `--glass-strong` | 66% / 82% | floating layers (via `.t-glass`) |

**Lines and fills.** These are white at low alpha, so they work on any surface.

| Token | Value | Use |
|---|---|---|
| `--line-1` | white 5.5% | dividers inside a panel |
| `--line-2` | white 8% | edges of panels, cards and controls |
| `--line-3` | white 13% | hover and interactive edges |
| `--line-4` | white 22% | active and focus |
| `--fill-1`…`--fill-4` | white 3 / 5.5 / 8.5 / 12% | rest → hover → pressed → selected backgrounds |

**Text.** Four steps, never pure white.

| Token | Value | Use |
|---|---|---|
| `--text-1` | #EDEDEF | content, titles, values |
| `--text-2` | #A0A0A7 | secondary text, control labels at rest |
| `--text-3` | #6C6C73 | meta, hints, table headers, counts |
| `--text-4` | #47474D | disabled, placeholders on wells |

**Meaning.**
- **The primary action** is `--inv-bg` #EDEDEF with `--inv-text`. **One per view.** Examples: Read it, Construct, Random input, Add key.
- **Status:** `--ok`, `--warn`, `--bad` and `--info`, each with a `-soft` version at 11%.
  - Use them **only** as a 6px dot (`.t-dot.ok` etc.), a word, a 2px rule, or a fill of 11% or less (table cells such as the mechanism map).
  - **Never** use them for a button, a big block or a heading. Always pair the colour with a word.
- **Data:** `--div-neg` #E66767 ↔ `--div-mid` ↔ `--div-pos` #3987E5. This is the only place two hues meet. Sequential data uses one hue; categorical data needs a validated palette (see the dataviz guidance) and is never improvised.

**Don't:**
- hue-tinted greys (no blue-black);
- coloured tiles or tabs;
- gradients as decoration;
- white text at 100%;
- a second primary button.

## 3. Type

| Role | Font | Size / weight | Notes |
|---|---|---|---|
| UI and body | Inter | 13px / 400 | `cv11`, `ss03` on; antialiased |
| Labels, buttons | Inter | 12–13px / 500 | sentence case; no uppercase except the tiny brand tag |
| Panel titles | Inter | 13px / 600 | in a 44px header, with a count on the right in `--text-3` |
| Section titles | Inter | 14–15px / 600 | letter-spacing −0.01em |
| Big numbers | Inter | 28px / 500 | tabular, −0.025em |
| Long reading (answers, descriptions) | Inter | 14px / 400, line-height 1.7 | `text-wrap: pretty` |
| Numbers, ids, shapes, code | Geist Mono | 11–12px / 400–500 | tabular |
| Wordmark, the catalogue's page title, empty states | Instrument Serif | 20–30px | one serif moment per view; italic only for the wordmark and empty states |

**Rules:**
- Use weights 400, 500 and 600 only. 700 looks heavy on dark (the irradiation illusion).
- `text-wrap: balance` on headings and `pretty` on paragraphs; no orphaned words.
- Use real typographic characters where you write copy: the ellipsis "…", the multiplication sign "×" and the minus sign "−".

## 4. Shape, space, depth

- **Radii:** 4 (tiny), 6 (small controls), 8 (controls, rows, cells), 12 (panels), 16 (drawers and dialogs), pill (chips, capsules, badges). A child's radius is never larger than its parent's, and nested corners are concentric: an 8px segmented well with 2px padding holds 6px thumbs.
- **Spacing:** a 4px grid.
  - Panel padding is 12–16px; gaps between panels 12px.
  - Rows are 7–10px tall in lists and 8–10px in tables.
  - Control heights are 26px (small), 30px (default) and 36px (large).
- **Depth** is shown by surface lightness and edges first, shadows second.
  - `--shadow-panel`: two dark layers, nothing showy.
  - `--shadow-float`: a 0.5px black ring, a near shadow and a far shadow, plus an inner top highlight.
  - `--edge`: the 1px inner top highlight that every raised control carries.
- **The lit ring.** `.t-panel` and `.t-glass` draw their edge with a masked 1px gradient that is brighter at the top (`--ring`, `--ring-float`). This is what makes glass read as glass. Do not add a `border` to them as well.

## 5. Materials: when to use which

| You are making… | Use | Blur? |
|---|---|---|
| A panel or card holding content | `.t-panel` (or the page's panel class with the same recipe) | no |
| A top bar's group of actions, page navigation | a capsule with `.t-glass` | yes |
| A drawer, dialog, popover, tooltip | `.t-glass` (dialogs: `--glass-strong` fill) | yes |
| A modal backdrop | `--scrim` + `blur(4px)` | light |
| Something inside a panel that should look raised | `--surface-4` or `--fill-3` + `--edge` + `--line-2` ring | no |
| Something inside a panel that should look sunken | `--sunk` or `--well` + `--line-1` ring | no |

**Never:**
- glass on glass;
- glass in the content layer;
- milky glass (fill under 60%) behind text;
- blobs in the background.

The page background is the ambient light (`--ambient`) plus grain. Nothing else goes there.

## 6. Components (the classes in theme.css)

- **`.btn`:** secondary by default (`--fill-2`, ring, top highlight).
  - Variants: `.primary` (inverse; legacy alias `.orange`), `.yellow` (emphasised secondary), `.warm` (outline), `.ghost` (no chrome), `.on` (pressed or selected), `.sm` (26px).
  - Every button scales to 0.97 when pressed.
- **`.chip`:** 24px pill toggles and filters. Selected is **raised**, never inverted.
- **`.pill`:** a passive label.
- **`.seg`:** a segmented control (a well with a raised thumb).
- **`.toggle`:** a switch: `label.toggle > input[type=checkbox] + i`. On is inverse.
- **Fields:** inputs, selects and textareas sit on `--well` with a ring; focus adds a soft halo. Selects have a drawn chevron. Checkboxes are drawn: 14px, rounded, inverse when checked.
- **`details > summary`:** a drawn chevron that rotates.
- **Code:** `pre` on `--sunk`; inline `code` on `--fill-2`.
- **Tables:** hairlines between rows only, headers in `--text-3` at 12px/500, tabular numbers.
- **`.t-dot`:** a status dot (`.ok`, `.warn`, `.bad`, `.info`).
- **`kbd`:** keyboard hints.

**Patterns that are page CSS but shared in spirit:**
- **Status badges** (`.badge`, `.gb`, `.agree`, `.stage`, `.verdict`): a neutral pill or card, a coloured dot, a word.
- **Selected list rows** (`.tg.on`, `.entry.sel`, `.runitem.on`): `--fill-3` + `--line-1` ring. No inversion, no colour.
- **Panel header:** 44px, a 13px/600 title, a `--text-3` count or meta on the right, a `--line-1` divider.

## 7. Motion

- **Durations:** `--dur-1` 120ms (colour, hover), `--dur-2` 180ms (small moves, switches), `--dur-3` 280ms (panels entering, drawers).
- **Curves:**
  - `--ease-out` `cubic-bezier(.23,1,.32,1)` for anything entering;
  - `--ease-drawer` for the drawer;
  - plain `ease` for colour.
  - Never ease-in.
- Animate `transform` and `opacity` only; list properties explicitly (never `transition: all`).
- **Entrances:** `t-rise` (4px up and fade) for content swaps; `t-pop` (scale from 0.97) for dialogs; `t-pulse` for a running status dot.
- Don't animate what people do constantly (typing, scrolling lists, the heatmap).
- `prefers-reduced-motion` turns everything to 1ms; that rule is in the theme already.

## 8. Accessibility floor

- Every interactive element shows `--focus` (a 2px gap and a 2px white-alpha ring) on `:focus-visible`.
- Hover, active and focus each have more contrast than rest.
- Status is never colour alone (a dot plus a word).
- `color-scheme: dark` is set, so native scrollbars and pickers match.
- `prefers-reduced-transparency` swaps glass for solid surfaces.
- Text tokens 1–3 are readable on every surface; `--text-4` is only for disabled and placeholder text.

## 9. Checklist before you ship a UI change

1. Tokens only; no new raw colours. If you need a new value, add a token to `theme.css` with its job in a comment, and a row here.
2. Run `python3 lab/static/sync_theme.py`; `--check` must pass.
3. Screenshot every page you touched:
   - at 1600×1000 and 1280×900;
   - in dark and in light;
   - in every state: empty, loaded, selected, open drawers and dialogs, long content, errors.
   Use Playwright with `/usr/bin/google-chrome`, as in A49. Look at each screenshot, and at 2× crops of the details.
4. No page errors or console errors in any state.
5. Run the catalogue tests (`scripts/glass/atlas_catalog_test.py`, `scripts/glass/units/t5_catalog_test.py`, `scripts/glass/units/t6_test.py`, `scripts/glass/catalog_atlasonly_test.py`).
6. Ask whether each new element needs to exist. Remove before you add.

## 10. Planned, not done

These come from the research. They need a decision, or they touch more than presentation.
- **Navigation.** The catalogue links to the lab and the constructor, but those two don't link to the catalogue. A shared three-item capsule (Catalog · Lab · Constructor) on every page would make it one product. It is left out because it adds navigation, and A49 was presentation only.
- **Maths in the lab's answers.** Answers containing `\( … \)` show raw LaTeX here. The main branch renders it with KaTeX.
- **A proper icon set** (one stroke weight, 16px) to replace the text glyphs (`✕`, `▸`, `✓`). Lucide or Geist icons would fit.
- **The serif title in the catalogue.** Instrument Serif is condensed and editorial. If it ever feels too decorative, Inter Display at 28px/600 with −0.02em is the drop-in alternative.
- **Theme generation.** If accents or contrast levels are ever needed, generate the scales in OKLCH from three inputs (base, accent, contrast), as Linear did, instead of adding hand-picked values.
