# UI research: what makes a dark, glassy, minimal interface look professional

*Written Nov 20, 2026 (15:59 ET by the system clock) for the A49 redesign, and extended for A50 (§7). Branch `ui-design`. The current rules are in `THEME.md` (v2); the code is `lab/static/theme.css`, `shell.html` and `shell.js`.*

> §1–§6 record the first round (A49). Several of its decisions were replaced in round two (§7): no hue in the chrome, the serif, Inter and the panel layout. They are kept here because the measurements still hold.

Shreyas asked for a researched, production-grade redesign of the lab's three pages: subtle, minimal, decluttered, dark, and glassy, with glass that does not look amateur. This file records what I read and measured, what I concluded, and why.

## 1. How the research was done

I did three things.

1. **Read the primary design writing.** This covered:
   - Apple's Human Interface Guidelines on materials and Liquid Glass;
   - Material Design's dark theme;
   - Superhuman's dark-theme rules;
   - Linear's account of its redesign;
   - Vercel's Geist system and its Web Interface Guidelines;
   - Radix Colors' 12-step scale;
   - Josh Comeau on frosted glass and on shadows;
   - Emil Kowalski on motion;
   - Nielsen Norman Group on glassmorphism;
   - articles on dark-mode typography.
2. **Measured real products.** I loaded Linear, Raycast, Resend, Vercel, Geist (Vercel's design-system site), Cursor and PlanetScale in headless Chrome (dark colour scheme). I took screenshots and pulled the computed styles of every element:
   - background, border and text colours;
   - fonts, sizes, weights and letter spacing;
   - radii and shadows;
   - the exact recipe of every element that uses `backdrop-filter`.
   Opinions about "what Linear does" are cheap; the numbers in §3 are what their CSS actually says.
3. **Built, screenshotted and iterated.** Every page and state was screenshotted in both themes and compared against the references (§6).

## 2. What the sources say

### Dark mode
- **Don't use pure black or pure white.**
  - Superhuman: pure black hides shadows and smears on OLED screens; white on black (21:1) causes halation, worse for people with astigmatism. They set text at about 90% white.
  - Material recommends #121212 as the base surface.
- **Far surfaces are darker, near surfaces lighter.**
  - Superhuman: light falls from above, so a modal is lighter than the page. Inverting a light theme gets this backwards.
  - Material expresses depth by lightening surfaces as they rise.
- **Tune contrast by eye per element, not by one formula.** Superhuman puts contact details at 60% black in light mode but 65% white in dark mode. Both pass WCAG; the offset reduces fatigue.
- **Avoid large blocks of bright colour.** They pull focus from content in the dark; Superhuman removed a coloured overlay for this reason.
- **Desaturate accents.** Saturated colours "vibrate" on dark surfaces (Material; UX Planet). Superhuman deepened its purple so text stays the focus.
- **Shadows barely read on near-black.** Ported light-mode shadows look smudged. Depth on dark comes mostly from lighter surfaces and light edges.

### Glass
- **Glass belongs to the floating layer, not the content.**
  - Apple's current guidance: "Don't use Liquid Glass in the content layer." It forms "a distinct functional layer for controls and navigation… that floats above the content layer", and should be used sparingly on custom controls.
  - Inside the content layer Apple uses standard materials, where thicker means more opaque and more legible.
- **Glass only shows when something is behind it.**
  - Apple: use the clear variant only over visually rich backgrounds.
  - Axess Lab: glass works best over images or gradients.
  - On a flat background, a blurred translucent card is just a grey card that costs GPU time.
- **Don't stack glass on glass.** Avoid busy backdrops and overuse (LogRocket; NN/g: meet contrast requirements first).
- **Edges make glass read as glass.**
  - A thin, semi-transparent "specular" border simulates the lit edge of real glass.
  - Josh Comeau shows a refinement: a thin strip with a smaller blur and a brightness boost, so the slab looks 3-D.
  - Naive `backdrop-filter` only samples pixels directly behind the element; his extended-backdrop trick fixes that for sticky headers.
- **Give it a fallback and respect settings.** Use an opaque fallback when `backdrop-filter` is unsupported, and a solid surface under `prefers-reduced-transparency`.

### Colour systems
- **Linear** rebuilt its theme generator on LCH (perceptually uniform), going from 98 variables per theme to 3: base, accent and contrast. Its elevation levels (background, foreground, panels, dialogs, modals) are generated. The redesign:
  - limited blue in the neutrals for a more neutral, timeless look;
  - explored hierarchy with black and white opacities before adding any colour.
- **Vercel Geist** has 10 steps per scale, each with a job: 100 background, 200 hover, 300 active, 400 border, 500 hover border, 600 active border, 700–800 high-contrast fill, 900 secondary text, 1000 primary text. An alpha grey scale is used where the background must show through.
- **Radix** has 12 steps: 1–2 backgrounds, 3–5 component states, 6–8 borders, 9–10 solid fills, 11–12 text (guaranteed APCA contrast on step 2).
- **What I took from these:** every colour has a job, and states (rest, hover, active) are steps of the same scale.

### Typography
- **Linear** uses Inter Display for headings and Inter for the rest. Its measured body text is 13px, with a medium weight of 510 rather than 600.
- **Vercel** uses Geist and Geist Mono, with tight negative tracking at display sizes. Both are free on Google Fonts.
- **Light text on dark looks heavier** than the same weight dark on light (the irradiation illusion). Advice ranges from slightly lighter weights to a hair more tracking; Raycast uses +0.2px tracking on body text.
- **Always turn on `-webkit-font-smoothing: antialiased` on dark.** Subpixel antialiasing makes light-on-dark text look bold.
- **Use tabular numbers** for anything compared in columns (Vercel guidelines).

### Shadows and edges
- **Shadows** (Josh Comeau): one light source for the whole page, several stacked shadows at low opacity, offset and blur growing with elevation. Tint shadows to the background's hue where it has one.
- **Borders** (Vercel guidelines): layered shadows with semi-transparent borders give crisp edges. Nested radii must be concentric (child radius ≤ parent radius).
- **Interaction contrast:** hover, active and focus states must have more contrast than rest.
- **Native controls:** set `color-scheme: dark` so scrollbars and native controls match, and set `theme-color`.

### Motion
- **Emil Kowalski** (Vercel, Sonner, Vaul):
  - Ease-out for anything entering or exiting. Never ease-in on UI: it delays the moment the user is watching.
  - Usually under 300ms.
  - Don't animate things people do a hundred times a day.
  - Built-in CSS easings are too weak; use a strong curve such as `cubic-bezier(.23,1,.32,1)`.
- **Vercel guidelines:**
  - Animate only `transform` and `opacity`.
  - Never use `transition: all`.
  - Always honour `prefers-reduced-motion`.
- **web.dev** gives the same default: ease-out, 200–500ms.

### Detail work
- **Rauno Freiberg** (Invisible Details of Interaction Design): polish lives in details people don't consciously notice but feel.
- **Linear's sidebar alignment work** is described as "something that you'll feel after a few minutes."
- **Vercel's guidelines:**
  - deliberate alignment to a grid, an edge or an optical centre;
  - no orphaned words (`text-wrap: pretty`);
  - designed empty, sparse, dense and error states.

## 3. What the products actually do (measured)

Computed styles, headless Chrome, 1440×900, dark colour scheme, Nov 20, 2026.

| | Linear | Raycast | Vercel / Geist | Resend | Cursor |
|---|---|---|---|---|---|
| Page background | rgb(8, 9, 10) | rgb(7, 8, 10) | #000 (cards #0A0A0A) | #000 | rgb(20, 18, 11) (warm) |
| Most common border | 1px white 8% | 1px white 10%, then 6% | 1px white 14% / #2E2E2E | 1px rgba(214,235,253,.19) | 1px white-ish 10% |
| Primary text | rgb(208, 214, 224) | white | rgb(237, 237, 237) | rgb(240, 240, 240) | rgb(237, 236, 236) |
| Secondary text | rgb(138, 143, 152) | rgb(156,156,157) / white 60% | rgb(161, 161, 161) | rgb(161, 164, 165) | 55–60% of primary |
| Tertiary text | rgb(98, 102, 109) | rgb(106, 107, 108) | – | rgb(108, 108, 108) | – |
| UI font, size | Inter Variable, 12–15px | Inter, 13–16px (+0.2px tracking) | Geist, 14px | Inter, 14px | own gothic, 14px |
| Mono | Berkeley Mono | Geist Mono | Geist Mono | Commit Mono | Berkeley Mono |
| Display | Inter | Inter | Geist, -2.4px at 40px | a serif (Domaine) | EB Garamond in places |
| Radii in use | pill, 8, 4, 12, 6 | 11–12, 6, 10, 20 | 6, pill, 8 | pill, 16, 6, 8 | 4, 8, 10, 12 |

**The glass recipes:**
- **Linear's header:** `blur(20px)` over a vertical gradient from black at 80% to near-black at 76%.
- **Raycast's navbar:**
  - fill: a diagonal gradient from rgba(17,18,20,.75) to rgba(12,13,15,.90);
  - `blur(5px)`, 1px border at white 6%, radius 16;
  - an inner top highlight: `inset 0 1px 1px rgba(255,255,255,.15)`.
- **Raycast's product window:**
  - fill: a near-opaque gradient (96–98%) with `blur(30px)`;
  - a four-layer shadow: `0 16px 48px 8px` at 20%, a 0.5px black ring at 75%, `0 4px 8px` at 40%, `0 12px 32px` at 80%.
- **Raycast's floating pills:** `blur(24px) saturate(1.5)` over a white 12–15% gradient that is brighter at the top and bottom edges.
- **Linear's secondary button:** white 5% fill with `blur(4px)`. Its shadow:
  - an inset 1px white 3% ring;
  - an inset top line of 1px white 4%;
  - an outer 1px black 60% ring;
  - a soft `0 4px 4px` black 10% drop.

**What the numbers show:**
- Real dark UIs sit at L ≈ 3–4% (rgb 7–10), not at #121212, and never at pure #000 for app surfaces.
- They use white-alpha hairlines at 6–14%, and three or four text steps.
- They use 13–14px Inter-class type at regular and medium weights.
- Their glass is **almost opaque** (75–98% fill) with a lit edge. Very transparent milky glass is what amateur glass looks like.
- **Resend and Cursor** pair a serif display face with a sans UI, which is the role Instrument Serif plays here.

## 4. What was wrong with the previous look

**Lab and constructor (cream, neo-brutalist):**
- 3px black borders and hard offset shadows on every panel;
- four saturated accent colours with no assigned jobs;
- italic serif titles on every panel;
- monospace everywhere.
Everything shouted at the same volume.

**Catalogue (A48, "neutral glass"), which read as amateur glass because:**
1. **Glass everywhere.** Every card, tile, tab bar and nav had `backdrop-filter`. Behind them was a flat background, so the blur did nothing visible: Apple's "don't use glass in the content layer", violated wholesale.
2. **Smudges for a background.** Three white radial "blobs" at 5–11% with a 30px blur read as dirt on the screen, not light.
3. **Milky surfaces.** White 4.5% over black, with a 22px radius and a big soft shadow everywhere. The result is bubbly and toy-like, with no hierarchy between hero, tile and section.
4. **A gimmick highlight.** Every tile had a diagonal white gradient overlay.
5. **Loud selection.** Fully white chips, nav items and labels for "on". Inverted white is the strongest signal on the page, and it was spent on filters.
6. **Heavy type.** 700-weight numbers, uppercase tracked monospace labels on every tile, and italic serif on every heading.

## 5. The decisions (and the source for each)

| Decision | Why |
|---|---|
| Background #09090A; panels near-opaque (78–84%) with a faint top sheen | Measured products sit at rgb 7–10; glass fill 75–98% (Raycast, Linear) |
| Glass with real blur only for floating chrome: top-bar capsules, the drawer, dialogs, the tooltip | Apple HIG; blur shows only where content passes behind |
| A 1px ring brighter at the top (a masked gradient), instead of a flat border | The "specular edge" (glass references; Raycast's inset highlight); a flat border reads as a box |
| A soft ambient light at the top of the page, plus 3.5% grain | Gives translucency something to show without blobs; grain prevents banding (Vercel guideline) |
| Neutral greys, no hue in the chrome | Shreyas, A48; Linear limited blue for a timeless neutral |
| Four text steps (#EDEDEF, #A0A0A7, #6C6C73, #47474D) | Matches the measured primary/secondary/tertiary of Linear, Vercel and Raycast; never pure white (Superhuman) |
| White-alpha hairlines at 5.5 / 8 / 13 / 22% | Measured: 6–14% everywhere |
| One primary action per view, drawn inverse (light on dark); no coloured buttons | Vercel's primary button; concentrates attention |
| Selected = raised (a lighter surface with an edge), not inverted | Avoids spending the strongest signal on filters (§4.5) |
| Status = a coloured dot plus a word, fills ≤ 11% | Never colour alone (Vercel); no large colour blocks (Superhuman) |
| Inter (optical sizes) for UI at 13px; Geist Mono for numbers and code; Instrument Serif only for the wordmark, the catalogue's page title and empty states | Linear's density; Geist Mono measured on Raycast and Vercel; one editorial moment per page (Resend, Cursor) |
| Weights 400/500/600 only; tabular numbers | Linear's 510 medium; Vercel's tabular-nums rule |
| Radii 4/6/8/12/16 plus pill; concentric nesting | Measured radii; Vercel's nested-radius rule |
| Motion: 120–280ms, strong ease-out, transform and opacity only, reduced-motion honoured | Kowalski, Vercel, web.dev |
| Heatmap: blue ↔ red with a neutral grey midpoint, from a validated palette | A diverging scale needs two poles and a neutral midpoint (dataviz reference palette) |

## 6. How it was tested

- **Pages:** all three, at 1600×1000 and 1280×900, in dark and light.
- **States:**
  - lab: default, a bigger network, a region dragged on the heatmap with the tooltip, python mode, a run's output (rebuilt from a saved run, so no money was spent), the notebook drawer with a row ticked, a saved run's dialog, the prompt preview;
  - constructor: default, scrolled through the tables, the prior-only run, the language-model tasks opened, a small network;
  - catalogue: every tab, with the atlas step-through and the playground run on local endpoints, filtered by chip and search, and another network.
- **Errors:** none in the console or page in any state.
- **Tests:** the catalogue's own tests pass (see `docs/glass/CHANGES.md` A49).
- **Scripts:** `/tmp/ui/states.py` during the work; the method is described in `THEME.md` §9.

## 7. Round two (A50): from a restyle to a revamp

**Shreyas's verdict on A49:** still incremental. He asked for everything to change (font, design, layout), with Linear, Raycast and Vercel as the design systems, plus a reference image. The image is a dark "SalesOps" sales dashboard shown at three depths: Overview, Team Performance, Pipeline.

### What the reference image does (read off the image)
- **App shell.**
  - A ~250px left sidebar: brand chip (white rounded square, dark glyph) and name at the top; icon + label nav items about 42px tall.
  - The selected item has a **green 1.5px outline with a faint green glow** and a green icon.
  - A 64px header with a 20px title and a context line (calendar icon, "Last 30 days"). On the right: a search field, a bell with a green dot, and a gradient avatar.
- **KPI cards.**
  - Grey label top left, a small bordered icon box top right.
  - A ~32px semibold number with a coloured delta beside it (green "↗ +12.5%", red "↘ −5").
- **Cards with a title and a grey subtitle:**
  - "Revenue Trend · Monthly performance vs target", with a two-series line chart on dashed gridlines and a legend of dots;
  - "Pipeline Stages · Distribution by stage", made of progress rows (label, count, percent, a thin coloured bar).
- **Colour:**
  - near-black zinc surfaces, cards one step lighter with hairline borders;
  - **green** as the brand and positive colour, **cyan** as the first data series;
  - orange and red only for a stage or a decline;
  - the primary button is green.
- **Type:** a modern grotesk with tight tracking on headings. Geist's proportions match it closely (the measured Vercel stack).

### How the three named systems map onto it
- **Linear:** zinc near-black at rgb 8–10; white-alpha hairlines (8% most common); 13–15px UI type in an Inter-class face; dense lists with quiet group headers; a selected row as a tinted background, not an inversion.
- **Raycast:** lit edges (an inner top highlight); near-opaque floating surfaces with `blur(24–30px) saturate(1.5)`; layered window shadows with a 0.5px black ring.
- **Vercel / Geist:** the Geist and Geist Mono fonts (both free on Google Fonts); 10-step scales where each step has a job; underline tabs; tabular numbers; the Web Interface Guidelines for focus, motion and contrast.

### What changed from A49, and why
| A49 | A50 | Why |
|---|---|---|
| Inter + Instrument Serif + Geist Mono | **Geist + Geist Mono** only | "Use their fonts": Geist is Vercel's face and matches the reference's grotesk. One family reads as one product; the serif read as decoration |
| No hue anywhere in the chrome | **Green accent + cyan partner**, status colours | The reference is green-led; colour now marks the primary action, selection and meaning |
| Three floating panels per page, same layout as the old pages | **App shell**: sidebar (all three pages in one navigation), header, list panes, KPI rows, cards, underline tabs | The ask was a revamp, not a restyle. Layout is what makes the reference read as a product |
| Selection = a raised grey surface | **Selection = a green outline and tint** (sidebar, chips, tensors, tabs, list bar) | Taken directly from the reference's sidebar |
| Fidelity numbers in a key–value table | **Progress rows** (the "Pipeline Stages" pattern) | Shares read faster as bars |
| Summary numbers as tiles inside one panel | **KPI cards** with icon boxes and deltas (e.g. "+2.7 pts" against the task rule; "+80 pts" against a constant output) | The reference's main pattern; the deltas compare numbers the page already shows |
| Text glyphs as icons (✕, ▸, ◐) | **Lucide icons** throughout (ISC licence) | Linear, Raycast and the reference all use a consistent line-icon set |
| Heatmap blue ↔ red | **Orange ↔ cyan**, neutral zinc midpoint | Matches the palette; validated with the dataviz palette checker (CVD ΔE 21, both poles in the dark lightness band) |

### Nothing fake
The reference has a bell and a user avatar. The lab has no notifications and no accounts, so neither was added. Every number on a KPI card is one the page already had: params, tensors, numbers on the slide, prompt tokens, stages, matches, characters. The deltas are differences between two numbers already shown side by side.

### Testing
- **Pages and states:** every page at 1600, 1440 and 1280 wide, in dark and light, in the same states as round one: lab heatmap selection, output, drawer and dialogs; constructor tables, runs and tasks; catalogue tabs, step-through, playground and filters.
- **Sidebar:** the rail breakpoint and the manual collapse were both checked.
- **Errors:** none in any state.
- **Tests:** `atlas_catalog_test.py` (with t5 and t6) and `catalog_atlasonly_test.py` pass.

## 8. Round three (A51): cards and progressive disclosure

**Shreyas's verdict on A50:** it still showed walls of text, fully laid-out lists and filters everywhere. He wants a usable, readable, comfortable system:
- everything in cards;
- long text, lists and options put away behind a button, a menu or a "see all", until he asks;
- no big headline numbers.

His second reference is a card-based dark dashboard ("Tract SEO"). Its pattern:
- a gradient balance card as the one bright element;
- small cards, each with an icon box, a title and one action;
- a notification column of cards with "See all";
- a chart card with an open arrow (↗);
- options inside dropdowns ("Last 7 Month ▾").

**What the usability literature says:**
- **Progressive disclosure** is Nielsen's name for this. Jakob Nielsen defines it as showing users only the most important options at first and revealing the specialized ones on a secondary display when asked. NN/g lists it as one of application design's primary guidelines.
  - It improves 3 of usability's 5 components: learnability, efficiency of use and error rate.
  - Placement carries meaning: whatever appears on the initial display tells users it is important.
  - The cost is discoverability. The secondary controls must be clearly visible and simple to operate.
- **Cards** (NN/g): a card is a snapshot meant to invite a click or tap for more details. The border groups its content, and a slight shadow signals that it is clickable.
- **Card guidelines** from design systems:
  - The European Commission's component library: cards are an entry point to more detailed information. Use them for browsing and scanning in small portions, not for searching. Limit links on a card, ideally to the one that opens it.
  - UX Planet: one subject per card (a title, a short summary, optional actions). Cards make comparing harder, so use a list or grid view for comparison. That is why comparisons here (fidelity measures, sites, interventions) are progress bars or tables inside a sheet.

**How it was applied:**

| Was always on screen | Now |
|---|---|
| Catalogue: a 58-row network list | A network menu in the header; "All networks" opens every network as a card, grouped by family, with a filter menu |
| Catalogue: five headline-number tiles | A gradient hero card (name, task, ids, actions) plus a "How faithful" card with two bars ("Show every measure" reveals the rest) |
| Catalogue: findings as numbered paragraphs | The three key findings as cards, clamped; "See all" opens the rest in a sheet |
| Catalogue: a 180-row mechanism table with three always-visible filters | A summary card (counts, the top 3); "Browse all" opens a view of mechanism cards, 30 at a time, search plus a filter menu, each card expanding for its effect |
| Catalogue: certificate table, not-explained list, failures, code, report | Buttons on the construction card, each opening a sheet; the description is clamped with "Read the full description" |
| Lab: the specimen list, weight variants, unit picker, format options, templates and all reader settings | The header specimen menu; the "Trained weights ▾", "More ▾", "Format ▾" and "Template ▾" menus; a settings menu, with a one-line summary beside "Read it" |
| Lab: four headline-number tiles | Removed. The counts sit in card subtitles |
| Lab: a reading's thinking, code and answer all expanded | A reading card: the answer clamped to 7 lines ("Read the full answer", or open it in a sheet); thinking and Python folded; maths rendered with KaTeX |
| Lab: 15 slide items in rows | The first 6, then "+9 more" |
| Constructor: run settings, a runs list, a tasks list and three always-open panes of logs and tables | "New run ▾" and "Language-model tasks" in the header; runs as cards; the construction as a verdict hero card, stage cards and evidence cards (Units, Interventions, Mechanism, Circuit, Transfer, Benchmark), each opening its table in a sheet; the reader's log as one-line step cards that expand |

**Nothing was lost.** Every table and text the pages had is one click away.
- The catalogue tests still find their DOM: the mechanism rows in `#atlas table.mechs` are kept and drawn as cards, and `#parts` keeps its text.

## Sources

- [Apple HIG: Materials](https://developer.apple.com/design/human-interface-guidelines/materials)
- [Apple HIG: Dark Mode](https://developer.apple.com/design/human-interface-guidelines/dark-mode)
- [Material Design: Dark theme](https://m2.material.io/design/color/dark-theme)
- [Superhuman: How to design delightful dark themes](https://blog.superhuman.com/how-to-design-delightful-dark-themes-7b3da644ff1f)
- [Linear: How we redesigned the Linear UI (part II)](https://linear.app/now/how-we-redesigned-the-linear-ui)
- [Vercel Geist: Colors](https://vercel.com/geist/colors), [Materials](https://vercel.com/geist/materials), [Font](https://vercel.com/font)
- [Vercel: Web Interface Guidelines](https://vercel.com/design/guidelines)
- [Radix Colors: Understanding the scale](https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale)
- [Josh W. Comeau: Next-level frosted glass with backdrop-filter](https://www.joshwcomeau.com/css/backdrop-filter/)
- [Josh W. Comeau: Designing beautiful shadows](https://www.joshwcomeau.com/css/designing-shadows/)
- [Emil Kowalski: Great animations](https://emilkowal.ski/ui/great-animations)
- [web.dev: Choosing the right easing](https://web.dev/articles/choosing-the-right-easing)
- [Rauno Freiberg: Invisible Details of Interaction Design](https://rauno.me/craft/interaction-design)
- [NN/g: Glassmorphism](https://www.nngroup.com/articles/glassmorphism/)
- [CSS-Tricks: dark mode and variable fonts](https://css-tricks.com/?p=330342)
- Measured live: linear.app, raycast.com, resend.com, vercel.com, vercel.com/geist, cursor.com, planetscale.com
- [NN/g: Progressive disclosure](https://www.nngroup.com/articles/progressive-disclosure/), [NN/g: Cards component](https://www.nngroup.com/articles/cards-component/), [Nielsen: Progressive disclosure (UX Tigers)](https://uxtigers.com/post/progressive-disclosure)
- [European Commission component library: Cards](https://ec.europa.eu/component-library/eu/components/card/usage), [UX Planet: Simple design tips for crafting better UI cards](https://uxplanet.org/simple-design-tips-for-crafting-better-ui-cards-19c1ac31a44e)
- [Lucide icons](https://lucide.dev) (ISC licence), [Geist on Google Fonts](https://fonts.google.com/specimen/Geist)
- Shreyas's reference images: a dark "SalesOps" dashboard and a dark "Tract SEO" card dashboard (Nov 20, 2026, in the conversation; not in the repo)
