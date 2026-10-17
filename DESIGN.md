---
name: Agri-Doctor Pro
description: Crop leaf diagnosis read like a soil-colour chart page, chips mounted with exact notation on a neutral mat.
colors:
  soil: "#2e241b"
  soil-raised: "#3d3126"
  soil-hover: "#43352a"
  soil-rule: "#55473a"
  bone: "#ece5d8"
  bone-dim: "#c4b8a3"
  mat: "#c9c4ba"
  mat-deep: "#bdb7ac"
  mat-hover: "#b4aea2"
  mat-edge: "#a9a398"
  sheet-paper: "#f1ece3"
  ink: "#231b14"
  ink-2: "#4a3f33"
  rust: "#9a3a2a"
  ochre: "#8a6420"
  leaf: "#46692a"
  signal: "#e0b12e"
  maize-swatch: "#c3933f"
  rice-swatch: "#8a9a5a"
  cotton-swatch: "#d9cfbf"
  router-key: "#d8c49a"
  model-key: "#8c7a62"
typography:
  display:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "clamp(2.1rem, 4.2vw, 3.6rem)"
    fontWeight: 800
    lineHeight: 1.02
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "2.6rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.03em"
  verdict:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "clamp(1.5rem, 2.6vw, 2.1rem)"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "-0.025em"
  wordmark:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.9rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.03em"
  title:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.55rem"
    fontWeight: 800
    letterSpacing: "-0.02em"
  chip-title:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.02rem"
    fontWeight: 400
    lineHeight: 1.6
  lead:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "1.1rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Hanken Grotesk, system-ui, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 600
  notation:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.78rem"
    fontWeight: 400
    letterSpacing: "0.04em"
    fontFeature: "tnum"
  notation-tag:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.66rem"
    fontWeight: 400
    letterSpacing: "0.06em"
    fontFeature: "tnum"
rounded:
  hair: "1px"
  chip: "2px"
  tab: "3px"
  sheet: "4px"
spacing:
  xs: "6px"
  sm: "0.5rem"
  md: "0.75rem"
  lg: "1.25rem"
  xl: "1.75rem"
  2xl: "3rem"
components:
  button-primary:
    backgroundColor: "{colors.soil}"
    textColor: "{colors.bone}"
    typography: "{typography.label}"
    rounded: "{rounded.sheet}"
    padding: "0.8rem 1.4rem"
  button-primary-hover:
    backgroundColor: "{colors.soil-hover}"
    textColor: "{colors.bone}"
  crop-tab:
    backgroundColor: "{colors.mat-deep}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.tab}"
    padding: "6px 12px 6px 8px"
  crop-tab-active:
    backgroundColor: "{colors.soil}"
    textColor: "{colors.bone}"
  state-tag-disease:
    backgroundColor: "{colors.rust}"
    textColor: "#ffffff"
    typography: "{typography.notation-tag}"
    rounded: "{rounded.chip}"
    padding: "2px 6px"
  state-tag-weed:
    backgroundColor: "{colors.ochre}"
    textColor: "#ffffff"
    typography: "{typography.notation-tag}"
    rounded: "{rounded.chip}"
    padding: "2px 6px"
  state-tag-healthy:
    backgroundColor: "{colors.leaf}"
    textColor: "#ffffff"
    typography: "{typography.notation-tag}"
    rounded: "{rounded.chip}"
    padding: "2px 6px"
  verdict-band:
    textColor: "#ffffff"
    typography: "{typography.verdict}"
    rounded: "{rounded.sheet}"
    padding: "1.6rem 1.75rem"
  conditions-cell:
    backgroundColor: "{colors.mat-deep}"
    textColor: "{colors.ink}"
    padding: "0.8rem 0.9rem"
  dropzone:
    backgroundColor: "{colors.mat-deep}"
    rounded: "{rounded.sheet}"
    padding: "2.25rem 1.75rem"
  dropzone-hover:
    backgroundColor: "{colors.mat-hover}"
  prescription-sheet:
    backgroundColor: "{colors.sheet-paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sheet}"
    padding: "2rem 2.25rem 1.5rem"
  rail:
    backgroundColor: "{colors.soil}"
    textColor: "{colors.bone}"
  rail-input:
    backgroundColor: "{colors.soil-raised}"
    textColor: "{colors.bone}"
    rounded: "{rounded.sheet}"
---

# Design System: Agri-Doctor Pro

## Overview

**Creative North Star: "The Soil Chart Page"**

The diagnosis reads like a page from a Munsell soil-colour book: flat earth swatches mounted on a neutral grey mat, each with exact notation beside it. A dark soil-brown rail on the left carries the wordmark, field inputs and a pipeline legend in bone text; the main area is a single mat ground where crop pages, mounted quadrant chips, a flooded verdict band and a paper prescription sheet appear in pipeline order.

Density is that of a reference chart, not a dashboard. Surfaces are separated by tone (mat against mat-deep) rather than strokes, numbers are always tabular, and colour carries meaning: three earth state colours for disease, weed and healthy, and one reserved signal yellow that belongs only to the AI overlay drawn on the leaf. The world rejects the white-card, green-button agritech dashboard: the primary action is soil brown and leaf green means "healthy" and nothing else.

**Key Characteristics:**
- Dark soil rail, grey mat ground, flat earth chips without borders.
- Every quadrant is a mounted chip: the crop image plus a tone strip of its measured mean colour, with its hex in mono.
- Three state colours (rust, ochre, leaf) used only to mean state.
- One reserved signal yellow for the AI quadrant overlay.
- Hanken Grotesk at weight 800 for titles; JetBrains Mono, tabular, for notation and scores.

## Colors

An earth palette of soil, bone and chart-mat greys, with three muted state pigments and a single reserved signal.

### Primary
- **Soil Brown** (soil): the rail ground, the primary button, the active crop tab, the crop-confidence meter fill, focus outlines, text selection and scrollbar thumb. It is the system's only action colour.
- **Raised Soil** (soil-raised): sidebar input fields and the model note block on the rail.
- **Lit Soil** (soil-hover): primary button hover only.
- **Soil Rule** (soil-rule): the 1px rule above rail headings and the rail's scrollbar thumb; also the sidebar border colour in the Streamlit theme.

### Secondary (state pigments)
- **Rust** (rust): disease. Fills the disease state tag, the chip confidence meter of a diseased quadrant, and the verdict band when any disease is found.
- **Ochre** (ochre): weed. Same three uses for weed quadrants and weed verdicts.
- **Leaf Green** (leaf): healthy. Same three uses for healthy quadrants and the healthy verdict.

### Tertiary
- **Signal Yellow** (signal): the quadrant cross-lines and Q1 to Q4 tags drawn onto the uploaded leaf (tags carry ink text), and the matching "Quadrant scan" key in the rail legend.
- **Crop swatches** (maize-swatch, rice-swatch, cotton-swatch): the small square keys inside the crop tabs, one per crop.
- **Pipeline keys** (router-key, model-key): legend squares for the router and crop-model stages in the rail; router-key is also the sidebar primary colour (toggle) in the Streamlit theme.

### Neutral
- **Bone** (bone): primary text on the rail, button text on soil, the AI-agronomist legend key.
- **Dim Bone** (bone-dim): captions, notation and secondary text on the rail.
- **Chart Mat** (mat): the main app background.
- **Deep Mat** (mat-deep): every inset surface on the mat: dropzone, ghost chip slots, inactive crop tabs, conditions cells, scrollbar track.
- **Pressed Mat** (mat-hover): dropzone hover.
- **Mat Edge** (mat-edge): hairlines (conditions grid gaps, sheet-head rule, expander border), the dashed dropzone stroke, meter tracks.
- **Sheet Paper** (sheet-paper): the prescription sheet, the one light surface in the system.
- **Ink** (ink) and **Ink 2** (ink-2): primary and secondary text on the mat and the sheet.

### Named Rules
**The Reserved Signal Rule.** Signal yellow marks only the AI overlay on the leaf and its own legend key. It never fills a button, tag, heading or background.

**The State-Only Pigment Rule.** Rust, ochre and leaf green appear only where they mean disease, weed or healthy: the chip's state tag, its confidence meter, and the verdict band. They are never decoration, and leaf green is never an action colour.

**The Held Verdict Rule.** The verdict band floods in its state colour with white text; the outcome is held in colour, not in an icon or outline.

## Typography

**Display Font:** Hanken Grotesk (with system-ui, sans-serif)
**Body Font:** Hanken Grotesk (with system-ui, sans-serif)
**Label/Mono Font:** JetBrains Mono (with ui-monospace, monospace)

**Character:** A heavy, tightly tracked grotesk for names and verdicts paired with a tabular mono that reads like chart notation. Grotesk says what something is; mono says how much and where.

### Hierarchy
- **Display** (800, clamp(2.1rem, 4.2vw, 3.6rem), 1.02): the page title only, max 22ch, balanced wrap.
- **Headline** (800, 2.6rem, 1; 2.1rem under 640px): the detected crop name on the crop page.
- **Verdict** (800, clamp(1.5rem, 2.6vw, 2.1rem), 1.1): the verdict band title.
- **Wordmark** (800, 1.9rem, 1): the rail wordmark.
- **Title** (800, 1.55rem): section titles on the mat; the sheet head uses the same voice at 1.45rem.
- **Chip title** (800, 1.15rem, 1.2; 1rem under 640px): the class name under each mounted chip.
- **Lead** (400, 1.1rem, 1.55): the page lead, max 60ch.
- **Body** (400, 1.02rem, 1.6 to 1.65): sheet text and status lines, 64 to 72ch.
- **Label** (600, 0.86 to 0.9rem): widget labels, crop tabs, the "Confidence" label.
- **Notation** (mono 400, 0.7 to 0.78rem, 0.03 to 0.05em, tabular): crop confidence line, leaf caption, chip IDs and hex values, runner-up scores, verdict counts, sheet notation, wordmark subtitle.
- **Notation tag** (mono 400, 0.66rem, 0.06em, uppercase): conditions field names and state tags only.

### Named Rules
**The Tabular Notation Rule.** Every score, percentage, reading and hex value is set in JetBrains Mono or with tabular figures, so columns of numbers align like chart notation.

**The Heavy Name Rule.** Names of things (page, crop, class, verdict, sheet) are Hanken Grotesk 800 with negative tracking; nothing between 600 and 800 is used for headings.

## Layout

A fixed left rail (Streamlit sidebar, soil ground) and a main mat column capped at 1280px, with 2.75rem top and 5rem bottom padding. Before upload the main column shows the title, lead, dropzone and a row of four ghost chip slots. After upload it splits into two columns: the leaf with its overlay on the left; crop tabs, crop name, confidence meter, conditions strip and the primary button on the right. Below, full width: section title, the four-chip grid, the verdict band and the prescription sheet, in pipeline order.

The chip grid is four equal columns (gap 1.75rem by 1.5rem); each chip mount is image plus a 22% tone strip with a 6px gap. The conditions strip is a three-column grid. Spacing runs on a small rem rhythm (6px, 0.5rem, 0.75rem, 1.25rem, 1.75rem, 3rem before section titles).

Under 640px: chip and ghost grids drop to two columns, conditions to two columns, the chip region name is hidden (only Q1 to Q4 remains), the crop name drops to 2.1rem and the sheet padding tightens to 1.4rem by 1.25rem.

## Elevation & Depth

Flat by default. Depth on the mat comes from tone: inset surfaces are mat-deep on mat, and grouped cells are separated by 1px mat-edge gaps. Only two objects lift.

### Shadow Vocabulary
- **Button lift** (`box-shadow: 0 2px 6px rgba(46, 36, 27, 0.25)`): the primary button only.
- **Sheet lift** (`box-shadow: 0 6px 18px rgba(35, 27, 20, 0.14)`): the prescription sheet, which reads as a paper sheet laid on the chart.

### Named Rules
**The Flat Chip Rule.** Chips, swatches, tabs, conditions cells and the verdict band carry no shadow. Lift is kept for the one action and the one paper sheet.

## Shapes

Nearly square corners graded by size: 1px for meters, 2px for chips, swatches, legend keys and state tags, 3px for crop tabs and photographs, 4px for containers (buttons, dropzone, conditions strip, verdict, sheet, alerts, rail note). Strokes are rare: 1px hairlines inside grouped strips and under the sheet head, a 1px soil-rule above rail headings, and a 1.5px dashed mat-edge stroke on the empty dropzone.

**The Borderless Mat Rule.** Chips and cards have no outline. A surface is defined by its tone against the mat; strokes are only dividers inside a group or the dashed edge of an empty page.

## Components

### Buttons
- **Shape:** gently squared (4px).
- **Primary:** soil ground, bone text, weight 700 at 1rem, padding 0.8rem by 1.4rem, button lift shadow.
- **Hover / Focus:** background shifts to lit soil over 160ms ease-out; pressing nudges down 1px over 120ms; focus-visible draws a 2px soil outline offset 2px.
- **Upload:** the dropzone's button uses the same soil and bone pairing at weight 600.

### Chips (crop tabs and state tags)
- **Crop tabs:** deep mat ground, ink-2 text, 600 at 0.9rem, 3px corners, a 14px crop swatch on the left. The detected crop's tab inverts to soil and bone. Tabs are a display of the routing result, not navigation.
- **State tags:** mono uppercase 0.66rem in white on the state pigment, 2px corners, padding 2px by 6px.

### Cards / Containers
- **Conditions strip:** a 4px-cornered grid of deep-mat cells divided by 1px mat-edge gaps; each cell holds a mono uppercase field name over a 700-weight tabular value.
- **Verdict band:** flooded state colour, white title in the verdict voice, mono counts below (always including the AI-classification-not-laboratory caveat), then run details under a 1px translucent white rule.
- **Prescription sheet:** sheet paper, 4px corners, sheet lift, padding 2rem by 2.25rem; a head row with a 1.45rem title and right-aligned notation over a mat-edge rule; body text at 1.02rem / 1.65, max 72ch.

### Inputs / Fields
- **Rail inputs:** Streamlit fields on raised soil with bone text and caret; labels 600 at 0.86rem in bone; captions dim bone at 0.78rem.
- **Dropzone:** styled as an empty chip page: deep mat, 1.5px dashed mat-edge stroke, 4px corners; on hover the ground darkens to pressed mat and the stroke turns soil over 160ms.

### Navigation
- **Rail:** soil ground with the wordmark, a mono notation subtitle, field inputs, the AI toggle, then a "How a leaf is read" legend: each pipeline stage has a 14px square key in its own colour (router key, model key, signal, bone) with a bone name and dim-bone description.

### Mounted Quadrant Chip (signature)
The quadrant crop as a square image with a vertical tone strip (22% wide) filled with that quadrant's measured mean colour, both at 2px corners. Below: a mono ID row (Q number and region on the left, the tone's hex on the right), the class name in chip-title voice, a state tag, a "Confidence" row with a tabular score, a 6px meter (mat-edge track, state-colour fill) and mono runner-up scores. Chips mount in with a 520ms cubic-bezier(0.16, 1, 0.3, 1) rise and clip reveal, staggered 90ms each, and the animation is off under reduced motion. Before upload, ghost chips (deep-mat blocks with Q labels) hold the four slots.

## Do's and Don'ts

### Do:
- **Do** mount every quadrant result as image plus mean-tone strip with its hex in mono.
- **Do** set every number in JetBrains Mono or tabular figures.
- **Do** separate surfaces on the mat by tone (mat-deep on mat) and 1px mat-edge gaps.
- **Do** flood the verdict band in its state colour and keep the AI-classification caveat in it.
- **Do** keep soil brown as the only action colour, with bone text.
- **Do** honour reduced motion by dropping the chip-mount animation.

### Don't:
- **Don't** use signal yellow anywhere except the leaf overlay and its legend key.
- **Don't** use rust, ochre or leaf green for anything other than disease, weed and healthy state.
- **Don't** outline chips or cards; the only strokes are group hairlines, the rail rule and the dashed empty dropzone.
- **Don't** add pure white surfaces or white cards; sheet paper is the one light surface, and only for the prescription.
- **Don't** add shadows beyond the primary button and the prescription sheet.
