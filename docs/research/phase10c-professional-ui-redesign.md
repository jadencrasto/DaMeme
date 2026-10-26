# Phase 10C: professional cinematic UI redesign

## Scope and boundary

This frontend-only redesign retains the 281-item A0 catalog and existing FastAPI contracts. It does not expose M1/OpenCLIP, M2/RRF, the archived lexical reranker, the recovered 1K catalog, research ratings, or any new model behavior.

## Design system and page structure

The interface uses near-black charcoal surfaces with warm amber calls to action and restrained lavender lighting. Editorial serif display type is reserved for the brand and primary headings; a readable system sans-serif handles controls. Borders, shadows, radius, and animation are deliberately quiet so the source images remain the focal point.

The homepage is organized as a compact header (including local-feedback access), editorial hero, discovery panel, image-first recommendation gallery, and collapsible session-history drawer. It keeps a single page and does not add accounts, social features, or navigation destinations.

## Discovery modes and mood selection

The discovery panel has keyboard-operable tabs:

- **Describe a situation** submits only situation text plus an optional selected mood.
- **Just a mood** submits only the required mood and never carries situation-tab draft text.

Each tab preserves its own draft. The shared request pipeline continues to use A0, batch IDs, cancellation/version protection, safe exclusions, template diversity, feedback identity, and exhaustion behavior.

The mood selector makes eight common moods available as quick chips and provides a search field for all 32 taxonomy entries. Selected states use `aria-pressed`; optional mood selection can be cleared, while the mood-only selector cannot deselect its required current choice. The selector uses only existing taxonomy IDs, descriptions, and mappings.

## Gallery and interactions

Recommendation cards now prioritize naturally sized images in a responsive, masonry-inspired grid. Image text is never cropped or distorted. Ordinary card surfaces omit similarity values, classifier categories, and retrieval explanations; inspection, download, sharing, and useful/not-relevant feedback remain separately labelled actions. The existing modal retains Escape closing and focus restoration.

“Show me more” retains the active successful request rather than live form state, so changing drafts cannot corrupt exclusions. The history drawer retains bounded session-only entries, supports revisit across all three input combinations, and clear-history control.

## Accessibility and performance

- Native buttons, landmarks, visible focus treatment, ARIA tabs/panels, pressed mood chips, and live loading/result/error states are retained.
- Tab arrow/Home/End handling switches the active discovery mode; ordinary Tab, Enter, and Space controls remain available.
- Layout collapses to single-column controls and cards on small screens without horizontal overflow.
- Existing lazy-independent image loading and `object-fit: contain` preserve media performance and geometry. No dependencies were added.
- Reduced-motion preferences suppress decorative transition movement.

## Validation and browser limits

Frontend ESLint passed; Vitest passed 19 tests in 7 files with no skipped tests; TypeScript/Vite production build passed; and `git diff --check` passed. The updated suite replaces the prior skipped legacy App-layout tests with active cinematic interface coverage, including both modes, combined mood submission, searchable mood selector, loading/error state, and image action availability.

Unit tests cannot validate real browser download prompts, native share sheets, clipboard permissions, physical-device touch ergonomics, or the appearance of locally served source images. These remain manual browser checks for the existing 281-image startup path.
