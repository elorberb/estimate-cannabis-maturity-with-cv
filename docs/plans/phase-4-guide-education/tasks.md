# Phase 4: Lens Guide, Photo Quality & Education — Tasks

**9 tasks** | Weeks 11-12

## Task Breakdown

| # | Task | Assignee | Description | Dependencies | Scope |
|---|------|----------|-------------|--------------|-------|
| P4.1 | **Sharpness check endpoint** | Etay | Implement image quality check using Laplacian variance (adapt from `src/data_preparation/`). Return sharpness score + pass/fail | Phase 1 complete | Small |
| P4.2 | **Content API endpoints** | Etay | Serve lens guide + education content as structured JSON. Store in Supabase or static files | Phase 1 complete | Small |
| P4.3 | **Write educational content** | Etay | Adapt paper content for growers: trichome stages, harvest timing, how to use app. Markdown/JSON format | None | Medium |
| P4.4 | **Write lens guide content** | Etay | Research and write lens recommendations with links, attach/use instructions, example photos | None | Medium |
| P4.5 | **Camera quality check integration** | Gili | Before upload, call quality-check endpoint (or on-device Laplacian). Show warning if blurry, retake option | P4.1 | Medium |
| P4.6 | **Lens guide screen** | Gili | Render lens guide content, product cards with images/links, example photo gallery | P4.2, P4.4 | Medium |
| P4.7 | **Education/Learn screen** | Gili | Render educational content sections, illustrated with trichome/stigma images, smooth scrolling | P4.2, P4.3 | Medium |
| P4.8 | **Navigation updates** | Gili | Add Learn tab or section to Home, add lens guide entry point on Camera + Home screens | P4.6, P4.7 | Small |
| P4.9 | **Phase 4 tests** | Etay + Gili | Quality check accuracy tests, content rendering tests | All P4 tasks | Small |

## Assignment Summary

| Person | Tasks | Focus |
|--------|-------|-------|
| **Etay** | P4.1-P4.4 (4 tasks) | Sharpness endpoint, content API, write educational + lens content |
| **Gili** | P4.5-P4.8 (4 tasks) | Camera integration, lens guide screen, education screen, navigation |
| **Both** | P4.9 (1 task) | Testing |

## Work Order

**Weeks 11-12**
- Etay: P4.1, P4.2, P4.3, P4.4 (backend endpoints + content writing)
- Gili: P4.5, P4.6, P4.7, P4.8 (mobile screens + integration)
- Both: P4.9 (testing)
