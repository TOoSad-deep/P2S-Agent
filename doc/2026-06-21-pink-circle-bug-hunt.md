# Pink Circle Bug Hunt

Test target: `http://localhost:5174/`

Test image: `/Users/douwen/Documents/HUAWEl/AI_Shader_视效生成/paopaoTheme/image/pink_circle.png`

## Summary

This pass used the pink bubble PNG to exercise the running Studio page, the `/png-shader/run` pipeline, candidate inspection, parameter editing, upload validation, the parameterization endpoint, and follow-up Canvas workspace interactions.

## Resolution (2026-06-21)

| Bug | Severity | Status | Notes |
| --- | --- | --- | --- |
| BUG-001 | High | **Deferred (documented)** | Not a selection-policy bug — a color-blind objective metric. Needs a benchmark-backed scoring change; see analysis below. |
| BUG-002 | High | **Fixed** | Added the missing `use_active_model` import. |
| BUG-003 | Medium | **Fixed** | Scalar `#define` rewrite now emits a GLSL float literal; numeric inputs no longer snap transient input to 0. |
| BUG-004 | High | **Deferred (documented)** | Evidence points to a screen-capture artifact, not a render bug; revisit only if reproduced in a clean browser session. |
| BUG-005 | High | **Fixed** | Render resolution accepts either `_render.png` or `_webgl.png` across all three call sites. |
| BUG-006 | High | **Fixed** | `activeDrawId` is now derived from the active run's `draw_session_id`, so a switched-to draw-card run reappears. |
| BUG-007 | Medium | **Fixed** | A stale `?view=canvas` deep link with no run normalizes back to Studio (URL + page). |
| BUG-008 | Low | **Fixed** | A default profile (`updated_at: 0.0`) renders "未更新 Never updated", not the 1970 epoch. |
| BUG-009 | Medium | **Fixed** (layout) | Inspector panel stacks above the Preview dock so the branch-draft Submit stays clickable. Build-verified; visual confirmation pending a browser session. |
| BUG-010 | High | **Fixed** | `refinement_summary.final_score`/`improved` now reconcile to the authoritative post-VLM-gate selected score. |

Verification gate after the round-1 fixes: **1175 pytest + 127 vitest + `npm run build`**, all green.
Round-2 fixes (007–010) gate: **1176 pytest + 129 vitest + `npm run build`**, all green. Each fix landed test-first (RED→GREEN) except the two component-level frontend changes (BUG-003 input guard, BUG-006 derivation), which the repo has no component-test harness for and were verified by `tsc` typecheck + build + reasoning (their trigger states require a full pipeline run with draw-session data the dev server can't synthesize cold).

## Follow-up Completeness Pass (2026-06-21)

Scope: continued Computer Use testing on the running app at `http://localhost:5174/?view=canvas`, focusing on no-run Canvas entry, Studio configuration controls, preference profile UI, Seed GLSL, advanced strategy controls, and Canvas preview artifact availability. No model-consuming run was triggered and destructive preference actions were not clicked.

Pass / stable observations:

- Custom model form opens, shows label/model/base URL/API key fields, keeps "Add" disabled when required fields are empty, and exposes the local-browser API-key storage warning.
- Seed GLSL expands to a textarea and file picker; a short Shadertoy-style `mainImage` snippet can be entered without layout breakage, and disabling Seed GLSL collapses the section.
- Advanced strategy expands and exposes the expected controls: iteration limit, trigger threshold, high-score early stop, minimum improvement, no-improvement patience, optimizer budget, residual layers, VLM review, repair orientation, and protected hierarchy toggles.
- Preference profile panel loads successfully and shows empty positive/negative preference lists with disabled Add buttons until input is provided.
- `selected_render` is now available for both previously failing Canvas runs:

  ```text
  run_aadf9f48 selected_render: 200 image/png 10686
  run_2e21ad10 selected_render: 200 image/png 13100
  ```

Verification run for this follow-up pass:

```text
npm test -- --run src/lib/branchCanvasModel.test.ts src/lib/branchCanvasLayout.test.ts src/lib/glsl-parser.test.ts
Test Files  3 passed (3)
Tests       91 passed (91)
```

## BUG-007: No-run Canvas deep link keeps `?view=canvas` while rendering Studio empty state

Severity: Medium

Repro:

1. Open or remain on `http://localhost:5174/?view=canvas` without an active completed run/result in memory.
2. Observe the page body.

Actual:

- The address bar stays at `?view=canvas`.
- The visible UI is the Studio empty state: upload dropzone, AI/model/self-optimize controls, preprocess/candidate/quality/image-diff empty panels.
- There is no visible Canvas empty state, no Canvas/Studio toggle, and no message explaining why the Canvas view is unavailable.

Expected:

Either the route should normalize back to Studio when Canvas cannot be shown, or the app should show an explicit Canvas empty state with a clear way to start/open a run.

Likely cause:

`frontend/src/App.tsx` derives an `effectivePage` that clamps to `"studio"` when `showCanvas` is false, but the URL query remains `view=canvas`.

## BUG-008: Empty preference profile displays Unix epoch as "Updated At"

Severity: Low

Repro:

1. Open the Preferences panel on an empty/default preference profile.
2. Inspect the read-only metadata.
3. Confirm the profile endpoint returns default metadata:

   ```json
   {"updated_at":0.0,"summary_source_event_count":0}
   ```

Actual:

The UI displays `1970/1/1 (1970-01-01T00:00:00.000Z)` even though there are zero source events and no real update has occurred.

Expected:

Display `Never updated`, `未更新`, or hide the timestamp until `updated_at > 0`.

Likely cause:

`frontend/src/components/PreferencePanel.tsx` calls `relativeTime(profile.updated_at)` and `new Date(profile.updated_at * 1000).toISOString()` unconditionally, while the backend default profile intentionally uses `updated_at: 0.0`.

## Model + Human Canvas Pass (2026-06-21)

Scope: continued Computer Use testing focused on configured model calls, human-directed branch operations, and the Canvas inspector / preview sidebar. This pass used the same `pink_circle.png` via the real browser upload picker.

Pass / stable observations:

- `/api/models` returned three configured image-capable presets: Claude Opus, MiMo v2.5, and Qwen 3.7 Plus.
- Studio upload through the macOS file picker worked for `pink_circle.png`.
- AI Auto + Claude Opus created `run_db5348e8`, produced a real `llm_0` candidate with `llm_io`, selected the LLM GLSL, and completed successfully.
- `run_db5348e8` final summary: `completed`, selected `llm_0`, final score `0.8023181124854739`, quality band `good`, refinement history length `2`.
- Canvas run-node selection updates the Inspector, exposes title/Favorite/Continue actions, and the Preview dock loads both the reference image and node render.
- Continue-from-final opens a branch draft in the Inspector without immediately submitting; the draft exposes feedback text, Refine/Polish/Continue modes, layout/palette/background/small-edit locks, Explore variants, Batch Draw, Fine controls, Cancel, and Submit.
- Human-directed branch submit created `run_5d879ae2` from `run_db5348e8` with feedback `保持粉色玻璃球整体结构，增强左上高光和边缘透明感，背景保持白色。`.
- The human branch completed and retained LLM evidence in `refinement_history[0].llm_io`; its selected render endpoint works:

  ```text
  run_5d879ae2 selected_render: 200 image/png 13185
  ```

## BUG-009: Preview dock can cover the branch-draft Submit button in the Canvas sidebar

Severity: Medium

Repro:

1. Open Canvas for `run_db5348e8`.
2. Select the active run node so the Preview dock opens.
3. Click `Continue from final`.
4. Enter branch feedback in the Inspector.
5. Try to click `运行分支 Submit` while the Preview dock remains expanded.

Actual:

The Preview dock overlaps the lower part of the Inspector and covers the Submit area. The first click on the accessible `运行分支 Submit` element did not trigger submission. After collapsing the Preview dock, the same Submit button became visibly accessible and the branch submitted successfully.

Expected:

The Inspector action footer should remain reachable without requiring users to collapse Preview, or the Preview dock should avoid overlapping the Inspector / branch form.

Likely area:

Canvas side panels / Preview dock positioning and z-index. The draft form footer needs reserved space, scrolling, or layout coordination with the dock.

## BUG-010: Human-directed branch score fields disagree after a rejected LLM refinement

Severity: High

Repro:

1. Complete an AI Auto run (`run_db5348e8`) on `pink_circle.png`; final score is `0.8023181124854739`.
2. In Canvas, select the run, choose `Continue from final`.
3. Enter feedback: `保持粉色玻璃球整体结构，增强左上高光和边缘透明感，背景保持白色。`
4. Enable `保持构图 Layout` and `保护背景 Background`.
5. Submit the branch and wait for `run_5d879ae2` to complete.

Observed API evidence from `run_5d879ae2`:

```json
{
  "selected_candidate_id": "seed_0",
  "selected_candidate_score": 0.7341226787398317,
  "quality_final_score": 0.7341226787398317,
  "objective_simple_ssim": 1,
  "refinement_summary": {
    "initial_score": 0.8023181124854739,
    "final_score": 0.8023181124854739,
    "improved": false
  },
  "history": [
    {
      "score_before": 0.8023,
      "score_after": 0.7683,
      "delta": -0.034,
      "accepted": false,
      "best_score_after": 0.8023
    }
  ]
}
```

Observed UI evidence:

- Canvas header shows `SSIM: 100%`.
- Active child run node shows `score 0.734`.
- Refinement proposal node shows `0.768` with a down arrow and `0.034`.
- Current best node shows `0.734`.

Expected:

If the LLM refinement proposal was rejected and `best_score_after` remained `0.8023`, the branch's selected/final score and Current best should stay aligned with the retained best artifact. If the branch intentionally selected the lower-scoring seed/optimized result, the refinement summary should not claim final score `0.8023`.

Likely area:

GLSL branch refinement / optimizer bookkeeping: selected candidate metrics, top-level `quality_router`, `refinement_summary`, and Canvas node model are reading inconsistent score sources after a rejected refinement.

### BUG-002 — fix

`backend/app/api/routers/core.py` now imports `use_active_model` from `p2s_agent.config`. The same latent `NameError` also affected the human-refine endpoint (`core.py:315`), which the original report did not flag — the single import fixes both. Regression test: `test_parameterize_resolves_use_active_model` in `backend/tests/unit/test_router.py`.

### BUG-003 — fix

Two layers were addressed:

- Root cause of `'*' : wrong operand types`: `updateShaderParam` (`frontend/src/lib/glsl-parser.ts`) wrote scalar `#define` values via `String(newValue)`, so a whole number became an **int** literal (`0`), which breaks GLSL ES float math. It now formats scalars through `formatComponent`, matching the vec path (`0` → `0.0`). This fixes the compile error for *every* whole-number scalar, not just the empty-input case. Tests in `frontend/src/lib/glsl-parser.test.ts`.
- Trigger: the numeric inputs in `PngShaderParamPanel.tsx` used `parseFloat(e.target.value) || 0`, snapping empty/intermediate input to 0. They now ignore non-finite parses, so transient input never rewrites the GLSL.

### BUG-005 — fix

Root cause: the scoring pipeline writes a candidate's render as `candidates/<id>_render.png` (DSL-rasterized) **or** `candidates/<id>_webgl.png` (GLSL scored via WebGL, `core/pipeline/scoring.py`), but three resolvers hard-reconstructed only `_render.png`. Added a shared helper `candidate_render_relative()` in `p2s_agent/orchestration/checkpoints.py` that prefers `_render.png`, falls back to `_webgl.png`, and defaults to `_render.png` when neither exists (stable 404 target, security checks preserved). Now used by `resolve_checkpoint_artifact` (both render branches), `sessions._resolve_run_render`, and the fusion region-metrics path. Tests in `test_checkpoints.py` and `test_router.py`.

### BUG-006 — fix

Root cause confirmed: `branchCanvasModel.ts` skips any run carrying a `draw_session_id` (expecting `buildDrawSessionModel()` to reintroduce it), but `BranchCanvasWorkspace` never derived `activeDrawId` from the branch tree — so a within-tree switch to a draw-card run left it invisible until a live draw action set the pointer. Added an effect in `BranchCanvasWorkspace.tsx` that derives `activeDrawId` from the active run's `draw_session_id`, triggering the draw-session fetch so its cards render.

### BUG-001 — deferred, root-cause analysis

The selection policy is **not** at fault: `select_best_candidate` (`backend/p2s_agent/core/pipeline/pool.py:349`) correctly picks the highest `final_score`. The defect is upstream in scoring — `final_score` (from `compute_objective_metrics` / the quality router) is effectively **color-blind**: a flat white circle on black is structurally/luminance-similar to the pink circle on black, so it scores ~0.44 and wins despite discarding the subject's color. A real fix requires adding a color/palette-fidelity term to the objective metric (or a palette-divergence penalty), which changes selection for **every** run and must be validated against the golden/benchmark set before merging. Treat as a tracked follow-up, not a surgical patch.

### BUG-004 — deferred, assessment

The React accessibility tree showed the Canvas fully mounted (header, nodes, inspector, edges) and the view became visible again after refocusing the tab; only the *captured* window was blank, alongside `SCStreamErrorDomain Code=-3811` capture errors in the tooling note. This is most consistent with a macOS ScreenCaptureKit capture artifact rather than an app render bug. The hidden-parent React-Flow-init hypothesis is plausible but unproven; a defensive "re-measure on visible" effect could be added if the blank state is ever reproduced in a clean browser session (no screen-capture tooling involved).

### BUG-007 — fix

`frontend/src/App.tsx` clamped `effectivePage` to `studio` when no run exists but left `?view=canvas` in the address bar. Added an effect that, when Canvas cannot be shown, normalizes both the URL (`history.replaceState` to the bare pathname) and the page state back to Studio.

### BUG-008 — fix

`frontend/src/components/PreferencePanel.tsx` rendered `updated_at: 0.0` (the default-profile sentinel) as the Unix epoch. `relativeTime` now returns "未更新 Never updated" for `epoch <= 0`, and the ISO timestamp span is hidden when `updated_at <= 0`. Unit test in `PreferencePanel.test.ts`.

### BUG-009 — fix (layout)

The Inspector (`Panel position="top-right"`) and Preview dock (`Panel position="bottom-right"`) are sibling React Flow panels with equal z-index; the later-rendered dock intercepted clicks on the branch-draft Submit when a tall draft form pushed the Inspector's footer into the bottom-right overlap zone. The Inspector panel now carries `zIndex: 6` (above the dock's default 5) so its footer wins pointer events. When the Inspector is short (no draft) the panels don't overlap, so the dock stays fully visible. Build/typecheck-verified; visual confirmation pending a real browser session.

### BUG-010 — fix

Root cause: the VLM final gate (`backend/p2s_agent/core/pipeline/graph.py`, in `_run_post_pipeline`) blends a semantic score into `selected.final_score` **after** `refinement_summary` recorded the objective refinement trajectory. So the selected/quality score the UI shows (blended, e.g. 0.734) disagreed with `refinement_summary.final_score` (objective best, e.g. 0.8023) for every VLM-judge-enabled run — most visibly after a rejected refinement. The fix reconciles `refinement_summary["final_score"]` to the authoritative `selected.final_score` and recomputes `improved` against the summary's own `initial_score`, immediately after the VLM gate. Regression test: `test_run_post_pipeline_reconciles_refinement_summary_with_vlm_final_gate` in `backend/tests/unit/test_graph.py`.

## BUG-001: Non-LLM pipeline drops the detected pink bubble appearance

Severity: High

Repro:

1. Submit `pink_circle.png` to `/png-shader/run` with `llm.mode=off` and `quality.refinement_mode=off`.
2. Poll `/png-shader/status/{run_id}` until `completed`.

Evidence from run `run_be5e6c1f`:

- Preprocess detected the expected palette: `#F8F8F8`, `#F8D8E8`, `#F8E8F8`, `#F898B8`, `#F84878`.
- Preprocess detected `gradient_score=0.9834` and `photo_like_score=0.7072`.
- Selected candidate was `cv_0`, score `0.44396`, quality router `preview/poor`.
- Selected GLSL only emitted a white solid circle:
  - `L0_fill_r 0.972549`
  - `L0_fill_g 0.972549`
  - `L0_fill_b 0.972549`
  - no pink gradient, rim, halo, highlight, or shadow terms.

Expected:

The deterministic or fallback candidate should use the detected dominant pink palette and gradient/highlight evidence, or avoid selecting a candidate that visibly discards the subject's core appearance.

Actual:

The selected output is essentially a flat white circle on black, despite preprocessing having the correct color/gradient signals.

Likely area:

- Candidate generation/scoring handoff from preprocess features to CV/rule/fallback candidates.
- Selection policy may overweight shape match and underweight color/gradient/highlight loss.

## BUG-002: Parameterize endpoint returns 500 because `use_active_model` is not imported

Severity: High

Repro:

1. Complete a run and save `selected_glsl`.
2. Call:

   ```bash
   curl -X POST http://localhost:5174/png-shader/parameterize/run_be5e6c1f \
     -F 'glsl=</tmp/pink_selected.glsl'
   ```

Actual:

HTTP 500:

```json
{"detail":"Parameterization failed: name 'use_active_model' is not defined"}
```

Expected:

The endpoint should either return parameterized GLSL or a controlled LLM/service error.

Likely cause:

`backend/app/api/routers/core.py` uses `use_active_model(store._get_run_model(run_id))` inside `parameterize_png_shader`, but `use_active_model` is not imported in that module.

## BUG-003: Numeric parameter editing can corrupt preview into shader compile error

Severity: Medium

Repro:

1. In Studio, use the pink bubble result with a parameterized LLM/refinement output visible.
2. Edit `CENTER X` in the Parameters panel.

Observed via Computer Use:

- The UI changed `CENTER X` from `0.500` to `0.000` during editing.
- The preview label changed to `edited`.
- Shader output displayed:

  ```text
  着色器错误 Fragment shader is not compiled.
  ERROR: 0:114: '*' : wrong operand types
  ```

Expected:

Transient input states should not immediately rewrite GLSL to invalid values, and invalid edits should be rejected, clamped, or revertible without breaking the preview.

Actual:

The edited GLSL can become non-compilable and the preview remains in an error state.

Likely area:

- `frontend/src/components/PngShaderParamPanel.tsx` numeric inputs call `handleChange(parseFloat(e.target.value) || 0)`, which converts empty/invalid intermediate input to `0`.
- Parameter edits are applied directly to GLSL before validation.

## BUG-004: Canvas workspace mounts data but renders as an empty purple viewport

Severity: High

Repro:

1. Open a completed pink-bubble run in Chrome at `http://localhost:5174/`.
2. Click the top-level `画布 Canvas` toggle.
3. Confirm the URL becomes `http://localhost:5174/?view=canvas`.

Observed via Computer Use:

- Chrome accessibility tree shows the Canvas page is mounted:
  - top-level `画布 Canvas` button is focused.
  - Canvas sub-toggle exists with `列表 List` and `画布 Canvas`.
  - React Flow content exists with edges like `Edge from input to run:run_aadf9f48`.
  - Node labels exist, including `Input PNG`, `Selected baseline`, `Iteration 1 proposal`, `Current best`, draw-session cards, inspector, fusion builder, and preview dock.
- The actual captured Chrome window content is a flat purple/blank rectangle with no visible header, graph nodes, inspector, or controls.
- Clicking a canvas node accessibility element such as `Iteration 1 proposal` did not update the inspector; it stayed at `选择一个节点查看详情 Select a node`.

Expected:

The Canvas page should visibly render the graph, toolbar, inspector, preview dock, and allow node selection.

Actual:

The Canvas DOM/accessibility content exists, but the visible window is blank and node selection feedback does not occur.

Likely area:

- `frontend/src/App.tsx` keeps Studio and Canvas both mounted and toggles them with `display: none`. React Flow is sensitive to hidden/zero-size initialization; it may need a resize/fitView after becoming visible, or Canvas should mount only when active.
- `frontend/src/components/BranchCanvas.tsx` accepts `onSelectionChange` but `BranchCanvasWorkspace` does not pass it; selection currently depends on node click reaching React Flow.
- `frontend/src/components/BranchCanvasWorkspace.tsx` / React Flow sizing should be checked for hidden-parent initialization and pointer-target overlays.

Evidence narrowing:

- `/png-shader/runs/run_be5e6c1f/branches` returned a valid branch tree.
- `/png-shader/runs/run_be5e6c1f/timeline` returns a wrapper object with `timeline`, and `fetchTimeline` correctly unwraps it, so this is not a timeline-shape bug.
- `/png-shader/runs/run_be5e6c1f/artifacts/selected_render` and `/png-shader/runs/run_be5e6c1f/artifacts/checkpoint:final:selected:render` both returned HTTP 200 PNG data.
- Follow-up note: after refocusing the P2S Chrome tab, the same Canvas view did become visible and node selection worked. Treat this as intermittent/window-visibility-sensitive until it is reproduced in a clean browser session.
- Focused frontend tests passed:

  ```bash
  npm test -- --run src/lib/branchCanvasModel.test.ts src/lib/branchCanvasLayout.test.ts
  ```

  Result: 2 test files, 84 tests passed.

## BUG-005: Canvas previews fail because `selected_render` resolves to missing `*_render.png` files while runs produced `*_webgl.png`

Severity: High

Repro:

1. In Canvas `列表 List`, select branch run `run_2e21ad10`.
2. Observe the Compare strip.
3. Directly request the active and parent selected render artifacts:

   ```bash
   curl http://localhost:5174/png-shader/runs/run_2e21ad10/artifacts/selected_render
   curl http://localhost:5174/png-shader/runs/run_aadf9f48/artifacts/selected_render
   ```

Observed via Computer Use:

- Compare showed `无预览 / No preview` for both `当前 / Current` and `父级 / Parent`.
- Canvas/Fusion thumbnails also rely on `/artifacts/selected_render`, so the same missing artifact path affects multiple canvas preview surfaces.

API evidence:

```json
{"detail":"artifact not found: seed_0_render.png"}
```

for `run_2e21ad10`, and:

```json
{"detail":"artifact not found: llm_0_render.png"}
```

for `run_aadf9f48`.

Filesystem evidence:

- `backend/test_results/2026-06-20_png-shader_single_run_2e21ad10/candidates/seed_0_webgl.png` exists, but `seed_0_render.png` does not.
- `backend/test_results/2026-06-20_png-shader_single_run_aadf9f48/candidates/llm_0_webgl.png` exists, but `llm_0_render.png` does not.
- Other candidates in the same run use `*_render.png`, so artifact naming is inconsistent across render backends/sources.

Expected:

`selected_render` should serve the selected candidate render regardless of whether the actual file is named `*_render.png` or `*_webgl.png`, or the pipeline should normalize the written artifact name.

Actual:

The selected render endpoint is 404 for valid completed runs with previewable selected candidates, causing Canvas Compare/Preview/Fusion thumbnails to show empty previews.

Likely cause:

`backend/p2s_agent/orchestration/checkpoints.py` `resolve_checkpoint_artifact()` always maps candidate render artifacts to `candidates/{candidate_id}_render.png`; the scoring pipeline can write `candidates/{candidate_id}_webgl.png`.

## BUG-006: Canvas hides the active draw/variant branch after switching to it

Severity: High

Repro:

1. In Canvas `列表 List`, click branch `run_2e21ad10` (`conservative`, score `0.810`).
2. Confirm the header changes to `1 candidates` and `SSIM: 86%`, proving the active run switched.
3. Switch back to Canvas sub-view `画布 Canvas`.

Observed via Computer Use:

- Left toolbar shows active run short id `2e21ad10` and status `completed`.
- The visible graph only contains:
  - `Input PNG`
  - root run `aadf9f48`
  - unrelated branch `2a18ec59`
- The active run `2e21ad10` is not visible anywhere in the graph.
- The other draw/variant sibling runs `30d406ed`, `6803c8a6`, and `32a06481` are also absent.

API evidence:

`/png-shader/runs/run_2e21ad10/branches` correctly returns:

- `active_run_id: "run_2e21ad10"`
- root child runs `run_2e21ad10`, `run_30d406ed`, `run_6803c8a6`, `run_32a06481`, and `run_2a18ec59`.

Expected:

The Canvas graph should always show the active run, especially after the user switches to that run from the branch list. Draw/variant runs should appear as variant or draw-card nodes, or at minimum the active branch should have a visible fallback node.

Actual:

The active run can disappear from the graph even though the toolbar and list view indicate it is selected.

Likely cause:

- `frontend/src/lib/branchCanvasModel.ts` excludes runs with both `variant_group_id` and `draw_session_id` from the regular and variant passes.
- Those runs are expected to be reintroduced through `buildDrawSessionModel()`, but `BranchCanvasWorkspace` only renders that model when `drawSession` is loaded.
- After switching/reloading, `activeDrawId`/`drawSession` is not derived from the branch tree's `draw_session_id`, so draw-card runs vanish.

## Checked Non-Bug

Invalid image upload is guarded correctly:

```bash
curl -X POST http://localhost:5174/png-shader/run \
  -F 'image=@README.md;filename=fake.png;type=image/png'
```

Returned HTTP 422:

```json
{"detail":"uploaded file is not a valid image"}
```

## Tooling Note

Computer Use could read the Chrome accessibility tree and send keyboard input, but coordinate click / set-value operations intermittently failed with macOS ScreenCaptureKit errors such as:

```text
SCStreamErrorDomain Code=-3811
```

Keyboard navigation and page state reading were still usable for the UI observations above.
