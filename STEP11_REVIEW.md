# Step 11 review — 2026-10-09

The user accepted the observed game objective. A read-only crop shows **“GEMMA chose Oshawott!”**, which is beyond the introduction. This review sent zero inputs and made zero model calls. The emulator was not moved, resized, or reset.

All **249 tests pass** (16.713 seconds); Python compilation passes. Conditional title presses obtain separate fresh Laya decisions, name-entry observations latch input off, and Laya stop, focus, geometry, deadline, identity, and uncertain-delivery gates remain enforced. These fixture checks do not establish Laya semantic reliability.

The current game scene confirms progress, but the saved Step 11 traces do not establish autonomous completion of character selection and name entry. The latest trace stopped after one completed menu x with `new_game_reentry_forbidden`. User acceptance and recorded autonomous verification remain separate in the [review record](runs/step11-review.json).

## Findings

1. **P1 — Scope needs positive introduction evidence.** `models/intro_check.py:58` accepts any `dialogue` classification. Starting a new introduction job on later gameplay dialogue, including the currently visible starter scene, can therefore authorize x. The name latch only persists within one run. Require positive introduction-stage evidence and add a later-game dialogue refusal fixture before widening live gameplay.
2. **P2 — A transition can stop the job prematurely.** `harness/live_intro.py:61` terminates if an immediate post-input observation still shows the menu, because successful key dispatch has already set `new_game_dispatched`. The latest run demonstrates this path. Add bounded no-input recaptures to allow animation to settle; do not replay an input whose delivery is uncertain.

This was a review only: source behavior was not changed, and no further live progression was attempted. Step 12 has not started.
