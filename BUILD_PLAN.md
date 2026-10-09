# Pokémon Black CUA harness: build and verification plan

Status: the user requested skipping Steps 12–13 and switching to the existing
Pokémon White save. Current implementation supports bounded walking and read-only
party/Bag inspection; live verification is in progress. Steps 1–10 and the Step 11
review below retain the historical Black checkpoint evidence. A complete autonomous
Black introduction trace remains unverified. Laya semantic reliability remains
unvalidated after Step 8.

The user requested keeping the emulator at its current position and size for
demonstration and stable FPS. Step 3 did not move, resize, zoom, or change emulator
settings. Geometry-failure checks use simulated windows; an actual position change
also triggered refusal and was handled by recalibrating in place.

Current user preference (Step 9): one prompt starts a bounded autonomous job,
with no routine human review. The system should enter the game and pursue an
objective such as getting the first Pokémon, or play for a specified duration.
This replaces the earlier preview/manual-only preference and human-review pause
policy. Numbered development checkpoints remain separate from runtime decisions.

## Outcome

Build a local computer-use agent harness that captures the melonDS game display,
sends images to hosted Gemma through the Gemini API, validates Gemma's proposed
GUI actions, evaluates text decision context with local fine-tuned Laya, and
executes permitted mouse and keyboard input through PyAutoGUI.

The generic executor supports cursor movement, clicking at explicit coordinates,
keyboard presses, bounded text typing, and waiting. Pokémon name entry uses
touchscreen clicks. Generic typing is verified separately in a disposable local
text field because game input is mapped to DS buttons.

The first live integration checkpoint is one bounded action. Later checkpoints
progress through character/name entry and first-Pokémon acquisition, or timed
play, according to the prompt. Boy and HUY remain introductory defaults when the
prompt does not specify choices. Full-game completion and additional fine-tuning
are outside this milestone.

## Existing setup and evidence

- Workspace: `/Users/adamvu/pokemon-cua`.
- Python project and dependency lockfile exist; `main.py` now runs prompt-driven
  configuration preflight.
- `.env` exists and is now excluded by `.gitignore`. Step 1 reads known settings
  without displaying the API key or changing the file.
- ROM supplied by the user:
  `/Users/adamvu/Pokemon_retro/nintendo_ds/pokemon_black/Pokemon - Black Version (USA, Europe) (NDSi Enhanced).nds`.
- User screenshots show melonDS 1.1 at the Pokémon Black title screen.
- Laya checkpoint:
  `/Users/adamvu/laya-project/laya/laya_finetuned_typed_decisions`.
- Checkpoint metadata identifies `answerdotai/ModernBERT-large`, typed decision
  heads, a maximum sequence length of 1024, and a head length of 256. State capacity
  depends on the questions included; a fixed character limit is insufficient.
- Laya source examples expose local `laya.load(...)` and
  `agent.predict(state, questions)` with typed answers and probability distributions.
  Use that loader rather than a generic text-generation pipeline.
- Saved training metrics are explicitly in-sample calibration metrics, not
  independent evidence of performance on CUA tasks.

## Responsibilities and action contract

```text
Emulator image → Gemma proposal → deterministic validation
                                 → Laya text decisions → harness policy
                                 → PyAutoGUI → fresh emulator image
```

Gemma perceives images, chooses generic input actions, and checks visible outcomes.
Laya evaluates serialized text context, never screenshots. The harness owns input
validation, focus and geometry checks, observation freshness, limits, retries,
logging, routing, and termination. Laya cannot override a hard block.

Initial action types:

- `MOVE(x, y)`: emulator-relative normalized coordinates.
- `CLICK(x, y)`: explicit target; executor moves and clicks as one bounded action.
- `PRESS(key, duration)`: one permitted physical key, with bounded duration.
- `TYPE(text)`: bounded literal text in a designated text-input environment.
- `WAIT(seconds)`: bounded delay.

Coordinates range from 0 to 1000 inclusive and map to valid pixels within a
calibrated rectangle, including the last pixel rather than one beyond its edge.
Capture pixels and desktop input coordinates must be calibrated separately for
Retina scaling. Touch input is restricted to the verified lower-screen rectangle.
Proposed initial limits: one state-changing action per observation, a maximum
one-second key hold, a maximum two-second wait, and a maximum 32-character typing
action. Tune explicit settings only after measuring emulator behavior.

Physical key mappings from the user's screenshot:

| DS button | Physical key |
| --- | --- |
| A | X |
| B | Z |
| X | S |
| Y | A |
| L / R | Q / W |
| Start | Return / Enter |
| Select | Left Shift |
| Directions | Num arrow bindings; verify actual injected events |

## Laya semantics

Use the existing agent-trace questions as the initial schema: `action`,
`needs_review`, `outcome`, `risk`, and `urgency`. Do not assume that this checkpoint
was trained to answer the three new binary questions from the pasted proposal.

The saved `action` labels mean:

| Trained answer | Initial harness policy |
| --- | --- |
| continue | Eligible for execution if all other gates pass |
| observe | Eligible for execution; flag the trace for later sampling |
| human_review | Fresh automatic capture/replan and consistency check; bounded recovery |
| stop | Halt without executing the pending action |

The checkpoint's trained `human_review` definition still says to queue a trace
for review. The autonomous routing is harness policy chosen to honor the user's
request: after a fresh replan, a repeated exact action on the same claimed screen
state can proceed if its new decision and hard checks permit it. No person is
asked to approve it. Changed evidence requires another check. Recovery exhaustion
stops safely; it never escalates to a human-review queue.

`observe` retains its continue-and-sample meaning. Screenshot freshness is always
a harness rule. `outcome` describes the trace and is not independent success proof.
Preserve raw predictions and probabilities separately from policy. Unknown output,
inference failures, tied action probabilities, serializer loss, and tokenizer
truncation trigger bounded automatic recovery without input. No CUA-calibrated
confidence cutoff has been established; ties only test argmax uniqueness.
The automatic check does not fix or validate the checkpoint's semantic accuracy.

## Step-by-step checkpoints

At each implementation checkpoint, show the result, test evidence, and remaining
uncertainty, then wait for the user to request the next build step. Runtime jobs
contain no human approval pauses. A failed required test keeps its build step open.

### 1. Configuration and project foundation

Build typed settings, explicit runtime modes, minimal modules, and ignored paths
for `.env`, emulator assets, checkpoints, screenshots, and run artifacts. Reference
the existing Laya source and checkpoint without copying or editing them. Start
with no input execution and no automatic run loop.

Test: required settings fail clearly when missing; diagnostics never print the
API key; `.env` is ignored by Git; configured paths exist; dependency imports work.
Inspect local loader compatibility before changing dependencies.

Pass: reproducible startup with sanitized diagnostics and no desktop input.

Implemented: `harness/settings.py`, `harness/cli.py`, the `main.py` entry point,
`.env.example`, ignored secret/artifact paths, usage documentation, and
`tests/test_foundation.py`. Verified against the existing local `.env`, ROM,
and checkpoint. Core non-desktop dependency imports passed. Preflight intentionally
does not import PyAutoGUI or model runtimes, and local Laya installation remains
Step 7. Package availability is reported separately from runtime compatibility.

Test evidence: 14 foundation tests passed, along with real prompt/check startup,
startup outside the project directory, Git ignore checks, actual API-key redaction
on a validation error, and a runtime-import guard. No model calls, screenshots,
emulator launches, or desktop input were performed. In the sandbox, the existing
virtual environment was used with `uv run --locked --no-sync --no-cache` to avoid
the inaccessible global uv cache.

### 2. Capture and geometry calibration

Build emulator-window discovery or verified manual calibration, with rectangles
for the complete game display and lower touchscreen. Store capture scale, desktop
position, dimensions, observation ID, and timestamp. Crop both DS screens while
excluding desktop content and window chrome.

Test: save and inspect one capture; verify both screens are readable; test
normalized corners and center against desktop coordinates without clicking;
move/resize the emulator and confirm stale calibration is rejected or refreshed.

Pass: accurate capture and input mapping, including Retina scaling.

Implemented: `computer/geometry.py`, `computer/capture.py`, and CLI calibration,
confirmation, and prompt-driven single-capture operations. Window-only native
macOS capture excludes desktop content; PyAutoGUI is reserved for later input
execution. Calibration stores the actual capture scale, frame and screen bounds,
window identity, timestamp, and review status. Captures get unique observation
IDs, timestamps, prompt text, and coordinate samples. Legacy `.env` rectangles
are preserved but are not used for window capture.

Live evidence: the supplied ROM was opened through the emulator file dialog
without advancing gameplay. A visually reviewed capture measured 1074×1608 game
pixels at 2× Retina scale, with two 1074×804 screen rectangles. A real window zoom
caused stale-calibration rejection before saving an observation; the window was
then restored and prompt-driven capture passed again. All 30 foundation and
capture tests passed. A live import guard confirmed that capture imports no
model or input runtime. Synthetic tests additionally exercise moves, restarts, changed
capture scale, movement during capture, crop edges, normalized corners/center,
ambiguous windows, permissions, CLI errors, and prompt-secret redaction.

Scope limit: the profile assumes unrotated, equal vertical screens with the lower
screen as the touchscreen. The current melonDS preferences show no screen swap,
no rotation, and no gap. Layout changes without a geometry change require manual
recalibration; they are not all automatically detected by this prototype.

### 3. Deterministic executor and manual control checks

Build the generic action schema and validator before enabling manual execution.
Use explicit click targets, permitted keys, bounded durations, focus checks,
geometry checks, a stop mechanism, and key release in cleanup. Execution requires
the intended target to be foreground; stop instead of sending input elsewhere.
TYPE is disabled in the game profile and enabled in a disposable text-input test
profile. Use no shell commands as agent actions.

Test: simulated input backend rejects missing fields, extra fields, unsupported
keys, out-of-range coordinates, wrong screen click targets, excessive durations,
lost focus, and changed geometry. Live manual checks verify cursor placement,
one Return press, one lower-screen click where appropriate, and exact literal
typing in the disposable field. Verify the Num arrow bindings separately.
Interrupt a held input and verify every key is released.

Pass: all generic capabilities work and invalid actions emit zero input events.
Restore the agreed benchmark start state before later gameplay checks.

Implemented: strict action schemas, preview-by-default manual requests,
explicit one-action execution, bounded key/click holds, verified cursor placement,
native focus/dialog/geometry checks, persistent stop signals, and release cleanup
that still runs after a fail-safe or Ctrl+C. TYPE is blocked in the game profile.
An owned disposable field tests printable ASCII typing and stop/release behavior.
No model inference or autonomous gameplay loop is present.

Live evidence: Accessibility access for the Python host app now passes. The
executor reached desktop point (654, 695) for normalized (500, 750), sent one
bounded click on the blank lower touchscreen, and verified the mouse button was
released. A Return press opened the main menu. Down changed the highlight from
New Game to Mystery Gift; Up returned it to New Game. The configured Num arrow
bindings therefore accept these injected Up/Down events. DS B (Z) restored the
title screen without starting New Game. WAIT passed with no input. Final native
checks showed Return, Up, Down, B, and Shift released and unchanged calibrated
bounds (386, 64, 538, 833). A real foreground mismatch blocked input.

The initial Tk field was unreliable even with apparent focus. It was replaced
with an owned native macOS text field and event dispatch, with a bounded wait
for native focus and stable bounds. Two successive live tests received exactly
`HUY 42!`, interrupted a real Shift hold, and verified release. All 54 automated
tests passed, covering validation, focus, geometry, permissions, stop signals,
Ctrl+C, fail-safe cleanup, failed cursor placement, and owned-field readiness.

Scope limits: the blank lower-screen click proves bounded event delivery and
release, not selection of a game control. Visible touchscreen selection will be
verified in name entry. Left/Right game responses remain to be verified on a
screen with horizontal navigation. No Gemma or Laya calls were made. Durable
model/action traces are still Step 6; current manual commands report their result
to the console, while captures save separate observation metadata.

### 4. Gemma perception only

Build a bounded Gemini API adapter. Send a cropped game image and request a short
screen description; execute no model output. Apply timeouts and limited retries.

Test: real API requests recognize the live Black title sequence and a supplied
title-screen fixture with PRESS START, including correct screen attribution.
Simulated missing credentials, API errors, and empty responses produce clear
errors without input or unbounded retries.

Pass: verified image understanding through the configured model and account.

Implemented: `models/perception.py`, prompt-driven `--perceive`, and
`tests/test_perception.py`. One capture preserves both DS screens in the saved
game image. The adapter sends two labeled, native-resolution PNG screen crops
from that same frame, with their rectangles in the complete image. It strips
embedded screenshot metadata and excludes desktop/window chrome. No emulator
layout, sizing, or position changes are needed. The configured Gemma model is
called through the Gemini API using the existing key; no Laya runtime or input
executor is imported by perception.

API attempts have a 25-second wall timeout and at most one retry, after one
second, for transient transport/timeout/selected HTTP failures. SDK retries are
disabled. Output is limited to 600 tokens; empty, blocked, truncated, or function
responses fail clearly. Descriptions and sanitized failures are saved separately
from capture metadata, with source kind, observation ID, screen rectangles,
attempts, and `executed: false`. Ctrl+C records cancellation with an unknown API
attempt count rather than claiming no remote request occurred.

Live evidence: the configured `gemma-4-26b-a4b-it` read “Adventures Await” and
identified a black lower screen in 7.233 seconds. Another live request read
“Pokémon BLACK VERSION” on the lower screen and described the upper creature
artwork in 4.560 seconds, correctly reporting no visible button prompt. These
initial requests used one combined image. The supplied game-only title fixture
was sent after explicit user approval. Its first request hit HTTP 500 and stopped
after two attempts. A subsequent request read PRESS START but incorrectly merged
the screens and called the lower screen blank. That result was preserved as a
failed screen-attribution check.

The final two-image adapter corrected this failure: a real fixture request took
9.493 seconds and correctly assigned the title and PRESS START to the upper
display, and the creature artwork plus “Developed by GAME FREAK inc.” to the
lower touchscreen. The fixture is labeled `user_supplied_fixture`, with no claim
of a current capture timestamp. No model response reached input execution.
Manual benchmark setup requests were refused by the focus guard; no Start/B
keypresses were sent during Step 4. The game remains in its title/intro cycle.

All 70 automated tests passed, including original-resolution pixels from both
screens, explicit image labels, secret redaction, no input/Laya imports,
credential/model errors, transient retry bounds, cancellation of stalled
requests, invalid images, empty/blocked/truncated responses, and CLI cancellation.
These checks verify this perception checkpoint, not general gameplay reliability.
Model descriptions remain claims to compare with the saved image; structured
actions and independently verified game outcomes remain later steps.

### 5. Structured Gemma proposals

Add a typed proposal containing observation ID, visible observation, intended
action, expected result, and a brief rationale. Include physical key mappings.
Initially permit one action per turn. Reject invalid output rather than guessing
coordinates or silently repairing action meaning.

Test: title-screen proposal uses physical `x` (DS A) as the user instructed,
including when the visible prompt says PRESS START. Test explicit lower-screen
targets with synthetic name-entry proposals; real name-entry images and click
effects are deferred to Step 12 because the game has not reached that screen.
Test malformed JSON,
missing action fields, unsupported actions, stale observation IDs, non-finite
values, and an unsupported completion claim.

Pass: valid proposals are inspectable; invalid proposals cannot reach execution.

Implemented and verified on 2026-10-09. `--propose` captures one frame, sends two
labeled screen crops to Gemma, and validates a strict proposal containing the
observation ID, per-screen description/text, screen kind, one action or pause,
explicit mouse target, expected result, rationale, and `task_complete: false`.
The schema is supplied in the prompt and enforced locally; no provider schema
support is assumed. Malformed output is rejected without semantic repair.

The operator-reported title behavior is encoded as one short physical `x` press
per opening/title proposal, requiring a fresh observation for every further
proposal. The main menu and unknown screens require pause/no action; New Game
or Continue labels cannot be paired with a contradictory title/opening claim.
The checkpoint does not select New Game or an existing save. A pause does not
establish task completion. This is a policy over model claims, not independent
OCR or proof of visual grounding.

All 90 tests passed. Coverage includes duplicate keys, fenced/malformed JSON,
missing/extra/wrong-type fields, repeats/chords/action lists, unsafe tools,
durations and coordinates, wrong-screen clicks, stale IDs, unsupported success,
menu pauses, synthetic letter targets, sanitized records, API failures and
cancellation, and CLI imports that prohibit the executor and local model runtime.
Adapter Python source is retained by Git while generated model artifacts remain
ignored; the earlier broad `models/` ignore rule was corrected.

Real API evidence:
- Live capture `309a8fdfeb834d1cb087ac109ce404ed`: title screen with PRESS START
  and lower Reshiram/credit correctly described; one PRESS x, 0.1 seconds,
  validation passed, one API attempt, 8.14 seconds.
- Approved title fixture `title-proposal-398f14c89fa24c51a77eb3ac37b2ee84`:
  one PRESS x, 0.1 seconds, validation passed, two attempts, 20.661 seconds.
- Initial fixture `title-proposal-e44009e880da4e13bfc3afcfcc82ee4e` chose enter;
  validation rejected it. The title override was made prominent in the system
  instruction and request before new tests, with the rejected record preserved.
- Live `1ff8865d90af44c286537d8b7880110a` encountered transport failure and
  stopped after two attempts; no validated proposal or success was recorded.

Every record has `executed: false` and `laya_called: false`. No game controls,
window settings, or input focus were changed. The real title transition from an
x press, visual main-menu recognition, and real name-keyboard target grounding
remain later gameplay checks. Verification index: `runs/step5-verification.json`.

### 6. Dry-run pipeline and durable logs

Connect capture, planning, validation, and logging in a single-turn dry run. Record
run IDs, images, proposed/blocked/executed status, model configuration, latency,
expected effects, observed effects, and error details. Actual outcomes are
recorded only after the next observation.

Test: run against the title screen and inspect the trace. A spy executor receives
zero calls. A dry-run proposal is never recorded as executed or as successful.

Pass: complete reviewable evidence with no desktop input.

Implemented and verified on 2026-10-09. `--dry-run` performs one fresh capture,
one bounded hosted Gemma planning request, strict proposal validation, and an
atomic run trace. It does not import the executor, PyAutoGUI, or a local Laya
runtime, and cannot be combined with execution or another operation.

The trace has a run ID separate from its observation ID, artifact paths,
calibration snapshot, game-image checksum, model configuration/instruction
checksum, request metrics, stage/total timings, ordered events, validation,
model claims, predicted effects, and sanitized errors. Harness facts remain
separate: execution is disabled with zero calls, task outcome is not assessed,
and observed effects have no post-action observation. Valid menu pauses do not
count as task success. Actual outcomes remain for later execution checkpoints.

Trace snapshots are persisted before capture and planning and after stage results.
Capture metadata and proposal/perception JSON also use atomic replacement.
Writes flush/synchronize a temporary file, atomically replace the destination,
and synchronize its directory. A persistence failure stops further work; a
force-killed process may leave the last complete snapshot in progress.
Screenshot PNGs are separate and can be incomplete/orphaned after interruption.
Stop checks occur before capture, after capture, and after planning; an API
request already in flight remains bounded by the existing timeout/retry policy.

All 113 tests passed, including executor spies with zero calls, prohibited
input/local-model imports, linked artifacts/checksums, fresh IDs, readable
intermediate snapshots, proposal/paused/blocked/cancelled/failed outcomes,
raw rejected responses, API retries/errors, unknown counts on cancellation,
stop checks, sanitized internal errors, initial/later trace-write failures,
atomic rename/flush failure preservation, and nested secret redaction.

Live evidence: run `7a8d98c790564729bf35331461764347`, observation
`2375d0a3e0fc4cb194463f84b0fc0fd1`, passed validation with one physical
PRESS x for 0.1 seconds. Gemma read the title logo above and Reshiram/credit
below; the captured animation frame had no PRESS START text. One API attempt
took 8.764 seconds; total pipeline time was 9.512 seconds. Image review and
artifact checks agreed with the trace. Calibration position/size remained unchanged.

Initial live runs `1a3de1e572474f8cab38ff9110039734` and
`427544ddf1e84213aa5a61b74e25310e` saved blocked capture traces with zero API
requests when macOS reported the existing window offscreen. The guard remained
in place. An accessibility Raise attempt did not restore native visibility;
the user brought the existing window into view before the completed test.
No game keyboard/mouse actions were sent, and Laya was not loaded or called.
Verification index: `runs/step6-verification.json`.

### 7. Load the actual local Laya checkpoint

Use the existing Laya package's loader with this checkpoint's tokenizer, encoder,
decision weights, question definitions, and calibration metadata. Prefer a
verified MPS path when available; CPU is an explicit measured fallback. Prevent
accidental remote checkpoint downloads and record the runtime device.

Test: load the supplied directory offline; run a small text trace with the
existing agent-trace questions; verify typed answer structure, expected label
sets, finite probabilities, and token-budget behavior. Measure startup and warm
inference time. Compare results on representative original-workflow examples
before attributing a CUA error to transfer.

Pass: local text inference works with the actual checkpoint, with no screenshots
or desktop input sent to Laya.

**Implemented and verified.** `--laya-test [--prompt TEXT] [--laya-device auto|mps|cpu]`
runs the local package loader in an isolated offline worker, with private metadata
copies and the original weights. It preserves all five saved agent-trace question
definitions, validates native typed outputs/probabilities/token usage, checks three
warm repeats, compares four original-workflow examples, and flags a synthetic
overflow trace. Original checkpoint checksums are unchanged. Device, forward
precision, fallback counts, source fingerprints, calibration, and timings are
durably recorded. No screenshot, hosted API, or input executor is used.

134 automated tests pass, including invalid/missing local artifacts, wrong labels,
nonfinite probabilities, inconsistent scores/token counts, image-object rejection,
credential-free offline process setup, forced network-attempt rejection,
cancellation/timeout cleanup, persistence failures, CLI isolation, and preserved
original-workflow question semantics. Existing dry-run tests still verify zero
Laya and executor calls.

Real Metal run `f08a83324567450686373e63f205f0d5` used the user's menu prompt:
loader 4.4635s, warm median 0.1456s, total suite 11.0593s. Actual forwards were
MPS/FP16, with zero CPU fallbacks. CPU run `6444f0e2d15447bda56f69e59046a9a3`
used the same prompt, FP32/two threads, warm median 0.4351s, total suite 18.0649s.
Both flagged overflow and attempted zero network connections. No game input was
sent; the test did not change window position, size, or emulator settings.

Original-workflow pilot uses local test.jsonl lines 1, 2, 4, and 70, the first
matching example for each gold action. Action modal agreement was 2/4 on both
devices; needs_review/outcome/risk each 3/4, urgency 4/4. Gold human_review predicted
observe; gold stop predicted human_review. This is a small compatibility pilot,
with soft target distributions and unverified training independence, not a held-out
accuracy claim. Stored temperatures are applied; calibration and reliability for
CUA traces remain unvalidated. Enforcement stays disabled. Step 8 is the next
evaluation checkpoint; no semantic thresholds were selected here.

Verification index: `runs/step7-verification.json`.

### 8. Laya transfer evaluation and shadow mode

Serialize the goal, Gemma's claimed observation, pending action, actual recent
executions, failures, and objective harness facts into a bounded text trace. Keep
model claims distinct from facts. Evaluate the original question semantics.
Start with shadow mode: Laya records decisions but does not gate execution.
Any live baseline action in shadow mode remains manually approved at this stage.

Test: hand-label development examples covering ordinary progress, conflicting
instructions, repeated failures, unsupported success claims, and scope violations.
Use recorded real traces as well as clearly labeled synthetic cases. Check false
permits, false interruptions, uncertainty, context truncation, and latency.
Reserve separate examples for later held-out evaluation.

Pass: adapter and routing semantics are correct and transfer results are reported
honestly. If behavior is unreliable, continue diagnosis in shadow mode; do not
claim or enable validated autonomous semantic gating yet.

**Implemented and verified, with semantic enforcement disabled.** `--shadow`
connects one fresh dry run to bounded text inference. `--shadow-trace PATH` reads
a completed saved dry run without new capture/API calls or input replay.
`--transfer-test` evaluates the fixed development set only. The original trace is
immutable; a separate shadow record links source checksums, serialized text,
native predictions, hypothetical routing, uncertainty, token usage, and runtime.
No shadow label authorizes input, and no executor is imported by the shadow module.

The serializer separates current-run facts and executed history from model claims
and pending actions. Missing facts stay unknown. Text/list limits expose every
omission; contexts above 5,000 characters fail closed. Model token truncation and
serializer shortening are separate metrics, both structurally ineligible.
Observe retains its keep-running/sampling semantics. Human review queues a trace
in the training definition; the proposed harness pause is explicitly a separate,
conservative interpretation. No probability threshold is selected for enforcement.

161 automated tests pass, including claim/fact separation, bounded histories,
unknown facts, corrupted traces, source immutability, label leakage prevention,
case/result matching, false-permit/false-interruption/outcome accounting, token
budgets, durable failure handling, reserved-case separation, fresh-call wiring,
offline CLI isolation, and zero input even when a model continues past hard blocks.
Real MPS checks used twelve development cases: nine manually authored synthetic
cases and three saved real traces. Labels were written by Codex before inference;
they are not independent human ground truth. Four distinct cases remain reserved
and unevaluated. Both development dataset and checkpoint checksums are recorded.

Final action agreement: 6/12; hypothetical false permits 3/8 interruption cases;
false interruptions 3/4 progress cases. Missed interruptions: repeated non-progress,
unsupported completion, explicit stop. One unsupported success classification
occurred in the completion-claim case. All twelve model inputs fit the token
budget without serializer loss; top action probabilities 0.3817–0.4893.
The earlier serializer version matched 7/12; specifying current-run history and
unknown facts changed one extra decision. Both reports remain saved and labels
are unchanged. Original-workflow action agreement was already 2/4 in Step 7;
transfer is not proven to be the sole source of errors. Step 8 verdict: semantic enforcement is unqualified. Step 9 implements the
user-requested automatic policy with fixtures; it does not resolve these errors
or claim validated semantic enforcement.

Real budget probes flagged a clipped long goal and a 2,104-token Unicode state
that dropped 1,159 state tokens. Both gave an incomplete-context diagnostic pause.
The standalone saved-title CLI check loaded MPS in 3.9265s and cold inference
took 1.1376s. The development/probe runs overlapped, so those timing measurements
are not isolated benchmarks. All real Step 8 checks attempted zero network
connections, used zero CPU fallbacks, and sent zero game input. No new screenshot
or hosted API request was used; fresh pipeline wiring was verified with fakes.

Verification index: `runs/step8-verification.json`. Final projection:
`runs/afbdcf01d08d4f09ac3fc5d7a9d709e1.shadow.json`; actual final model run:
`runs/cb8045c29ebe4dae9c02a2b492349ac9.shadow.json`; title CLI:
`runs/a5ec2cc9d01b429c9fe99e8f595e453e.shadow.json`; budget probes:
`runs/6d647a324f7d45cea3e06471cb60cc05.shadow.json`.

### 9. Autonomous state machine and enforcement routing

**Implemented and verified in simulation.** `harness/agent.py` owns capture, plan,
validate, evaluate, pre-execute, execute, verify, automatic recovery, and stop.
`harness/agent_job.py` parses objective/timed requests and budgets. The default
objective budget is 600 seconds, maximum 1,800; prompt/explicit times must agree.
The deadline includes capture, planning, and inference. Action, model-call,
consecutive-recovery, observation-age, and non-progress limits are explicit.

`harness/agent_proposals.py` supports dialogue, character/name entry, starter
selection, battle, overworld, and party screens, beyond the older menu-only preview.
Only one action is allowed. Game TYPE and upper-screen clicks remain disabled;
name letters use touchscreen clicks. Existing-save menus forbid New Game clicks
and require a claimed Continue selection before x confirmation. These rules use
model-quoted labels; they do not independently establish screenshot truth.
`models/agent_planner.py` provides the goal-conditioned two-screen payload and
single-attempt deadline-bounded API call for Step 10; no real API was called here.

No routine human approval exists. `continue` permits an eligible action, `observe`
also flags sampling, `human_review` obtains a fresh replan and automatically checks
consistency, and `stop` halts. Review clearance requires the repeated exact action,
same claimed screen kind/text, new valid Laya output, fresh identity, and hard checks.
Changed evidence cannot reuse the old authorization. Structural consistency is
not proof of semantic safety, and Step 8 reliability results remain unchanged.

Durable atomic traces precede dispatch, bind unique input IDs to observations,
record possibly delivered input conservatively, and never retry an input error.
Goal candidates receive a separate fresh observation and assessment, independent
of Laya's outcome head. Completion is a simulated goal or live visual-model report,
not independently verified game state. A timed job does not end early at a starter.
Native executor deadline checks release owned keys/buttons on expiry.

Tests: scripted first-Pokémon and two-minute jobs; automatic review rechecking;
changed action/frame replanning; menu stop; stop/focus/geometry/freshness blocks;
malformed/tied/truncated decisions; API, model/action, recovery, non-progress and
clock limits; partial delivery without replay; durable-write failures; cleanup STOP;
fresh goal identity/uncertainty; outcome success not authorizing completion;
credential and runtime isolation; two-image planner payload; single-attempt API;
key/mouse release after deadline. All tests use fakes; no real model or game input.
Animation pixels alone cannot prove progress. The conservative non-progress rule
currently compares claimed screen kind/text and may stop valid movement with no
changing text; Step 10 must verify semantic progress against live observations.

CLI: `--agent-plan` previews a job; `--agent-test` exercises a scripted simulation.
Both accept `--prompt` and optional `--seconds`. No live play entry point is enabled
yet. Simulation inputs/model calls are separate from actual input/inference counts.
Verification index: `runs/step9-verification.json`.

All 202 automated tests pass. The CLI first-Pokémon simulation completed ten
fixture inputs with ten automatic rechecks and zero human requests. The timed
simulation dispatched 27 fixture inputs, all before its 120-second deadline;
the final late result was discarded at 120.1 simulated seconds. The menu-stop
simulation ended after one fixture input. Real input, hosted calls, screen capture,
and Laya weight loads were zero. Calibration and checkpoint metadata were unchanged.

Pass: deterministic autonomous routing and bounded failure behavior with no human
review requests, no duplicate dispatch, and zero real input during Step 9 tests.

### 10. Live menu arrival without selection

Implemented: `harness/live_agent.py` connects native capture, goal-conditioned
Gemma planning, local Laya, focus/geometry/stop/deadline guards, and PyAutoGUI.
`models/menu_check.py` independently checks a fresh two-screen crop before input
and again on a separate frame for goal evidence. `--play` is explicitly menu-only;
no mouse input, New Game confirmation, or Continue selection is enabled.
`--focus-game` optionally activates the calibrated process once at startup.
Later focus loss still stops input and does not trigger automatic refocusing.

Only opening/title proposals for one short physical x are eligible. Native menu
labels latch all input off; the user-requested menu stop also forbids every menu
action. Each dispatch needs a single-use ticket from a separate fresh entry check.
The ticket binds the original action/observation to its fresh gate image and cannot
be reused. Raw model claims remain separate from execution and outcome facts.

The live preset caps actions at 8, model calls at 48, consecutive recoveries at 10,
and observation age at 60 seconds, with the same absolute job deadline. Unknown
opening frames get no input. For this narrow, operator-approved entry gesture,
Laya's review label requires an independent fresh screen check instead of a
redundant planning/inference cycle. The action must still match exactly and pass
local Laya and the fresh entry check. Other controller jobs retain fresh replanning,
new Laya inference, and exact state/text matching. A live attempt with repeated
planning and cold inference sent four x presses, but gaps of 39–120 seconds let
the title/opening sequence cycle; that attempt did not reach the menu.
After removing redundant review planning, another attempt sent three x presses
26–29 seconds apart and still did not reach the menu. A controlled stop ended it
before further inputs. Cold local model reloads took about nine seconds per check.
Live inference now uses one isolated offline model load per job, with a fresh
prediction and context-bound result file for each action. Each result checks
checkpoint hashes and network attempts; EOF/stop/deadline cleanup closes the worker.
Goal candidates never receive input. Fresh goal verification can run after the
last allowed input instead of terminating before observing its effect.

Google SDK requests initially suffered connection drops/timeouts, and a long
read-only diagnostic exposed schema-copy responses. Compact response-instance
prompts fixed the format; strict parsers still reject malformed output. A direct
system-curl HTTP/2 request succeeded, so live operations use that single-attempt
REST transport. Credentials are passed only through stdin to the fixed Google
host. TLS verification remains enabled; redirects are disabled; cancellation
terminates the child and removes temporary payloads. httpx HTTP/2 dependencies
were added without changing any existing locked dependency version. Legacy SDK
preview calls also enable HTTP/2, but their reliability has not been established.

229 tests pass, covering real-port wiring with fakes, forced-stop zero input,
already-at-menu zero input, menu appearance during a pending proposal, native
focus failures, PNG integrity, single-use ticket/idempotency, offline result
matching, automatic rechecking across blinking/opening frames, unknown-frame
recovery, two-image REST payloads, credential handling, HTTP/transport rejection,
function/truncated replies, cancellation cleanup, persistent-worker reuse with
fresh contexts, failed identity/checkpoint/network checks, stopped/dead workers,
and bounded waits without hidden retries. Forced-stop testing uses
fixture outputs only; real Laya predictions are recorded separately, never altered.

Live verification passed: `runs/41df304285ca4a9889ab893489ecbbb6.agent.json`
records two guarded physical x presses and fresh menu confirmation. No input
followed the menu latch; New Game was never activated. The game highlights that
option by default. Three fresh text-only Laya predictions used one offline MPS
worker, with zero CPU fallbacks/network attempts and unchanged source/checkpoint
hashes. The run ended after 139.566 seconds, with zero human-review requests.
Final screenshot: `screenshots/567c62b3a5ec4f71a10d161ee176d5ba.png`.
Verification index: `runs/step10-verification.json`, preserving prior failures.

Pass: an actual guarded CUA loop reaches a visibly confirmed New Game menu and
stops without selecting it; no routine human-review requests occur.

### 11. Bounded introduction progression

Readiness reviewed on 2026-10-09: all 229 tests pass again, the current crop still
shows New Game, calibration matches, and no persistent STOP is set. The reusable
capture, planner, offline Laya session, executor, and trace components are ready.
The review preceded implementation and sent no input. The separate introduction
policy is now implemented in `harness/live_intro.py`, with visual facts from
`models/intro_check.py`. It preserves Step 10's menu-stop policy.
Review record: `runs/step11-readiness.json`. No game input was sent in this review.

The user's x rule now chooses one short x from a recognized opening/title,
New Game menu, introduction dialogue, or visibly selected Boy screen. An existing
save, unknown transition, changed selection, or name keyboard cannot authorize
that press. Every proposed press obtains a fresh text-only Laya prediction and
independent phase/selection check. A single-use dispatch ticket binds the action
to its checked frame; native focus, geometry, stop, and deadline checks remain.
The name screen latches all further input off and receives independent fresh
goal verification. The preset permits at most 40 inputs and 150 model calls,
10 consecutive recoveries, and 60-second observation age within the job deadline.

249 tests pass, including x-only introduction progression with no human review,
already-at-name-entry zero input, existing-save refusal at planning and dispatch,
name keyboard appearing before dispatch, changed Girl selection, Laya stop,
no repeated New Game activation, ineffective-dialogue termination, unknown
transitions, unsupported name claims, two-image identity binding, and CLI scope.
The user requested two or three x presses. Opening/title progression now uses a
conditional pair with a separate observation and Laya decision for each press.
A reviewed static title-logo pixel anchor accelerates the second proposal and
entry guard; its fixed tolerance cannot be relaxed by a model. The anchor is
bound to the reviewed calibration and image hash. It matched five recorded title
captures and rejected two recorded menus and a fading transition, plus unit
tests for menu replacement, animation outside the logo, tampering, stale
calibration, and missing references. A failed local match falls back to hosted
checks. Menus, dialogue, and gender choices retain one press per observation.
The first live attempt stopped at its recovery limit after two x presses while
the title sequence cycled. Its trace remains `runs/f5cfc8bec39645d78af91b819c638f48.agent.json`.
The paired attempt reached New Game with presses 2.713 seconds apart, but visual
facts contradicted their selected-option field; the harness sent no menu input.
Explicit highlight-to-field instructions passed a read-only menu diagnostic.
Another zero-input attempt exposed complete JSON wrapped in Markdown. Introduction
facts now accept one exact complete JSON code block, retaining strict object,
field, identity, and action checks and rejecting outside prose. The wrapper is
recorded; malformed replies preserve redacted diagnostics. All 249 tests pass
after this protocol fix. Failed and interrupted attempts remain in the final index.
Step 11 review on 2026-10-09: the user accepted the game objective. A read-only
capture shows “GEMMA chose Oshawott!”, beyond the introduction. All 249 tests
pass again, and compilation passes. The latest saved Step 11 trace stopped
after one completed menu x with `new_game_reentry_forbidden`; the current scene
cannot be attributed to a complete autonomous introduction run from these
records. No input or model calls were made during review. Two findings remain:
generic dialogue lacks a positive introduction-scope check, and an immediate
post-input menu capture can terminate before transition settling. Source behavior
was not changed. See `STEP11_REVIEW.md` and `runs/step11-review.json`. Step 12
has not started.

Run short batches with one state-changing action per observation. Recognize
dialogue progress, use the verified DS A key, and reach character selection.
Distinguish animation settling from lack of progress. Automatically recheck review labels and recover within bounds. Stop on Laya stop,
hard violations, failure limits, or the budget boundary. Never resume a terminated
job without a new prompt.

Test: inspect before/after captures and verify dialogue advances without input
bursts. Demonstrate that repeated ineffective input triggers bounded recovery
and then stops. Confirm male selection from visible game evidence.

Pass: progression to the name-entry screen with a trace for every action.

### 12. Touchscreen name entry and completion — skipped by user

Have Gemma locate H, U, Y, and confirmation controls from fresh images. Execute
one explicit click per observation after validation and Laya gating. Verify the
name incrementally and use visible deletion/correction controls if needed.

Test: captures show H, HU, and HUY; a mistaken letter is detected and corrected
within the configured recovery limit. Stop only after a fresh image provides the
agreed name-confirmation evidence. A budget exit or Gemma's unsupported success
claim cannot be recorded as task success. Verify no input follows termination.

Pass: name entry and subsequent first-Pokémon acquisition, or timed play as
requested, have fresh visible evidence and bounded autonomous routing. Laya
reliability remains measured separately from controller correctness.

### 13. Repeated benchmark and guardrail evaluation — skipped by user

Compare Gemma plus hard rules against the same system plus Laya, using matched
start states, prompts, model settings, observation policy, and budgets. Alternate
run order; use at least five trials per condition as a pilot, not a statistical
proof. Establish a repeatable reset through emulator UI only, protecting any
existing progress; no game-state parsing or memory access is used by the agent.

Test: report completion rates, incorrect inputs, corrections, API calls, recoveries,
elapsed time, Laya inference time, and manually labeled false permits/interruptions.
Evaluate thresholds on held-out examples without tuning against them. Preserve
partial progress, failure, budget expiry, and unresolved automatic recovery as
distinct statuses. No runtime human review is required.

Pass: a reproducible harness and an evidence-based account of Laya's benefit,
overhead, and limits. The harness can work even if the experiment shows no benefit;
improvement must not be assumed from merely inserting Laya into the loop.

## Completion checklist

- [x] Cropped visual observations and verified screen-to-input mapping.
- [x] Generic MOVE, CLICK, PRESS, TYPE, and WAIT capabilities verified.
- [x] Hard validation and stop controls tested before model-driven execution.
- [x] Hosted Gemma describes both DS screens with verified title/prompt reading.
- [x] Hosted Gemma returns validated action proposals.
- [x] Local Laya consumes only text and returns correctly interpreted decisions.
- [x] Laya routes actual menu input with automatic rechecking; hard stop enforcement is fixture-tested.
- [x] Controller simulation verifies fresh observations and rejects stale proposals.
- [x] Live observations verify effects of model-driven game inputs (menu entry).
- [x] Objective/timed controller jobs run without human-review requests in simulation.
- [x] Dry-run traces keep proposals, disabled execution, and unobserved effects distinct.
- [x] Real executions and post-action outcomes are integrated into durable traces.
- [ ] Pokémon Black intro, male selection, HUY entry, and termination verified.
- [ ] Matched baseline/guarded pilot and held-out decision evaluation reported.

Next implementation checkpoint after Step 10 passes: Step 11 introduction
progression only when the user requests it. The current live menu job must not
select New Game or Continue. Laya semantic reliability remains unvalidated; no
new training or calibrated threshold is claimed.


## Existing White save: navigation and party inspection

User requested on 2026-10-09: use the White ROM in `pokemon_white/roms`, keep the
existing save, wander with arrow keys, open the menu with s, confirm with x, and
go back with z. Steps 12 and 13 are skipped, rather than remaining prerequisites.
ROM_PATH and the example/default configuration now identify White; credentials
were not changed. Calibration was refreshed and visually reviewed for the restarted
melonDS window without moving or resizing it.

`harness/live_gameplay.py` connects the real guarded executor to the one-action
Gemma planner, persistent offline Laya session, independent fresh gameplay checks,
and before/after semantic effect checks. The new policy requires existing-save
walking/menu/party/Bag/Summary evidence; generic dialogue and battles cannot receive
input. Only safe Pokémon/Bag menu confirmations and Summary selection are allowed;
item use, party changes, saving, and settings are excluded from this initial scope.
Each press remains separate even when entering Pokémon takes two x presses.

265 tests pass, including menu/party dispatch, fresh changed selection, battle and
later-dialogue refusal, Laya stop, observed walking versus unchanged progress,
identity binding, four-image outcome transport, read-only popup/Bag policy, named
button bounds, bounded jobs, and CLI routing. The first live start exposed a
Pokémon-menu routing ambiguity; it was stopped with zero input and retained as
`runs/015c39e298f447c9bc1c87eb6ff36734.agent.json`. The routing fix is regression-tested.
Live results will be recorded separately from simulated checks.

Additional White checks: 272 tests now pass. A .15-second s press visibly opened
the game menu. Invalid key target fields and single-screen click coordinates were
rejected without dispatch. The response instruction now provides a concrete
keyboard-menu example and requires full-crop mouse coordinates. Neutral menu
activation and highlighted confirmation are distinct fresh actions, matching the
user's two-x explanation. Startup activation now precedes window discovery so
Stage Manager can restore the calibrated process once; exact geometry is still
required, and later focus loss terminates rather than repeatedly refocusing.
Regression checks cover neutral versus uncertain menu state, separate Laya calls
for both x presses, unavailable-window termination, wrong-process activation,
changed geometry after restoration, and bounded single activation timeout.
