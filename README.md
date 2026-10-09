# Pokémon White CUA harness

A prompt-driven computer-use harness with hosted Gemma for screenshots and local,
text-only Laya for decisions. The configured game is now the existing White save:
`/Users/adamvu/Pokemon_retro/nintendo_ds/pokemon_white/roms/Pokemon - White Version (USA, Europe) (NDSi Enhanced).nds`.
The harness does not load/reset the ROM automatically; keep that save open in melonDS.

**Current live scope: walking and read-only party/Bag inspection.** The user
requested skipping the earlier numbered Steps 12 and 13 and continuing from the
existing save. Earlier Black introduction/menu evidence remains historical.
The generic executor supports mouse/keyboard input; this live policy permits
only short navigation keys and verified Pokémon/Bag touchscreen buttons.

| Physical key | Game control |
| --- | --- |
| `s` | Open the game menu (DS X) |
| Arrow keys | Walk or move the menu selection |
| `x` | Confirm the selected option (DS A); opening Pokémon can require two separate presses |
| `z` | Return to the previous panel (DS B) |

Pokémon are inspected through **Pokémon**; **Bag** contains items. Summary viewing
is allowed. Item use, party rearrangement, release, Save, Options, introduction,
and battles are outside this initial exploration policy. Unsupported scenes stop
input. A new prompt starts a new bounded job without routine human review.

Inspect the party and stop on its list:

```bash
.venv/bin/python main.py --play --focus-game --laya-device mps --seconds 180 --prompt "Inspect my Pokemon party through the Pokemon menu and pause on the party list. Do not use items or change my Pokemon."
```

Inspect the party, return to the world, and wander within a two-minute wall-time budget:

```bash
.venv/bin/python main.py --play --focus-game --laya-device mps --seconds 120 --prompt "Inspect my Pokemon party, then return to the world and wander for 2 minutes. Do not use items or change my Pokemon."
```

The timer includes capture and inference; it is not two minutes of continuous
walking. An objective can end on fresh party evidence. A timed job ends at its
deadline or earlier on a guard/model/failure limit, and its trace records the reason.
Controls are released on termination. `--focus-game` activates the calibrated
process once at startup, allowing Stage Manager to restore its window before
exact identity/geometry checks; subsequent focus loss stops input. Window position, size, and
melonDS settings are not changed.

Each input uses a new Gemma proposal, fresh text-only Laya inference, and an
independent check of both newly captured screens. `x` must match the actual
highlighted safe option, except that one x can activate keyboard selection on a
freshly verified complete neutral menu; confirmation still needs a later frame. Touchscreen clicks must fall inside the freshly read
named button. No blind key bursts or uncertain-delivery retries are allowed.
After a short settling interval, a four-image before/after check records the
visible outcome. Walking progress requires displacement or coherent terrain
movement; blinking/animation alone is insufficient. Repeated unconfirmed progress
stops at the configured no-progress limit. The preset caps inputs at 40, model
calls at 160, recoveries at 5, and observation age at 60 seconds.

Party completion requires a different fresh frame with owned Pokémon names and
legible levels or HP. Laya `stop` remains terminal; `human_review` receives the
independent automatic check without asking a person. The checkpoint stays unchanged.
This verifies harness wiring, not Laya's semantic reliability (the earlier small
Step 8 development pilot was 6/12 acceptable labels).

273 tests pass; live White verification is being recorded. Current calibration
was reviewed for both DS screens after melonDS restarted. First-Pokémon acquisition,
battle play, and arbitrary general objectives are not enabled.

Earlier menu-stop and introduction presets remain available for their original
start states; do not run an introduction prompt on a later-game dialogue. See
[Step 11 review](STEP11_REVIEW.md) for the retained scope/settling findings.

Live requests use a single-attempt system-curl HTTP/2 transport to Google's
Gemini generateContent API, following the [official Gemma API interface](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api).
The existing credential travels through stdin, never command arguments, payload
files, or logs. Only the two game PNGs are sent; no redirects or TLS bypass are
allowed. Request cancellation terminates the child process. The SDK transport
remains available for legacy previews, with HTTP/2 support installed.

The strict response parser remains unchanged. Compact concrete response examples
replace embedded JSON schemas in the autonomous planner/checker prompts, preventing
schema-copy replies. Main `runs/<id>.agent.json` traces link a separate live-port
record containing provider results, worker records, fresh guard captures, unique
input IDs, and goal evidence. Failed attempts preserve their actual input counts.

`--agent-test` remains a **scripted simulation**, without controlling melonDS.
Objective and two-minute simulations do not establish real gameplay or Laya accuracy.
Existing dry-run/shadow commands remain previews; manual input remains separate.

Preview the interpreted job:

```bash
.venv/bin/python main.py --agent-plan --prompt "Enter the game and get your first Pokemon."
```

Exercise the controller without controlling melonDS:

```bash
.venv/bin/python main.py --agent-test --prompt "Enter the game and get your first Pokemon."
.venv/bin/python main.py --agent-test --prompt "Play the game for 2 minutes."
```

These fixtures test control flow; they do not test real Gemma/Laya decisions or
prove first-Pokémon acquisition. An arbitrary objective in `--agent-test` still
uses the same scripted starter scenario. The emulator window is untouched.

A duration-only play prompt runs until its deadline; an objective can finish
sooner after a separate fresh goal assessment. `--seconds 120` can specify the
budget explicitly; it must agree with any time stated in the prompt. Objectives
without a duration default to ten minutes; the maximum budget is thirty minutes.
The timer includes capture and model work. Input cannot begin at or after expiry,
and owned buttons/keys are released on interruption. A fault or stop signal can
end play earlier than the requested duration. The run record preserves why.

Defaults also cap actions at 100, model calls at 200, consecutive recoveries at
3, and repeated unchanged claimed screen states at 3. One proposal produces at
most one input attempt. A possibly delivered input is never retried. Changed
observations discard pending actions; stale frames, malformed decisions, ties,
context loss, and token truncation trigger bounded automatic recovery.

No calibrated probability cutoff is invented. Laya's saved questions/labels are
unchanged. Its Step 8 development results remain weak: **6/12 acceptable action
labels**. Automatic rechecking tests structural consistency; it does not establish
semantic correctness. Deterministic scope, focus, geometry, deadline, stop, and
input checks remain mandatory. Screenshot descriptions and goal conclusions are
model claims; the outcome head cannot prove completion. Existing-save menu checks
use quoted model labels and require visible Continue selection; they are not an
independent pixel-level guarantee.

`runs/<run-id>.agent.json` records jobs, observations, proposals, native decisions,
automatic routes, unique dispatch IDs, input receipts, fresh goal assessments,
and stop reasons. Simulation records explicitly show zero real input/inference.
`task_complete` remains false; a simulated goal is labeled `simulated_goal`, and
future live visual evidence is labeled `visual_model_report`.

From this project directory, validate your existing `.env`:

```bash
uv run --locked python main.py --check
```

Run one request through capture, planning, validation, and logging:

```bash
.venv/bin/python main.py --dry-run --prompt "Reach the New Game or Continue menu and pause there."
```

Add a Laya shadow decision for that request (still no input):

```bash
.venv/bin/python main.py --shadow --prompt "Reach the New Game or Continue menu and pause there."
```

Generate only the standalone proposal record:

```bash
.venv/bin/python main.py --propose --prompt "Reach the New Game or Continue menu and pause there."
```

Describe the current game screen for your request:

```bash
.venv/bin/python main.py --perceive --prompt "Describe both DS screens and quote the visible prompts."
```

Or enter the perception prompt interactively:

```bash
.venv/bin/python main.py --perceive
```

Test your prompt with local Laya, without observing or controlling the game:

```bash
.venv/bin/python main.py --laya-test --prompt "Reach the New Game or Continue menu and pause there. No action has executed; inspect this request only."
```

This command also runs repeatability, original-workflow, and token-overflow checks.
Use `--laya-device mps` to require Metal or `--laya-device cpu` for an explicit CPU
test. Without `--prompt`, it uses a small built-in text trace.

To capture the game, open Pokémon Black in melonDS with two equally sized screens
stacked vertically, zero screen gap, no rotation, and **Swap screens disabled**.
Keep the game window visible. Calibrate once for its current position and size:

```bash
.venv/bin/python main.py --calibrate
```

Open the printed PNG preview and check that both game screens are complete, the
title bar is excluded, and the lower half is the DS touchscreen. Confirm that review:

```bash
.venv/bin/python main.py --confirm-calibration
```

Then attach a single observation to your request:

```bash
.venv/bin/python main.py --capture --prompt "Capture the current game screen for review."
```

Use `--capture` without `--prompt` to enter the request interactively. A prompt
without an operation flag runs configuration preflight only. Images are saved to
`screenshots/`; observation IDs, timestamps, prompt text, crop geometry, Retina
scale, and coordinate samples are saved to `runs/`. Both directories are ignored
by Git. `--capture` saves locally and makes no model request.

## Shadow decisions and transfer evaluation

`--shadow` captures and asks Gemma through the existing dry-run pipeline, then
serializes the resulting trace for one isolated local Laya call. Failed capture or
planning stops before Laya. No action executes, including a `continue` decision.
Both the original dry-run trace and a separate `runs/<run-id>.shadow.json` are saved.
The shadow record links its offline worker record. `--execute` cannot be combined
with any shadow operation. `--dry-run` continues to call Gemma only.

Use a saved trace without new screen capture, hosted requests, or input replay:

```bash
.venv/bin/python main.py --shadow-trace runs/28915a654a894b26a5a52085c1a7c3db.trace.json
```

Saved-trace evaluation keeps that trace's original goal. It refuses unfinished or
inconsistent dry-run execution/completion/observation facts. It never opens linked
images or modifies the source trace. Evidence is historical, not a verified current
screen or an authorization to replay its pending action. Current-run input history
is zero for these dry runs; actions in other runs and prior game state are unknown.

The text context keeps recorded harness facts, pending action claims, Gemma's
unverified screen descriptions and expected effects, actual execution events, and
failures separate. Missing focus/scope information stays unknown. Text fields are
shortened to explicit limits; only the four most recent execution/failure events
are retained. Every omission is recorded. Final text is capped at 5,000 characters;
excessive combinations fail before inference. Laya's tokenizer separately reports
any model token truncation. Both kinds of incomplete context produce an ineligible
diagnostic disposition. Neither can authorize input.

`observe` means continue and flag for sampling, not capture another screen.
The trained `human_review` label queues a trace for review. Shadow reports preserve
the earlier hypothetical human-pause mapping separately from that label meaning.
The Step 9 autonomous controller instead performs bounded automatic rechecking;
shadow reports never execute either route. All five questions retain their saved
definitions. Outcome classifications remain model claims. Top probability, top-two
margin, and entropy confidence are reported without an enforcement threshold.
Recorded hard blocks remain separately visible regardless of the model label.

Evaluate the fixed development cases locally:

```bash
.venv/bin/python main.py --transfer-test --laya-device mps
```

The dataset in `tests/fixtures/shadow_development.json` contains nine synthetic
scenarios and three real saved traces. Labels were manually authored by Codex
before inference; they are not independent human labels or a held-out benchmark.
They are never sent to Laya. The command depends on this machine's saved trace
files. Four additional cases are reserved in `tests/fixtures/shadow_reserved.json`;
the development command does not load them and rejects the reserved split.

Final development results: **6/12 acceptable action labels**, **3/8 hypothetical
false permits**, and **3/4 false interruptions**. The missed interruptions concern
repeated non-progress, an unsupported completion claim, and an explicit stop flag.
The unsupported-completion case also classified its outcome as success without
verified completion. All twelve complete contexts fit the token budget. Top action
probabilities ranged from 0.3817 to 0.4893, with no CUA-calibrated threshold selected.
“False permit” here means a raw `continue`/`observe` label where interruption was
expected; no input was actually permitted or sent, and hard stops remain mandatory.

The first serializer version matched 7/12. Making current-run history limits and
unknown facts explicit changed one additional case to `observe`. Both reports
are preserved; labels were not changed. Original-workflow results in Step 7 were
already weak (2/4 action matches), so these errors cannot be attributed solely to
CUA transfer. Context loss is not the explanation for the twelve final cases.
Laya should remain advisory while these failures are diagnosed.

Step 8 passed 161 automated tests. Real Metal checks used saved game traces and the
fixed development set, with zero network attempts, zero CPU fallbacks, unchanged
checkpoint checksums, and zero input. Real budget probes separately confirmed
serializer loss and token overflow. The final saved-title CLI check loaded in
3.927 seconds; its cold inference took 1.138 seconds. Development evaluation and
budget probes overlapped, so their timings are diagnostic observations rather
than a performance benchmark. Fresh `--shadow` wiring was tested with fakes;
this step made no new hosted request or game capture. Keep the existing calibrated
window visible for your own fresh command; the Stage Manager delay below also
applies to `--shadow`.

Evidence index: `runs/step8-verification.json`. Final evaluation is
`runs/afbdcf01d08d4f09ac3fc5d7a9d709e1.shadow.json`, derived from the saved final
worker predictions to clarify native versus proposed routing semantics, with no
new inference or changed predictions. Actual final inference is linked through
`runs/cb8045c29ebe4dae9c02a2b492349ac9.shadow.json`; the standalone title check is
`runs/a5ec2cc9d01b429c9fe99e8f595e453e.shadow.json`.

## Offline Laya checkpoint test

`--laya-test` imports your configured Laya source in an isolated worker and calls
its existing `laya.load(..., backend="eager")` and `predict` APIs. It requires the
local weights, encoder config, tokenizer files, configuration, and saved questions.
The worker receives no API credentials; Hugging Face offline flags are set before
runtime imports and Python socket connections are blocked. Missing files fail
locally. This socket guard is additional protection, not an OS network sandbox.

Loader metadata is copied under `runs/laya-runtime/<run-id>/checkpoint`; weights
are referenced by a local symlink. The original checkpoint is checksummed before
and after inference. Source code fingerprints, package versions, stored/applied
temperatures, device selection, actual forward devices/precision, CPU fallbacks,
token usage, and timings are recorded in `runs/<run-id>.laya.json`. Startup uses a
240-second worker limit. Cancellation or timeout terminates the worker and records
the last available progress. Temporary runtime copies remain ignored by Git.
The test uses two CPU threads and never activates or changes the game window.

The five saved agent-trace questions retain their original meanings:

| Question | Native result |
| --- | --- |
| action | `continue`, `human_review`, `observe`, or `stop`, with probabilities |
| needs_review | `noul` is **P(true)**; the reported modal label uses 0.5 only for comparison |
| outcome | `failure`, `harmful`, `partial`, or `success`, with probabilities |
| risk / urgency | Expected score from 0 to 3, probabilities, and saved level descriptions |

For choice/score answers, `confidence` measures normalized entropy;
`answer_confidence` is the temperature-adjusted answer probability. The auxiliary
`act_probability` is a separate head and does not authorize a GUI action. The
trained `observe` decision means keep running and flag for sampling; it does not
mean take a screenshot. No enforcement thresholds are introduced here.

Every answer is checked for type, labels, finite probabilities, normalization,
score/legend consistency, and token accounting. The saved 1,024-token sequence
budget and 256-token head budget are used. Overflow is explicitly flagged and
marked structurally ineligible. Structural eligibility does not establish semantic
reliability, authorize execution, or prove that a game task has completed.

Live checks passed on CPU and Metal. For the supplied menu prompt, Metal loaded
the checkpoint in 4.46 seconds and warm inference had a median of 0.146 seconds;
CPU warm inference had a median of 0.435 seconds. Metal forwards used FP16 with
zero CPU fallbacks; CPU forwards used FP32. Both left the checkpoint unchanged,
attempted zero network connections, and produced the same modal decisions.
The Step 7 verification passed 134 automated tests.

The original-workflow pilot selects the first example for each gold action label
from your local `data/typed_decisions/test.jsonl`, requiring identical checkpoint
question definitions. This four-example pilot is not an accuracy benchmark; its
independence from training has not been verified. On both devices, action matched
2/4 gold modal labels, review need/outcome/risk each matched 3/4, and urgency 4/4.
The `human_review` example predicted `observe`, and the `stop` example predicted
`human_review`. Those mismatches remain unresolved. Step 8 evaluated transfer and shadow
decisions; its results above do not establish reliable semantic gating.

Evidence: `runs/f08a83324567450686373e63f205f0d5.laya.json` (Metal/menu prompt),
`runs/6444f0e2d15447bda56f69e59046a9a3.laya.json` (CPU/same prompt), and
`runs/step7-verification.json`. This command performs zero screenshots, hosted
model calls, or input actions. `--dry-run` still does not call Laya.

## Dry-run traces

`--dry-run` accepts one prompt, takes one fresh observation, and makes one bounded
Gemma proposal request. No input executor or local Laya runtime is imported.
There is no replay or gameplay loop. `--execute`, `--action`, and other operations
cannot be combined with it. Keep melonDS visible on the current desktop at its
calibrated position and size; an offscreen or stale window produces a blocked trace.

The printed `runs/<run-id>.trace.json` links the screenshot, capture metadata,
calibration profile, and proposal record. Its run ID is separate from the
observation ID. It contains capture geometry and an image checksum, the model
configuration and instruction checksum, API attempt counts and usage, stage and
total timing, ordered events, validation results, and sanitized error details.
Gemma's descriptions and expected effects are labeled as model claims/predictions.
Harness execution and outcome facts are kept separately.

Terminal statuses are `proposed`, `paused`, `blocked`, `cancelled`, or `failed`.
`proposed` means the action passed preview validation; `paused` means a valid
no-action proposal. Neither proves gameplay success. Every dry-run trace keeps
`executed: false`, execution call count zero, `task_complete: false`, and
`task_outcome: "not_assessed"`. Observed effects stay `not_observed` with no
post-action observation; later execution checkpoints will provide that evidence.

Trace snapshots are saved before capture and model work, and after each stage.
JSON writes use a flushed temporary file and atomic replacement, followed by
directory synchronization. A failed trace write stops further work. If a process
is force-killed, the last complete trace may remain `in_progress`; that is an
unfinished request, not an executed or successful task. Screenshot PNGs are
separate artifacts and may be incomplete/orphaned if capture is interrupted.

An existing `runs/STOP` blocks dry-run work before capture. Stop is also checked
after capture and after planning; it does not immediately interrupt an API request
already in flight. Cancellation records unknown API counts when a request may
have reached the service. API keys are redacted from records and console output.

Exit codes: 0 for a valid proposal/pause, 2 for invalid configuration/prompt,
3 for capture/stop/system errors, 4 for API or proposal validation failures,
5 for trace persistence failure, and 130 for cancellation. Invalid CLI argument
combinations also use 2. Accepted configured requests get a trace; requests
rejected before the pipeline starts do not.

Step 6 passed 113 automated tests, including executor spies, and a real live
Gemma dry run. It proposed one physical `x` press for 0.1 seconds, with zero
execution and no claimed gameplay outcome. The initial offscreen-window failures
were preserved as blocked traces; the live test completed after the user restored
visibility without changing calibration. See `runs/step6-verification.json`.

### Running from Terminal with Stage Manager

Screen Recording permission must be enabled for the app hosting Python, such as
Terminal or iTerm; permission for the Codex host does not cover every launcher.
After granting it, fully quit and reopen that terminal app.

Stage Manager can reduce melonDS to a small preview when you switch to Terminal.
The capture guard rejects that preview. Give yourself five seconds to switch back
to the full melonDS window, then leave it visible until the command finishes:

```bash
sleep 5 && .venv/bin/python main.py --dry-run --prompt "Reach the New Game or Continue menu and pause there."
```

This changes no emulator settings, moves no window, and sends no game input.
If melonDS was restarted, refresh calibration in place first. Use the same delay
for `--calibrate` and `--confirm-calibration`, reviewing the printed crop PNG
between those two commands. Do not confirm a Stage Manager preview.
Discovery now identifies small previews, offscreen game windows, and replaced
window IDs separately; 115 automated tests pass after these diagnostic changes.

## Gemma proposals

`--propose` handles one fresh observation and returns one JSON proposal, or a
pause with no action. It never imports the input executor or Laya, and cannot
be combined with `--execute`. Run it again for a fresh observation when reviewing
the next proposed step. It does not replay or execute an earlier proposal.

The user reports that physical `x` (DS A) advances the opening/title screens,
and that idling returns to the opening animation. The current policy permits one
short `x` press per opening/title proposal. It requires a pause when the model
reports the New Game/Continue menu or an unknown screen. It does not select or
load a save. This behavior has been checked in proposals; Step 5 sends no input
to verify the resulting game transition.

Gemma receives the JSON schema as an instruction; the adapter does not assume
provider-enforced structured output. Local validation rejects malformed or fenced
JSON, duplicate keys, extra fields, action lists, key chords, non-finite values,
stale observation IDs, unsupported completion claims, and invalid game actions.
It never repairs an action or guesses missing coordinates. Touchscreen clicks
must name a label in the model's lower-screen text and use full-game-image
coordinates. This consistency check does not independently prove the target's
location; review the screenshot alongside the model claims.

Inspect `runs/<observation-id>.proposal.json` for the raw response, validated
proposal or rejection, API attempts, latency on success, and `executed: false`.
`expected_result` is a prediction, and `task_complete` must remain false.
API transport retries use the same bounds as perception; invalid proposals are
not retried or repaired within the request.

Step 5 passed 90 automated tests and real Gemma title-screen proposal checks
on a live capture and the approved supplied crop. Both final proposals used
physical `x` for 0.1 seconds. The initial fixture proposal incorrectly chose
`enter`; validation rejected it, and a clearer title-rule instruction resolved
that case. A live transport failure stopped after two attempts. Main-menu stop
checks and explicit name-entry target checks use synthetic proposals; real
name-entry vision and click effects remain for the later gameplay checkpoint.

## Gemma perception

`--perceive` saves the complete calibrated crop, then extracts both equally sized
DS screens from that same frame at their original resolution. It sends two
separately labeled image parts: upper display first, lower touchscreen second,
with each rectangle's position in the full saved image. Neither screen is discarded when it
is blank or less useful in a particular scene. The emulator's display layout,
position, size, and settings remain unchanged.

This command sends only those game-screen PNGs and your request to Google's Gemini API,
using `GEMMA_MODEL` and `GEMINI_API_KEY` from your existing configuration. It uses
inline images rather than creating separate uploaded files. Embedded screenshot
metadata is removed from the screen inputs. Gemma returns a
brief screen description, with readable text and uncertainty. The output is
displayed as a model claim; it is never interpreted as an input action.
The command cannot be combined with `--execute` or `--action`, and imports no
PyAutoGUI or local Laya inference runtime.

Each attempt has a 25-second wall timeout. Transient connection errors, timeouts,
quota/rate limits, and selected server errors receive at most one retry, after
one second. SDK retries are disabled to avoid multiplying attempts. Invalid
credentials, missing models, rejected requests, empty responses, blocked responses,
unexpected function calls, and truncated output produce errors without input.
Model output is capped at 600 tokens. Request cleanup has a two-second bound.

Review the PNG alongside its `runs/<observation-id>.perception.json` record,
which contains the description or sanitized error, model, attempts, latency on
success, screen rectangles, image-input strategy, source kind, and `executed: false`.
Fixture tests are labeled separately from live captures. The separate capture JSON
records the capture stage itself (`model_called: false`); the perception record
records whether an API attempt occurred. Neither record claims that a description
has been independently verified. Use `--dry-run` for the integrated run trace.

Step 4 passed 70 automated tests and real Gemini API requests with the configured
Gemma model. Live captures verified title/animation reading. The approved
game-only fixture verified “PRESS START” and correct attribution to the upper
screen, with artwork and the Game Freak credit on the lower screen. The initial
combined-image strategy confused screen boundaries on that fixture; the final
adapter uses two labeled crops. A real HTTP 500 failure also stopped after the
configured two attempts. This is a small perception check, not a gameplay benchmark.

Capture uses macOS window discovery and a window-only native capture, so unrelated
desktop content is not included. It does not import PyAutoGUI or move the pointer.
Explicit manual controls use PyAutoGUI. Screen Recording permission is
required for the app running Python. The Codex shell sandbox can prevent discovery;
run these commands in your local terminal if that error appears.

The crop defaults to a standard macOS title bar and a one-pixel border. If your
window chrome differs, use `--calibrate --insets LEFT TOP RIGHT BOTTOM`, with values
in desktop points. `--screen-gap` specifies an explicit gap for calibration, also
in desktop points. Multiple main windows require an explicit `--window-id`.
Every new calibration needs preview review and confirmation.

Moving/resizing/restarting the emulator or changing its capture scale makes the
profile stale; capture stops and asks for recalibration. Changing screen order,
rotation, aspect ratio, or sizing also requires a new visual calibration, even
if the outer window geometry remains unchanged. The current prototype cannot
automatically detect every layout change inside an unchanged window.

Configuration requires a nonempty `GEMINI_API_KEY`, the existing `.nds` ROM,
the local Laya source, and the fine-tuned checkpoint. `.env.example` lists supported
settings; the current machine's supplied paths are defaults, so your existing
`.env` can remain unchanged. Process environment variables override `.env`.
Configuration is resolved from the project directory even if launched elsewhere.
API key values are hidden in reports and validation errors. Prompt-driven capture
also redacts the configured API key if it appears in the saved prompt.

During preflight the checkpoint is inspected for expected files, without loading its weights.
Package discovery reports availability; it does not establish runtime compatibility.
Laya is not installed automatically. Step 7 imports the configured local source
in its worker; preflight package discovery may therefore report `laya: not installed`
even when `--laya-test` works. Runtime compatibility and inference are verified
separately by that command.
Legacy `EMU_*` geometry values are preserved in `.env` but are not used by window
capture; that uses the verified `runs/calibration.json` profile.

Run all automated harness checks with:

```bash
uv run --locked python -m unittest discover -s tests -v
```

The executor supports cursor movement, explicit-target clicks, key presses, and
bounded waits. Literal typing is restricted to the disposable input-test field;
Pokémon name entry will use clicks on the DS touchscreen.

## Manual controls

Keep the emulator window at its calibrated position and size. Input never moves,
resizes, zooms, or activates that window automatically. Commands validate one
explicit JSON action; your natural-language prompt records your intent only.

Preview one Start press:

```bash
.venv/bin/python main.py --prompt "Press Start once" --action '{"tool":"PRESS","key":"enter","seconds":0.1}'
```

To execute it, add `--execute`. There is a three-second delay to bring melonDS
forward; Ctrl+C cancels. `--start-delay` can override that delay from zero to five
seconds. The window must already be in front when execution begins; a focused
dialog, another app, changed geometry, or missing permissions blocks input.

Supported actions:

| Action JSON | Limits |
| --- | --- |
| `{"tool":"MOVE","x":500,"y":750}` | Integer normalized coordinates 0–1000 inside the game crop |
| `{"tool":"CLICK","x":500,"y":750}` | Explicit target in the lower touchscreen; left button only |
| `{"tool":"PRESS","key":"x","seconds":0.1}` | One permitted physical key; duration greater than 0 and at most 1 second |
| `{"tool":"WAIT","seconds":0.5}` | Greater than 0 and at most 2 seconds |
| `{"tool":"TYPE","text":"HUY"}` | Game profile rejects typing; verified separately in an owned test field |

Permitted physical keys are `x`, `z`, `s`, `a`, `q`, `w`, `enter`, `shiftleft`,
`up`, `down`, `left`, and `right`. They correspond to the supplied DS mappings.
No key chords, shell actions, arbitrary buttons, or extra fields are accepted.
Retina half-point coordinates are rounded inward to valid desktop coordinates.

Verify literal typing and interruption cleanup in a disposable field:

```bash
.venv/bin/python main.py --input-test
```

This creates and closes its own native macOS test window, checks exact `HUY 42!` typing,
then triggers a stop while Shift is held and verifies that Shift was released.
It sends no game input and restores the previously active app. The initial typing
implementation supports up to 32 printable ASCII characters using the active
keyboard layout; it rejects newlines, tabs, and non-ASCII text.

Step 3 verification passed 54 automated tests plus live cursor movement, a
lower-screen click and release, Return, Up/Down menu navigation, bounded WAIT,
and two native-field typing/stop checks. DS B restored the title screen. The
blank touchscreen click does not yet verify selection of a game control;
name-entry clicks will test that later. Left/Right game responses also need a
suitable screen. No models are called in these manual controls.

Stop controls:

```bash
.venv/bin/python main.py --stop
.venv/bin/python main.py --clear-stop
```

`--stop` creates a persistent `runs/STOP` signal and works without API credentials.
An executing action checks it during bounded holds/waits, along with focus and
PyAutoGUI's corner fail-safe. Ctrl+C also interrupts. Cleanup releases only controls
pressed by this executor, including when focus changes or the fail-safe triggers.
`--clear-stop` sends no input and does not resume an earlier request.

Live input requires macOS Accessibility permission for the app running Python,
in addition to Screen Recording for game captures. Release modifiers and turn
Caps Lock off before testing. A force-killed process cannot run cleanup; use the
supported stop controls while input is active.

If melonDS moves, resizes, or restarts, refresh the saved crop with `--calibrate`,
review both screens, then use `--confirm-calibration` before another live job.
Startup errors now state the actual calibration/focus cause.
# pokemon_cuas
