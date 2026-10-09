"""Prompt-driven previews, autonomous controller simulations, and manual controls."""

import argparse
import json
import sys
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

from harness.settings import ConfigurationError, PROJECT_ROOT, RuntimeMode, Settings, load_settings


def dependency_availability() -> dict[str, bool]:
    """Discover packages without importing desktop or model runtimes."""
    availability = {}
    for name in (
        "google.genai", "PIL", "pyautogui", "pydantic", "dotenv",
        "torch", "transformers", "accelerate", "laya",
    ):
        try:
            availability[name] = find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            availability[name] = False
    return availability


def readiness_report(settings: Settings, prompt: str | None) -> str:
    lines = [
        f"Mode: {settings.mode.value} (one request; execution disabled)",
        "Configuration: valid; Gemini API key configured (hidden)",
        f"Gemma model: {settings.gemma_model}",
        f"Game: {settings.rom_path}",
        f"Laya checkpoint: {settings.laya_model_path}",
        f"Laya source: {settings.laya_source_path}",
        "Legacy EMU geometry: " + (
            "configured; not used by window capture"
            if settings.geometry_configured
            else "not configured; window capture uses --calibrate"
        ),
        "Packages (availability only):",
    ]
    for name, present in dependency_availability().items():
        lines.append(f"  {name}: " + ("available" if present else "not installed"))
    lines.append("This preflight does not load Laya; use --laya-test for the offline Step 7 checkpoint test.")
    if prompt is not None:
        lines.extend((f"Your request: {prompt}", "Request accepted for preflight only."))
    lines.append(
        "Preflight complete: no model requests, screen capture, or mouse/keyboard input."
    )
    return "\n".join(lines).replace(settings.gemini_api_key.get_secret_value(), "[hidden]")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prompt-driven Pokémon CUA: previews, autonomous controller simulations, and manual controls."
    )
    parser.add_argument("--prompt", help="One plain-language request; --agent-test simulates autonomous play without game input.")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true", help="Check configuration without a prompt.")
    operation.add_argument("--calibrate", action="store_true", help="Discover melonDS and save a crop preview.")
    operation.add_argument("--confirm-calibration", action="store_true", help="Confirm the crop after visual review.")
    operation.add_argument("--capture", action="store_true", help="Attach one calibrated game capture to your prompt.")
    operation.add_argument("--perceive", action="store_true", help="Capture both DS screens and ask hosted Gemma to describe them; sends no input.")
    operation.add_argument("--propose", action="store_true", help="Capture both screens and validate one Gemma action proposal; sends no input.")
    operation.add_argument("--dry-run", action="store_true", help="Capture, plan, and validate once with a durable run trace; sends no input.")
    operation.add_argument("--laya-test", action="store_true", help="Test the local Laya checkpoint offline with text only; no capture or input.")
    operation.add_argument("--shadow", action="store_true", help="One fresh Gemma dry run followed by a local Laya shadow decision; no input.")
    operation.add_argument("--shadow-trace", metavar="PATH", help="Evaluate one saved dry-run trace locally with Laya; no capture or replay.")
    operation.add_argument("--transfer-test", action="store_true", help="Evaluate the fixed Step 8 development cases offline; reserved cases stay unused.")
    operation.add_argument("--agent-plan", action="store_true", help="Parse an autonomous goal/time job without model calls or game input.")
    operation.add_argument("--agent-test", action="store_true", help="Run the Step 9 autonomous controller with simulated screens/models/input only.")
    operation.add_argument("--play", action="store_true", help="Live menu/intro goals, or existing-save walking and party inspection.")
    operation.add_argument("--action", metavar="JSON", help="Validate one explicit manual GUI action (preview by default).")
    operation.add_argument("--input-test", action="store_true", help="Verify typing and stop/release in a disposable field.")
    operation.add_argument("--input-check", action="store_true", help="Check game input access, focus, and geometry without model calls or input.")
    operation.add_argument("--stop", action="store_true", help="Set the persistent input stop signal.")
    operation.add_argument("--clear-stop", action="store_true", help="Explicitly clear the stop signal; sends no input.")
    parser.add_argument("--execute", action="store_true", help="Execute the validated --action once.")
    parser.add_argument("--start-delay", type=float, default=3, help="Seconds to bring melonDS forward before execution (0–5).")
    parser.add_argument("--focus-game", action="store_true", help="Focus the calibrated melonDS app once at --play startup; later focus loss still stops input.")
    parser.add_argument("--window-id", type=int, help="Select a main melonDS window during calibration.")
    parser.add_argument("--insets", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Calibration crop insets in desktop points; overrides standard window chrome.")
    parser.add_argument("--screen-gap", type=float, default=0, help="Gap between stacked DS screens in desktop points.")
    parser.add_argument("--mode", choices=[mode.value for mode in RuntimeMode])
    parser.add_argument("--laya-device", choices=("auto", "mps", "cpu"), default="auto", help="Device for local Laya operations (auto prefers MPS).")
    parser.add_argument("--seconds", type=float, help="Upper wall-time budget for an agent job; must agree with a prompt time limit.")
    args = parser.parse_args(argv)
    if args.focus_game and not (args.play or args.input_check):
        parser.error("--focus-game requires --play or --input-check.")
    if args.seconds is not None and not (args.agent_plan or args.agent_test or args.play):
        parser.error("--seconds requires an agent job operation.")
    if (args.agent_plan or args.agent_test or args.play) and args.mode is not None:
        parser.error("Agent jobs cannot be combined with preflight --mode.")
    local_operation = args.laya_test or args.shadow or args.shadow_trace or args.transfer_test or args.play
    if not local_operation and args.laya_device != "auto":
        parser.error("--laya-device requires a local Laya operation.")
    if local_operation and args.mode is not None:
        parser.error("--mode preflight cannot be combined with local Laya operations.")
    setup_operation = args.check or args.calibrate or args.confirm_calibration or args.input_test or args.input_check or args.stop or args.clear_stop or args.shadow_trace or args.transfer_test
    if setup_operation and args.prompt is not None:
        parser.error("Use --prompt with preflight or --capture, not with setup operations.")
    if not args.calibrate and (args.window_id is not None or args.insets is not None or args.screen_gap != 0):
        parser.error("--window-id, --insets, and --screen-gap require --calibrate.")
    if args.mode is not None and (args.capture or args.perceive or args.propose or args.dry_run or args.calibrate or args.confirm_calibration):
        parser.error("--mode preflight cannot be combined with capture or calibration.")
    if args.execute and args.action is None:
        parser.error("--execute requires one --action.")
    if not 0 <= args.start_delay <= 5:
        parser.error("--start-delay must be between 0 and 5 seconds.")
    if args.start_delay != 3 and not (args.execute or args.play or args.input_check):
        parser.error("--start-delay requires --execute, --play, or --input-check.")
    if args.mode is not None and (args.action is not None or args.input_test or args.input_check):
        parser.error("--mode preflight cannot be combined with input operations.")
    stop_file = PROJECT_ROOT / "runs/STOP"
    if args.stop or args.clear_stop:
        try:
            if args.stop:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch()
                print("Stop signal set. Input is blocked until --clear-stop.")
            else:
                stop_file.unlink(missing_ok=True)
                print("Stop signal cleared. No input sent; a new request is required.")
        except OSError:
            print("Could not update the stop signal.", file=sys.stderr)
            return 3
        return 0
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.input_check:
        from computer.diagnostics import check_game_input
        print(f"Read-only input check; starting in {args.start_delay:g}s. No game inputs or model calls.", flush=True)
        report = check_game_input(PROJECT_ROOT / 'runs/calibration.json', focus=args.focus_game, start_delay=args.start_delay)
        print(json.dumps(report, indent=2).replace(settings.gemini_api_key.get_secret_value(), '[hidden]'))
        return 0 if report['ready'] else 3

    prompt = args.prompt
    if not setup_operation and not args.laya_test and prompt is None:
        if not sys.stdin.isatty():
            print("Supply --prompt or --check when running without a terminal.", file=sys.stderr)
            return 2
        try:
            prompt = input("What would you like the agent to do? ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled; no actions executed.")
            return 130
    if prompt is not None:
        prompt = prompt.strip()
        if not prompt or len(prompt) > 4000:
            print("Prompt must contain between 1 and 4000 characters.", file=sys.stderr)
            return 2
    if args.agent_plan or args.agent_test or args.play:
        from harness.agent_job import job_from_prompt, live_menu_job, live_intro_job, live_gameplay_job
        try:
            job = job_from_prompt(prompt, seconds=args.seconds)
        except (ValueError, TypeError):
            print("Invalid job: use one matching time limit between 0 and 1800 seconds, with a nonempty goal.", file=sys.stderr)
            return 2
        if args.play:
            try:
                job = live_intro_job(job) if job.goal_kind == 'introduction' else live_gameplay_job(job) if job.goal_kind in {'party','exploration','time'} else live_menu_job(job)
            except ValueError:
                print("Live scope: menu/intro checkpoints, or existing-save walking and party inspection. Starter acquisition is not enabled.", file=sys.stderr)
                return 2
        print("Autonomous job (no human-review approval requests):\n" + job.model_dump_json(indent=2).replace(settings.gemini_api_key.get_secret_value(), "[hidden]"))
        if args.agent_plan:
            print("Job preview only. No model calls or game input.")
            return 0
        if args.play:
            from harness.agent import AgentRecordError
            if job.goal_kind in {'party','exploration','time'}:
                from harness.live_gameplay import run_live_gameplay as run_live
                print("Live White gameplay: guarded walking and read-only party/Bag inspection; one action per fresh screen.", flush=True)
            elif job.goal_kind == 'introduction':
                from harness.live_intro import run_live_intro as run_live
                print("Live Step 11: one guarded x per fresh introduction screen; stop before name input.", flush=True)
            else:
                from harness.live_agent import run_live_menu as run_live
                print("Live Step 10: fresh screens, hosted Gemma, local text-only Laya, and guarded x presses. No menu selection.", flush=True)
            print(f"Bring melonDS forward; starting in {args.start_delay:g}s.", flush=True)
            time.sleep(args.start_delay)
            if args.focus_game:
                from computer.executor import ExecutionError, focus_game_window
                from computer.capture import CaptureError
                from computer.geometry import GeometryError
                try:
                    focus_game_window(PROJECT_ROOT / 'runs/calibration.json')
                except (ExecutionError, CaptureError, GeometryError) as exc:
                    detail = str(exc).replace(settings.gemini_api_key.get_secret_value(), '[hidden]')
                    print(f"Game startup blocked: {detail}\nNo input sent. If the window moved, resized, or restarted, run --calibrate and --confirm-calibration.", file=sys.stderr)
                    return 3
            try:
                record, path = run_live(settings, job, PROJECT_ROOT, device=args.laya_device)
            except (AgentRecordError, OSError):
                print("Live record could not be persisted; no further input will execute.", file=sys.stderr)
                return 5
            print(f"Live status: {record['status']}\nReason: {record['reason']}\nTrace: {path}\n"
                  f"Completed inputs: {record['completed_inputs']}; model calls: {record['model_calls']}\n"
                  f"Elapsed: {record['elapsed_seconds']:g}s. Human review requests: 0.")
            if record['reason'] == 'hard_guard_blocked' and record['decisions']:
                detail = record['decisions'][-1].get('hard_guard', {}).get('reason', '')
                print('Guard detail: ' + detail.replace(settings.gemini_api_key.get_secret_value(), '[hidden]'))
            if record['observations']:
                print("Latest game screenshot: " + record['observations'][-1]['metadata'].get('image_path', 'not captured'))
            return record['exit_code']
        from harness.agent import AgentRecordError
        from harness.agent_simulation import run_agent_simulation
        print("Step 9 simulation: fake screens, fake Laya/Gemma, and simulated input. The real game is not controlled.", flush=True)
        try:
            record, path = run_agent_simulation(settings, job, PROJECT_ROOT / "runs")
        except (AgentRecordError, OSError) as exc:
            print("Could not persist the simulated agent record; work stopped.", file=sys.stderr)
            if isinstance(exc, AgentRecordError): print(f"Record target: {exc.path}")
            return 5
        print(f"Simulation: {record['status']}\nReason: {record['reason']}\nRecord: {path}\n"
              f"Simulated inputs: {record['completed_inputs']}; model calls: {record['model_calls']}\n"
              f"Simulated elapsed: {record['elapsed_seconds']:g}s; job limit: {job.seconds:g}s\n"
              "Human review requests: 0. Real game inputs: 0. Live integration is the next checkpoint.")
        return record["exit_code"]
    if args.laya_test:
        from models.laya_local import run_laya_test
        print("Offline Laya test: text only, original-workflow comparisons, and token-budget checks. No screen capture, API requests, or game input.", flush=True)
        try:
            record, path = run_laya_test(settings, PROJECT_ROOT / "runs", prompt=prompt, device=args.laya_device)
        except OSError:
            print("Could not persist the local Laya test record.", file=sys.stderr)
            return 5
        print(f"Laya test: {record['status']}\nRecord: {path}\nElapsed: {record['elapsed_seconds']:g}s")
        result = record.get("result") or {}
        if result.get("runtime"):
            print(f"Device: {result['runtime'].get('actual_device', 'not loaded')}")
        if result.get("status") == "passed":
            print(f"Warm inference median: {result['warm_median_seconds']:g}s\nOriginal-workflow modal agreement: {json.dumps(result['baseline_modal_agreement'])}")
            print(f"CPU inference fallbacks: {result['cpu_fallback_count']}")
            print("Typed output and offline inference verified. CUA semantic guardrail is not validated or enabled.")
            print("First text decision (model claim):\n" + json.dumps(result['cases'][0]['prediction']['answers'], indent=2))
        elif record.get("error"):
            print(f"Stopped: {record['error']}", file=sys.stderr)
            if result.get("error"):
                print(f"Stage: {result.get('stage')}; {result['error']}", file=sys.stderr)
        return record["exit_code"]
    if args.shadow or args.shadow_trace or args.transfer_test:
        from harness.shadow import run_shadow_trace, run_transfer_evaluation
        from models.laya_local import LayaError
        print("Shadow mode: decisions are recorded only. No input will execute or replay.", flush=True)
        trace_write_error_type = ()
        try:
            if args.shadow:
                from computer.capture import MacWindowBackend
                from harness.dry_run import TraceWriteError, run_dry_run
                trace_write_error_type = TraceWriteError
                trace, trace_path = run_dry_run(settings, prompt, MacWindowBackend(), PROJECT_ROOT / "runs/calibration.json",
                                               PROJECT_ROOT / "screenshots", PROJECT_ROOT / "runs")
                print(f"Fresh dry run: {trace['status']}\nTrace: {trace_path}")
                if trace["exit_code"] != 0:
                    print("Capture/planning did not complete; Laya was not called.", file=sys.stderr)
                    if trace.get("error"):
                        print(trace["error"]["message"].replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
                    return trace["exit_code"]
                record, path = run_shadow_trace(settings, trace_path, PROJECT_ROOT / "runs", device=args.laya_device)
            elif args.shadow_trace:
                record, path = run_shadow_trace(settings, Path(args.shadow_trace).expanduser(), PROJECT_ROOT / "runs", device=args.laya_device)
            else:
                record, path = run_transfer_evaluation(settings, PROJECT_ROOT / "runs", PROJECT_ROOT / "tests/fixtures/shadow_development.json", device=args.laya_device)
        except LayaError as exc:
            print(str(exc).replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
            return 4
        except OSError:
            print("Could not read or persist shadow artifacts; work stopped.", file=sys.stderr)
            return 5
        except Exception as exc:
            if isinstance(exc, trace_write_error_type):
                print(f"Could not persist the fresh dry-run trace: {exc.path}", file=sys.stderr)
                return 5
            print("Shadow system error; no input executed.", file=sys.stderr)
            return 4
        print(f"Shadow status: {record['status']}\nRecord: {path}\nExecution calls: 0")
        if record.get("runtime"):
            print(f"Laya device: {record['runtime']['actual_device']}; checkpoint loading: {record['load_seconds']:g}s")
        for decision in record["decisions"]:
            interpreted = decision["interpretation"]
            print(f"{decision['name']}: {interpreted['label']} (top probability {interpreted['top_answer_probability']:g}); {interpreted['diagnostic_disposition']}")
        if args.transfer_test and record.get("evaluation"):
            print("Development evaluation:\n" + json.dumps(record["evaluation"], indent=2))
        if record.get("error"):
            print(record["error"].replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
        print("Model outcomes are claims. Semantic enforcement remains disabled; no task completion verified.")
        return record["exit_code"]
    if args.dry_run:
        from computer.capture import MacWindowBackend
        from harness.dry_run import TraceWriteError, run_dry_run
        print("Dry run: one fresh capture and Gemma proposal, with durable trace checkpoints. No input or Laya inference will execute.", flush=True)
        try:
            trace, path = run_dry_run(
                settings, prompt, MacWindowBackend(), PROJECT_ROOT / "runs/calibration.json",
                PROJECT_ROOT / "screenshots", PROJECT_ROOT / "runs",
            )
        except TraceWriteError as exc:
            print(str(exc), file=sys.stderr)
            print(f"Trace target (may contain an earlier snapshot): {exc.path}")
            return 5
        report = (f"Run: {trace['run_id']}\nDry-run status: {trace['status']}\nTrace: {path}\n"
                  f"Game screenshot: {trace['artifacts']['image'] or 'not captured'}\n")
        if trace["proposal"] is not None:
            report += "Validated proposal (model claims):\n" + json.dumps(trace["proposal"], indent=2, ensure_ascii=False) + "\n"
        report += ("Execution calls: 0. Task outcome: not assessed; no post-action observation.\n"
                   f"Total elapsed: {trace['latency_seconds']['total']:g}s")
        print(report.replace(settings.gemini_api_key.get_secret_value(), "[hidden]"))
        if trace["error"] is not None:
            print(f"Dry-run stopped: {trace['error']['message']}".replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
        return trace["exit_code"]
    if args.action is not None or args.input_test:
        from computer.capture import CaptureError, load_calibration
        from computer.geometry import GeometryError
        from harness.actions import ActionError, parse_action, validate_game_action
        try:
            if args.input_test:
                from computer.input_test import run_input_test
                result = run_input_test(stop_file)
                report = f"Disposable input test passed: {result}"
            else:
                action = parse_action(args.action)
                profile = PROJECT_ROOT / "runs/calibration.json"
                validate_game_action(action, load_calibration(profile))
                if args.execute:
                    from computer.executor import execute_game_action
                    print(f"One action will execute after {args.start_delay:g} seconds. Bring melonDS forward; Ctrl+C cancels.", flush=True)
                    time.sleep(args.start_delay)
                    result = execute_game_action(action, profile, stop_file)
                    report = f"Your request: {prompt}\nManual action executed once: {result}\nNo model called."
                else:
                    report = (f"Your request: {prompt}\nValidated manual action: {action.model_dump_json()}\n"
                              "Preview only. Add --execute to send this one action; focus and geometry are checked then.")
        except KeyboardInterrupt:
            print("Cancelled; any held controls were released.")
            return 130
        except (ActionError, CaptureError, GeometryError) as exc:
            print(f"Input blocked: {exc}".replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
            return 3
        except Exception as exc:
            from computer.executor import ExecutionError
            message = str(exc) if isinstance(exc, ExecutionError) else "System input error or fail-safe stop. Check focus, permissions, and held controls."
            print(f"Input stopped: {message}".replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
            return 3
        print(report.replace(settings.gemini_api_key.get_secret_value(), "[hidden]"))
    elif args.calibrate or args.confirm_calibration or args.capture or args.perceive or args.propose:
        # Lazy import: normal preflight must never import any screen-capture runtime.
        from computer.capture import (
            CaptureError, MacWindowBackend, calibrate, capture_observation, confirm_calibration,
        )
        from computer.geometry import GeometryError

        backend = MacWindowBackend()
        profile, images, runs = PROJECT_ROOT / "runs/calibration.json", PROJECT_ROOT / "screenshots", PROJECT_ROOT / "runs"
        try:
            if args.calibrate:
                calibration, preview = calibrate(
                    backend, profile, images, window_id=args.window_id,
                    insets=args.insets, gap=args.screen_gap,
                )
                report = (
                    f"Calibration preview: {preview}\n"
                    f"Game image: {calibration.game_crop.width}×{calibration.game_crop.height} pixels\n"
                    f"Capture scale: {calibration.scale_x:g}×{calibration.scale_y:g} pixels per desktop point\n"
                    "Review both screens and crop edges, then run --confirm-calibration.\n"
                    "No game input or model calls were performed."
                )
            elif args.confirm_calibration:
                confirm_calibration(backend, profile)
                report = "Calibration confirmed for this window position, size, and capture scale. No game input sent."
            else:
                safe_prompt = prompt.replace(settings.gemini_api_key.get_secret_value(), "[hidden]")
                observation = capture_observation(backend, profile, images, runs, safe_prompt)
                if args.propose:
                    from harness.proposals import ProposalError
                    from models.perception import PerceptionError
                    from models.planner import propose_observation
                    print("Sending both cropped DS screens and your request to Gemma for one proposal. No input will execute.", flush=True)
                    try:
                        proposal, proposal_path = propose_observation(settings, observation)
                    except (ProposalError, PerceptionError) as exc:
                        print(f"Proposal error: {exc}".replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
                        print(f"Observation: {observation['image_path']}\nProposal record: {runs / (observation['observation_id'] + '.proposal.json')}")
                        return 4
                    report = (f"Your request: {safe_prompt}\nValidated Gemma proposal ({proposal['status']}):\n"
                              f"{json.dumps(proposal['proposal'], indent=2, ensure_ascii=False)}\n"
                              f"Game screenshot: {observation['image_path']}\nProposal record: {proposal_path}\n"
                              f"API attempts: {proposal['attempts']}; elapsed: {proposal['latency_seconds']:g}s\n"
                              "Screen descriptions and targets are model claims; review the image. Expected results have not occurred.\n"
                              "Proposal only. No game input or Laya inference performed.")
                elif args.perceive:
                    from models.perception import PerceptionError, perceive_observation
                    print("Sending the cropped image of both DS screens and your request to Gemma via the Gemini API. No input will execute.", flush=True)
                    try:
                        perception, perception_path = perceive_observation(settings, observation)
                    except PerceptionError as exc:
                        print(f"Perception error: {exc}".replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
                        print(f"Observation: {observation['image_path']}\nPerception record: {runs / (observation['observation_id'] + '.perception.json')}")
                        return 4
                    report = (f"Your request: {safe_prompt}\nGemma description (model output):\n{perception['description']}\n"
                              f"Game screenshot: {observation['image_path']}\nPerception record: {perception_path}\n"
                              f"API attempts: {perception['attempts']}; elapsed: {perception['latency_seconds']:g}s\n"
                              "Perception only. No game input or Laya inference performed.")
                else:
                    report = (
                        f"Your request: {safe_prompt}\n"
                        f"Game screenshot: {observation['image_path']}\n"
                        f"Observation metadata: {observation['metadata_path']}\n"
                        "One observation saved. No model calls or game input were performed."
                    )
        except KeyboardInterrupt:
            print("Cancelled; no game input executed.")
            return 130
        except (CaptureError, GeometryError, OSError, ImportError) as exc:
            print(f"Capture error: {exc}".replace(settings.gemini_api_key.get_secret_value(), "[hidden]"), file=sys.stderr)
            return 3
        print(report.replace(settings.gemini_api_key.get_secret_value(), "[hidden]"))
    else:
        print(readiness_report(settings, prompt))
    return 0
