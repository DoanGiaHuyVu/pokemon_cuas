"""Isolated offline Laya test process; imports Torch only after offline setup."""

import importlib.metadata
import json
import os
import shutil
import socket
import statistics
import sys
import time
from pathlib import Path

from harness.journal import atomic_json
from models.laya_local import ARTIFACTS, LayaError, load_questions, sha256, validate_prediction


def original_examples(source: Path, questions: dict) -> tuple[list[dict], dict]:
    dataset = source / "data/typed_decisions/test.jsonl"
    if not dataset.is_file():
        raise LayaError("Original-workflow comparison dataset is missing.")
    selected = {}
    with dataset.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("questions") != questions or not isinstance(row.get("gold"), dict):
                continue
            label = row["gold"].get("action", {}).get("label")
            if label in {"continue", "human_review", "observe", "stop"} and label not in selected:
                selected[label] = {"name": f"original_{label}", "state": row["state"], "gold": row["gold"],
                                   "line": line_number}
            if len(selected) == 4:
                break
    if len(selected) != 4:
        raise LayaError("Original-workflow dataset lacks four comparable action examples.")
    return list(selected.values()), {"path": str(dataset), "sha256": sha256(dataset),
                                    "selection": "first matching example for each modal gold action",
                                    "independent_holdout_verified": False,
                                    "purpose": "four-example compatibility pilot, not an accuracy benchmark"}


def main(request_path: Path, result_path: Path) -> int:
    start = time.perf_counter()
    report = {"status": "in_progress", "stage": "prepare", "cases": [], "network_attempts_blocked": 0}
    agent = None

    def save():
        report["elapsed_seconds"] = round(time.perf_counter() - start, 4)
        atomic_json(result_path, report)

    def deny_network(*args, **kwargs):
        report["network_attempts_blocked"] += 1
        raise LayaError("Network connections are disabled in the offline Laya worker.")

    # Defense in addition to HF's offline flags, set by the parent before process startup.
    # This is a Python socket guard, not an OS-level network sandbox.
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_IMPLICIT_TOKEN="1", USE_TF="0")
    try:
        request = json.loads(request_path.read_text())
        checkpoint, source = Path(request["checkpoint"]), Path(request["source"])
        questions = load_questions(checkpoint)
        report["question_definitions"] = questions
        before = {name: sha256(checkpoint / name) for name in ARTIFACTS}
        report["checkpoint_sha256"] = before
        # Laya's compatibility loader may repair tokenizer metadata. Give it private copies.
        runtime = result_path.parent / "checkpoint"
        runtime.mkdir()
        for name in ARTIFACTS:
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name == "model.safetensors":
                target.symlink_to(checkpoint / name)
            else:
                shutil.copyfile(checkpoint / name, target)
        config = json.loads((runtime / "rl_agent_config.json").read_text())
        if config.get("encoder") != "answerdotai/ModernBERT-large":
            raise LayaError("This adapter expects the supplied ModernBERT-large checkpoint.")
        max_len, head_max_len = config["max_len"], config["head_max_len"]
        if type(max_len) is not int or type(head_max_len) is not int or not 0 < head_max_len < max_len <= 1024:
            raise LayaError("Unexpected checkpoint token budget.")
        live = request.get("mode") == "live"
        shadow = request.get("mode") == "shadow"
        if shadow:
            states = request.get("states")
            if not isinstance(states, list) or not 1 <= len(states) <= 32 or any(
                not isinstance(item, dict) or set(item) != {"name", "state"}
                or not isinstance(item["state"], str) or not 1 <= len(item["state"]) <= 5000
                or not isinstance(item["name"], str) or not 1 <= len(item["name"]) <= 96 for item in states
            ):
                raise LayaError("Invalid bounded text-only shadow batch.")
        elif not live:
            examples, provenance = original_examples(source, questions)
            report["baseline_dataset"] = provenance
        report["token_budget"] = {"max_len": max_len, "head_max_len": head_max_len}
        report["stage"] = "import"
        save()
        import_start = time.perf_counter()
        sys.path.insert(0, str(source))
        import torch
        import laya
        if Path(laya.__file__).resolve() != (source / "laya/__init__.py").resolve():
            raise LayaError("Laya import did not resolve to the configured local source.")
        torch.set_num_threads(2)
        mps_available = torch.backends.mps.is_available()
        requested = request["device"]
        if requested == "mps" and not mps_available:
            raise LayaError("MPS was requested but is unavailable; choose CPU explicitly.")
        device = ("mps" if mps_available else "cpu") if requested == "auto" else requested
        report["runtime"] = {"requested_device": requested, "selected_device": device,
                             "mps_available": mps_available, "cpu_threads": 2,
                             "cpu_selection_reason": "MPS unavailable" if requested == "auto" and device == "cpu" else None,
                             "laya_source_file": str(Path(laya.__file__).resolve()), "laya_version": laya.__version__,
                             "versions": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "safetensors")}}
        report["import_seconds"] = round(time.perf_counter() - import_start, 4)
        report["stage"] = "load"
        save()
        load_start = time.perf_counter()
        agent = laya.load(str(runtime), device=device, backend="eager")
        report["load_seconds"] = round(time.perf_counter() - load_start, 4)
        report["runtime"]["actual_device"] = str(agent.device)
        if agent.device.type != device:
            raise LayaError("Loader could not retain the selected device; rerun with --laya-device cpu for an explicit fallback test.")
        report["calibration"] = {"stored_temperature": config.get("temperature"),
                                 "applied_temperature": agent.temperature,
                                 "applied_temperature_by_options": agent.temperature_by_options,
                                 "cua_calibration_validated": False}
        report["source_sha256"] = {str(path.relative_to(source)): sha256(path)
                                   for path in sorted((source / "laya").rglob("*.py"))}
        report["stage"] = "infer"
        save()

        forwards = []

        def before_forward(model, inputs):
            device_type = inputs[0].device.type
            autocast = torch.is_autocast_enabled(device_type)
            forwards.append({"rows": int(inputs[0].shape[0]), "sequence_tokens": int(inputs[0].shape[1]),
                             "device": str(inputs[0].device), "autocast": autocast,
                             "dtype": str(torch.get_autocast_dtype(device_type) if autocast else next(model.parameters()).dtype),
                             "completed": False})

        def after_forward(model, inputs, outputs):
            forwards[-1]["completed"] = True

        agent.model.register_forward_pre_hook(before_forward)
        agent.model.register_forward_hook(after_forward)

        def evaluate(name, state, *, gold=None, line=None):
            fallback_before = agent.cpu_fallback_count
            forwards.clear()
            before_call = time.perf_counter()
            result = agent.predict(state, questions, max_len=max_len, head_max_len=head_max_len)
            if str(agent.device) == "mps":
                torch.mps.synchronize()
            duration = time.perf_counter() - before_call
            validation = validate_prediction(result, questions, max_len)
            case = {"name": name, "state": state, "seconds": round(duration, 4), "prediction": result,
                    "validation": validation, "runtime_device": str(agent.device),
                    "forward_batches": list(forwards),
                    "cpu_fallbacks": agent.cpu_fallback_count - fallback_before}
            if gold is not None:
                expected = {key: value["label"] for key, value in gold.items()}
                case.update(dataset_line=line, gold=gold,
                            modal_label_agreement={key: validation["modal_labels"][key] == expected[key] for key in questions})
            report["cases"].append(case)
            save()
            return case

        if live:
            # One isolated model load, one fresh prediction per bounded request.
            # Results have fixed sequential paths; stdin cannot choose output files.
            report.update(status="ready", stage="ready", checkpoint_unchanged=True)
            save()
            for sequence in range(1, 49):
                line = sys.stdin.readline(70000)
                if not line: break
                item = json.loads(line)
                if (not line.endswith('\n') or not isinstance(item, dict) or set(item) != {'name', 'state'}
                        or not isinstance(item['name'], str) or not 1 <= len(item['name']) <= 96
                        or not isinstance(item['state'], str) or not 1 <= len(item['state']) <= 5000):
                    raise LayaError('Invalid live text request.')
                report.update(status='in_progress', stage='infer', cases=[])
                evaluate(item['name'], item['state'])
                if before != {name: sha256(checkpoint / name) for name in ARTIFACTS}:
                    raise LayaError('Source checkpoint changed during live inference.')
                if report['network_attempts_blocked']:
                    raise LayaError('Offline runtime attempted a network connection.')
                report.update(status='passed', stage='query_complete', checkpoint_unchanged=True,
                              sequence=sequence, cpu_fallback_count=agent.cpu_fallback_count)
                save()
                atomic_json(result_path.parent / f'query-{sequence}.json', report)
            report.update(status='passed', stage='complete')
            return 0

        if shadow:
            for item in states:
                evaluate(item["name"], item["state"])
            if before != {name: sha256(checkpoint / name) for name in ARTIFACTS}:
                raise LayaError("Source checkpoint changed during shadow inference.")
            if report["network_attempts_blocked"]:
                raise LayaError("Offline runtime attempted a network connection.")
            report.update(status="passed", stage="complete", checkpoint_unchanged=True,
                          cpu_fallback_count=agent.cpu_fallback_count)
            save()
            return 0
        small_state = request.get("prompt") or "User asks the agent to inspect a local game title screen and propose the next action. No keyboard or mouse input has executed. The run is paused for review."
        cold = evaluate("small_text_first", small_state)
        warm = [evaluate(f"small_text_warm_{i + 1}", small_state) for i in range(3)]
        if any(case["prediction"]["answers"] != cold["prediction"]["answers"] for case in warm):
            raise LayaError("Repeated identical text produced different answers.")
        for example in examples:
            evaluate(example["name"], example["state"], gold=example["gold"], line=example["line"])
        long_case = evaluate("synthetic_token_overflow", "Read-only trace event. " * 1200)
        if not long_case["prediction"]["usage"]["truncated"] or long_case["validation"]["structurally_eligible"]:
            raise LayaError("Overflow text was not flagged as truncated and ineligible.")
        if any(case["prediction"]["usage"]["truncated"] for case in report["cases"][:-1]):
            raise LayaError("A representative small trace was unexpectedly truncated.")
        if before != {name: sha256(checkpoint / name) for name in ARTIFACTS}:
            raise LayaError("Source checkpoint changed during the test.")
        report["checkpoint_unchanged"] = True
        report["runtime_metadata_changes"] = [name for name in ARTIFACTS if name != "model.safetensors" and sha256(runtime / name) != before[name]]
        report["warm_median_seconds"] = round(statistics.median(case["seconds"] for case in warm), 4)
        report["cpu_fallback_count"] = agent.cpu_fallback_count
        report["baseline_modal_agreement"] = {
            key: {"matched": sum(case["modal_label_agreement"][key] for case in report["cases"] if "gold" in case), "total": 4}
            for key in questions
        }
        if report["network_attempts_blocked"]:
            raise LayaError("Offline runtime attempted a network connection.")
        report.update(status="passed", stage="complete")
    except Exception as exc:
        report.update(status="failed", error_class=type(exc).__name__,
                      error=str(exc) if isinstance(exc, LayaError) else "Local runtime failed; exception details withheld.")
    finally:
        if agent is not None:
            agent.__exit__(None, None, None)
        save()
    return 0 if report["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
