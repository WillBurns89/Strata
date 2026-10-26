#!/usr/bin/env python3
"""Bounded active-peer prefix fork, cancellation and solo-MTP resume diagnostic.

Run with Python -B -X utf8, --config PROFILE --dump FRESH.json and a fresh
BATCH_TEST_LOG. The caller owns the external 660-second process guard, profile
placement checks, trace checks and cross-arm token comparison. No speed claim.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from batch_test import Engine, QUESTIONS, tokenizer
from cancel_park_capture import capture_native_modules


SUFFIX = "\nGive one concise explanation with a concrete example."
DEADLINE_SECONDS = 600


def progress_event(fields, row, batch_telemetry):
    """Validate the only nonterminal diagnostics this request sequence can emit."""
    kind = fields[0]
    if kind == "BSTAT":
        # The final window's sample may arrive after BDONE, including before the next GEN's RESUME.
        if not batch_telemetry or len(fields) != 5:
            raise ValueError("Unexpected or malformed BSTAT")
        windows, processed, active, elapsed = map(int, fields[1:])
        if not (0 < windows <= active <= processed and elapsed > 0):
            raise ValueError("Invalid BSTAT counters")
        return True
    if kind not in ("PP", "REUSED"):
        return False
    if (row is None or row["resume"] is None or row["tokens"]
            or row.get("done") is not None or row.get("prefill_reused") is not None):
        raise ValueError("Unexpected prompt progress outside prefill: " + kind)
    if kind == "REUSED":
        if len(fields) != 2 or int(fields[1]) != row["resume"]:
            raise ValueError("Malformed REUSED or disagreement with RESUME")
        row["prefill_reused"] = int(fields[1])
    else:
        if len(fields) != 5:
            raise ValueError("Malformed PP")
        reached, total = map(int, fields[1:3])
        elapsed, rate = map(float, fields[3:])
        if (total != len(row["prompt_tokens"])
                or not row.get("progress_position", row["resume"]) <= reached <= total
                or not all(math.isfinite(x) and x >= 0 for x in (elapsed, rate))):
            raise ValueError("Invalid PP progress or timing")
        row["progress_position"] = reached
    return True


def collect_batch(lines, record, *, batch_telemetry=False):
    """Require both admissions before any batch token; retain partial evidence."""
    rows = record["records"]
    active = set()
    admitted = 0
    for line in lines:
        fields = line.split()
        if not fields:
            raise ValueError("Empty native batch event")
        kind = fields[0]
        if kind == "ERR":
            raise ValueError("Native error: " + line)
        if progress_event(fields, rows[admitted] if admitted < 2 else None, batch_telemetry):
            continue
        if kind in ("RESUME", "T", "DONE", "BADM"):
            if admitted >= 2:
                raise ValueError("Unexpected admission event after both admissions: " + line)
            row = rows[admitted]
            if kind == "RESUME":
                if len(fields) != 2 or row["resume"] is not None or row["tokens"] or row["done"] is not None:
                    raise ValueError("Duplicate or late RESUME: " + line)
                row["resume"] = int(fields[1])
                if not 0 <= row["resume"] <= len(row["prompt_tokens"]):
                    raise ValueError("RESUME lies outside the prompt")
            elif kind == "T":
                if (len(fields) != 2 or row["done"] is not None or row["resume"] is None
                        or row.get("prefill_reused") != row["resume"]):
                    raise ValueError("Admission token outside RESUME/DONE: " + line)
                row["tokens"].append(int(fields[1]))
            elif kind == "DONE":
                if (row["done"] is not None or len(row["tokens"]) != 1
                        or int(fields[1]) != 1 or fields[5] != "length"
                        or int(fields[2]) != len(row["prompt_tokens"])
                        or int(fields[8]) != row["resume"]):
                    raise ValueError("Unsupported admission DONE: " + line)
                row.update(done=line, reused=int(fields[8]))
                if admitted == 1 and row["reused"] != record["prefix_length"]:
                    raise ValueError("Fork did not reuse exactly the shared prefix")
            else:
                if (len(fields) != 3 or row["done"] is None or int(fields[1]) != row["slot"]
                        or fields[2] != "1" or row["slot"] in active):
                    raise ValueError("UNSUPPORTED: requested slot was not admitted")
                row["admission"] = line
                active.add(row["slot"])
                admitted += 1
                record["both_slots_active"] = len(active) == 2
        elif kind in ("BT", "BDONE"):
            if admitted != 2 or not record["both_slots_active"]:
                raise ValueError("UNSUPPORTED: batch ran before both queued admissions")
            slot = int(fields[1])
            if slot not in active:
                raise ValueError("Batch event for inactive slot: " + line)
            row = next(row for row in rows if row["slot"] == slot)
            if kind == "BT":
                if len(fields) != 3:
                    raise ValueError("Malformed BT: " + line)
                row["tokens"].append(int(fields[2]))
            else:
                if (len(fields) != 5 or len(row["tokens"]) != 2 or int(fields[2]) != 2
                        or fields[3] != row["expected_finish"]):
                    raise ValueError("Unsupported batch completion: " + line)
                row["terminal"] = line
                active.remove(slot)
                if not active:
                    return
        else:
            raise ValueError("Unexpected native batch event: " + line)
    raise ValueError("Native EOF before both batch completions")


def collect_solo(lines, row, *, batch_telemetry=False):
    for line in lines:
        fields = line.split()
        if not fields:
            raise ValueError("Empty native solo event")
        kind = fields[0]
        if kind in ("ERR", "BT", "BDONE", "BADM"):
            raise ValueError("Unexpected solo event: " + line)
        if progress_event(fields, row, batch_telemetry):
            continue
        if kind == "RESUME":
            if len(fields) != 2 or row["resume"] is not None or row["tokens"]:
                raise ValueError("Duplicate or late solo RESUME")
            row["resume"] = int(fields[1])
            if not 0 <= row["resume"] <= len(row["prompt_tokens"]):
                raise ValueError("Solo RESUME lies outside the prompt")
        elif kind == "T":
            if (len(fields) != 2 or row["resume"] is None
                    or row.get("prefill_reused") != row["resume"]):
                raise ValueError("Solo token before RESUME")
            row["tokens"].append(int(fields[1]))
        elif kind == "DONE":
            row.update(terminal=line, reused=int(fields[8]), draft_offered=int(fields[7]))
            if (len(row["tokens"]) != 32 or int(fields[1]) != 32 or fields[5] != "length"
                    or int(fields[2]) != len(row["prompt_tokens"])
                    or row["resume"] != row["expected_reused"]
                    or row["reused"] != row["expected_reused"] or row["draft_offered"] <= 0):
                raise ValueError("UNSUPPORTED: solo resume lacks expected reuse, 32 outputs or MTP drafts")
            return
        else:
            raise ValueError("Unexpected native solo event: " + line)
    raise ValueError("Native EOF before solo completion")


def validate_profile(cfg):
    args = cfg["args"]
    for key, expected in (("--batch", "2"), ("--max-context", "262144"),
                          ("--kv", "fp16"), ("--spec", "4")):
        if args.count(key) != 1 or args[args.index(key) + 1] != expected:
            raise ValueError("UNSUPPORTED profile setting: " + key)
    if "--vision" not in args or "--mtp" not in args:
        raise ValueError("Profile must retain vision and MTP assets")
    for key, value in (("STRATA_TRACE", "1"), ("STRATA_BATCH_MTP", "0")):
        if str(cfg.get("env", {}).get(key)) != value:
            raise ValueError("Profile must explicitly set " + key + "=" + value)
    if not os.environ.get("BATCH_TEST_LOG") or Path(os.environ["BATCH_TEST_LOG"]).exists():
        raise ValueError("A fresh BATCH_TEST_LOG is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    args = parser.parse_args()
    # Reserve evidence before any native launch. No overwrites or retries.
    with args.dump.open("x", encoding="utf-8") as output:
        started = time.monotonic()
        deadline = started + DEADLINE_SECONDS
        result = {"schema": 1, "pass": False, "rounds": [], "events": [],
                  "expected_output_tokens": 72, "deadline_seconds": DEADLINE_SECONDS,
                  "external_guard_required": True, "timing_mode": "diagnostic_not_scored",
                  "native_log": os.environ.get("BATCH_TEST_LOG"),
                  "scope": "Active-peer shared-prefix fork, cancel, then target-state solo-MTP resume",
                  "limitations": ["No per-slot draft-KV-transfer claim with batch MTP disabled",
                                  "No solo-versus-batch parity, vision or long-context qualification",
                                  "Caller must check trace direct-copy/giveback evidence and cross-arm parity",
                                  "Raw event capture starts after READY; startup INFO is parsed by Engine"]}
        engine = None
        config_bytes = None
        try:
            if not sys.flags.utf8_mode:
                raise ValueError("Run with -X utf8")
            config_bytes = args.config.read_bytes()
            cfg = json.loads(config_bytes)
            result.update(config_path=str(args.config), config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                          exact_args=cfg["args"], exact_env=cfg.get("env", {}))
            result["helper_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("batch_test.py", "cancel_park_capture.py", "owner_fork_test.py")}
            validate_profile(cfg)
            tok = tokenizer(cfg["tokenizer"])
            suffix = tok.encode(SUFFIX, parse_special=True)
            if len(suffix) < 4:
                raise ValueError("Fixed suffix must contain at least four tokens")
            result.update(suffix_text=SUFFIX, suffix_tokens=suffix)
            prompts = [tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                                  "<think>\n\n</think>\n\n", parse_special=True) for q in QUESTIONS[:2]]
            if any(not p or len(p) + len(suffix) >= 512 for p in prompts):
                raise ValueError("UNSUPPORTED: prompts must remain short single-piece admissions")
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from serve.server import StrataEngine
            sampling = {**cfg.get("sampling", {}), "temperature": 0, "seed": 1}
            keys = "temperature=0" + StrataEngine.sampling_keys(sampling)
            result.update(effective_sampling=sampling, native_sampling_keys=keys)
            if time.monotonic() >= deadline:
                raise TimeoutError("Absolute deadline expired before launch")
            engine = Engine(cfg["exe"], cfg, 0, {}, [], owner_deadline=deadline)
            result.update(ready_observed=True, info=engine.info, native_module_capture={})
            if engine.info.get("batch_slots") != "2":
                raise ValueError("UNSUPPORTED: engine lacks two independent batch slots")
            capture_native_modules(SimpleNamespace(proc=engine.p), cfg, result["native_module_capture"])

            def captured_lines():
                previous = started
                for line in engine.lines():
                    now = time.monotonic()
                    result["events"].append({"elapsed_seconds": now - started, "line": line})
                    if now < previous or now >= deadline:
                        raise TimeoutError("Absolute deadline expired or monotonic clock regressed")
                    previous = now
                    yield line

            lines = captured_lines()
            for index, (a, b) in enumerate(((0, 1), (1, 0))):
                prefix = prompts[index]
                rows = [{"role": role, "slot": slot, "prompt_tokens": prompt, "tokens": [],
                         "resume": None, "done": None, "admission": None, "terminal": None,
                         "expected_finish": finish}
                        for role, slot, prompt, finish in (("A", a, prefix, "cancel"),
                                                          ("B", b, prefix + suffix, "length"))]
                record = {"round": index, "prefix_length": len(prefix), "records": rows,
                          "both_slots_active": False}
                result["rounds"].append(record)
                commands = [f"BGEN {row['slot']} {32 if row['role'] == 'A' else 2} {keys} "
                            + ",".join(map(str, row["prompt_tokens"])) for row in rows]
                commands.append(f"BSTOP {a}")
                record["queued_commands"] = commands
                engine.send("\n".join(commands))  # Exactly one write and one flush for all three commands.
                telemetry = str(cfg.get("env", {}).get("STRATA_BATCH_TELEMETRY")) == "1"
                collect_batch(lines, record, batch_telemetry=telemetry)
                solo = {"prompt_tokens": prefix + rows[0]["tokens"], "tokens": [], "resume": None,
                        "expected_reused": len(prefix) + 1, "terminal": None}
                record["solo"] = solo
                solo["command"] = f"GEN 32 {keys} " + ",".join(map(str, solo["prompt_tokens"]))
                engine.send(solo["command"])
                collect_solo(lines, solo, batch_telemetry=telemetry)
            result["actual_output_tokens"] = sum(len(row["tokens"]) for r in result["rounds"]
                                                  for row in [*r["records"], r["solo"]])
            if result["actual_output_tokens"] != 72:
                raise ValueError("Unexpected output count")
            result["collection_complete"] = True
            result["pass"] = True
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "closure"):
                result["closure"] = exc.closure
        finally:
            if engine is not None:
                try:
                    result["closure"] = engine.close_owned()
                except Exception as exc:
                    result["closure_error"] = f"{type(exc).__name__}: {exc}"
                    result["pass"] = False
            closure = result.get("closure", {})
            ram = closure.get("ram_guard", {})
            if (closure.get("exit_code") != 0 or closure.get("forced") or closure.get("deadline_expired")
                    or closure.get("close_errors") or not ram.get("sampler_stopped") or not ram.get("samples")
                    or ram.get("breach") or ram.get("sampler_error") is not None
                    or ram.get("minimum_available_bytes", 0) < 7 * 1024 ** 3):
                result["pass"] = False
            try:
                if config_bytes is not None and args.config.read_bytes() != config_bytes:
                    result.update(config_changed=True)
                    result["pass"] = False
            except Exception as exc:
                result["config_recheck_error"] = str(exc)
                result["pass"] = False
            result["elapsed_seconds"] = time.monotonic() - started
            json.dump(result, output, indent=2)
            output.write("\n")
            output.flush()
    print(("PASS" if result["pass"] else "FAIL") + " active-peer fork/cancel/solo-resume: " + str(args.dump))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
