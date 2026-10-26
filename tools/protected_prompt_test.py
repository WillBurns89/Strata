#!/usr/bin/env python3
"""One fixed fresh 32K prompt for protected expert-residency diagnosis.

Use Python -B -X utf8, --config PROFILE, --prompt-file the archived rendered
fixture, --dump FRESH.json and a fresh BATCH_TEST_LOG. The caller supplies the
external 660-second owned-job/resource guard and checks actual expert maps and
trace evidence. This is a diagnostic, not scored performance or quality acceptance.
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

from batch_test import Engine, tokenizer
from cancel_park_capture import capture_native_modules
from owner_fork_test import progress_event


RAW_SHA256 = "d72fc6d38709dd07e6685e3a17efa59935da66d40c206bb82f6368635e927112"
TEXT_SHA256 = "96e41e29e67b579e57c81d7346886ae7266708820828007db1d7ce0b54810e4e"
IDS_SHA256 = "71412d6282b7cf7277d92dffe00496a588c858cddfd3332c6f57f261a48c2183"
INPUT_TOKENS = 32768
MAX_NEW = 64
DEADLINE_SECONDS = 600


def collect_prompt(lines, row):
    """Accept one fresh GEN, preserving its complete native terminal and tokens."""
    for line in lines:
        fields = line.split()
        if not fields:
            raise ValueError("Empty native prompt event")
        kind = fields[0]
        if progress_event(fields, row, False):
            continue
        if kind == "RESUME":
            if len(fields) != 2 or row["resume"] is not None or row["tokens"] or int(fields[1]) != 0:
                raise ValueError("Prompt must be fresh with exactly one RESUME 0")
            row["resume"] = 0
        elif kind == "T":
            if (len(fields) != 2 or row["resume"] != 0 or row.get("prefill_reused") != 0
                    or int(fields[1]) < 0 or len(row["tokens"]) >= MAX_NEW):
                raise ValueError("Invalid token outside the fresh prompt budget")
            row["tokens"].append(int(fields[1]))
        elif kind == "DONE":
            row["terminal"] = line
            if len(fields) < 9:
                raise ValueError("Incomplete native DONE")
            generated, prompt, reused = int(fields[1]), int(fields[2]), int(fields[8])
            prompt_ms, decode_ms = float(fields[3]), float(fields[4])
            if (row["resume"] != 0 or row.get("prefill_reused") != 0 or reused != 0
                    or prompt != INPUT_TOKENS or generated != len(row["tokens"])
                    or not 1 <= generated <= MAX_NEW or fields[5] not in ("stop", "length")
                    or not all(math.isfinite(value) and value >= 0 for value in (prompt_ms, decode_ms))):
                raise ValueError("Native completion does not match the fresh32768/GEN64 contract")
            row.update(reused=reused, generated=generated, finish=fields[5],
                       native_prompt_ms=prompt_ms, native_decode_ms=decode_ms,
                       drafts_accepted=int(fields[6]), drafts_offered=int(fields[7]))
            if len(fields) >= 14:
                row["native_decode_tiers"] = {
                    "ram_blobs": int(fields[11]), "file_blobs": int(fields[12]),
                    "file_mb": float(fields[13]),
                    "scope": "Native DONE decode counters; excludes prompt and restoration traffic",
                }
            return
        else:
            raise ValueError("Unexpected native prompt event: " + line)
    raise ValueError("Native EOF before prompt completion")


def validate_profile(cfg):
    args = cfg["args"]
    for key, expected in (("--batch", "2"), ("--max-context", "262144"),
                          ("--kv", "fp16"), ("--kv-resident", "20480"), ("--spec", "4")):
        if args.count(key) != 1 or args[args.index(key) + 1] != expected:
            raise ValueError("UNSUPPORTED profile setting: " + key)
    if "--vision" not in args or "--mtp" not in args:
        raise ValueError("Profile must retain vision and MTP assets")
    if str(cfg.get("env", {}).get("STRATA_TRACE")) != "1":
        raise ValueError("Both diagnostic arms require STRATA_TRACE=1")
    if not os.environ.get("BATCH_TEST_LOG") or Path(os.environ["BATCH_TEST_LOG"]).exists():
        raise ValueError("A fresh BATCH_TEST_LOG is required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    args = parser.parse_args()
    with args.dump.open("x", encoding="utf-8") as output:
        started = time.monotonic()
        deadline = started + DEADLINE_SECONDS
        row = {"prompt_tokens": [], "tokens": [], "resume": None, "terminal": None}
        result = {"schema": 1, "pass": False, "events": [], "request": row,
                  "deadline_seconds": DEADLINE_SECONDS, "external_guard_required": True,
                  "timing_mode": "diagnostic_with_trace_and_module_capture_not_scored",
                  "native_log": os.environ.get("BATCH_TEST_LOG"),
                  "scope": "One fixed fresh32768-position prompt, greedy GEN64 maximum",
                  "expected_secret": "3f4c1b76ac41b889",
                  "limitations": ["No BGEN, vision request, full-context or held-out quality claim",
                                  "Caller must validate actual expert maps, trace traffic and cross-arm output",
                                  "DONE file counters are decode-only; startup cumulative counters are not prompt bytes"]}
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
                                       for name in ("batch_test.py", "cancel_park_capture.py",
                                                    "owner_fork_test.py", "protected_prompt_test.py")}
            validate_profile(cfg)
            raw = args.prompt_file.read_bytes()
            if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
                raise ValueError("Archived prompt byte identity changed")
            text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != TEXT_SHA256:
                raise ValueError("Normalized prompt identity changed")
            tok = tokenizer(cfg["tokenizer"])
            ids = tok.encode(text, parse_special=True)
            encoded_ids = ",".join(map(str, ids))
            if len(ids) != INPUT_TOKENS or hashlib.sha256(encoded_ids.encode("utf-8")).hexdigest() != IDS_SHA256:
                raise ValueError("Fixture must produce the exact32768 pinned token IDs")
            row["prompt_tokens"] = ids
            result["fixture"] = {"path": str(args.prompt_file), "raw_sha256": RAW_SHA256,
                                 "normalized_utf8_sha256": TEXT_SHA256, "token_ids_comma_sha256": IDS_SHA256}
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
            import psutil
            process = psutil.Process(engine.p.pid)
            created = result["native_module_capture"]["creation_time"]
            if process.create_time() != created or engine.p.poll() is not None:
                raise ValueError("Native identity changed before GEN I/O snapshot")
            io_before = process.io_counters()._asdict()
            row["process_io_before_gen"] = io_before
            request_started = time.monotonic()
            row["command_prefix"] = f"GEN {MAX_NEW} {keys}"
            engine.send(row["command_prefix"] + " " + encoded_ids)

            def captured_lines():
                previous = request_started
                for line in engine.lines():
                    now = time.monotonic()
                    result["events"].append({"elapsed_seconds": now - started,
                                             "request_seconds": now - request_started, "line": line})
                    if now < previous or now >= deadline:
                        raise TimeoutError("Absolute deadline expired or monotonic clock regressed")
                    previous = now
                    yield line

            collect_prompt(captured_lines(), row)
            row["request_wall_seconds"] = time.monotonic() - request_started
            if psutil.Process(process.pid).create_time() != created or engine.p.poll() is not None:
                raise ValueError("Native identity changed after DONE I/O snapshot")
            io_after = process.io_counters()._asdict()
            row["process_io_after_done"] = io_after
            deltas = {key: io_after[key] - io_before[key] for key in ("read_count", "read_bytes")}
            if any(value < 0 for value in deltas.values()):
                raise ValueError("Native process I/O counters regressed")
            row["process_read_transfer_delta"] = {
                **deltas,
                "scope": "Windows process I/O immediately before GEN through observed DONE, before QUIT; "
                         "includes prompt, restoration, decoding and other process reads, including pipes; "
                         "not physical SSD traffic or expert-only bytes",
            }
            row["decoded_output"] = tok.decode(row["tokens"])
            result.update(collection_complete=True, **{"pass": True})
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
                    result["config_changed"] = True
                    result["pass"] = False
            except Exception as exc:
                result["config_recheck_error"] = str(exc)
                result["pass"] = False
            result["elapsed_seconds"] = time.monotonic() - started
            json.dump(result, output, indent=2)
            output.write("\n")
            output.flush()
    print(("PASS" if result["pass"] else "FAIL") + " fixed32K prompt: " + str(args.dump))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
