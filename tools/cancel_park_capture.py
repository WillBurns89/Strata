#!/usr/bin/env python3
"""Capture the unchanged parked-cancellation case under an external owned-process guard.

Run with Python -B -X utf8, a config path, and --dump FRESH.json. The original
case's exit or exception is preserved; callers must also require capture_pass
and their independent resource/process-closure gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


CASE_SHA256 = "1451ed33cb913d193ab09125cedb48b243da3324959ab7e51814f9fa58348921"
CAPTURE_KEYS = (
    "A1", "B1", "A2", "A3", "outA", "outB", "outA3", "r0", "rb",
    "reused2", "fin2", "stopped", "reused3", "want",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    args = parser.parse_args()
    if not sys.flags.utf8_mode:
        raise RuntimeError("Run this case with -X utf8")
    for name in ("STRATA_EXTRA_ARGS", "STRATA_EXE", "SMOKE_LOG", "STRATA_REPO"):
        if name in os.environ:
            raise RuntimeError("Unreviewed case override is present: " + name)
    case = Path(__file__).resolve().with_name("cancel_park_test.py")
    source = case.read_bytes()
    if hashlib.sha256(source).hexdigest() != CASE_SHA256:
        raise RuntimeError("Cancellation case identity changed")
    config_bytes = args.config.read_bytes()
    # Reserve the output before any native process can start; never overwrite evidence.
    with args.dump.open("x", encoding="utf-8") as output:
        os.environ["STRATA_START_RETRIES"] = "0"
        os.environ["STRATA_KEEP_LEFTOVERS"] = "1"
        namespace = {"__name__": "__main__", "__file__": str(case)}
        previous_argv = sys.argv
        sys.argv = [str(case), str(args.config)]
        receipt = {
            "schema": 1,
            "scope": "Sequential parked-session cancellation; no active-peer claim",
            "case_sha256": CASE_SHA256,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "original_exit": None,
            "exception": None,
            "capture_pass": False,
            "external_guard_required": True,
        }
        try:
            exec(compile(source, str(case), "exec"), namespace)
            receipt["original_exit"] = 0
        except SystemExit as exc:
            receipt["original_exit"] = exc.code
            raise
        except BaseException as exc:
            receipt["exception"] = {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            raise
        finally:
            sys.argv = previous_argv
            receipt["case_values"] = {key: namespace[key] for key in CAPTURE_KEYS if key in namespace}
            engine = namespace.get("eng")
            last = getattr(engine, "last", {})
            receipt["engine_last"] = last
            receipt["engine_info"] = getattr(engine, "info", {})
            receipt["native_exit_code"] = getattr(getattr(engine, "proc", None), "returncode", None)
            checks = {
                "original_pass": receipt["original_exit"] == 0 and receipt["exception"] is None,
                "stop_sent": namespace.get("stopped") is True,
                "cancelled": namespace.get("fin2") == "cancel",
                "nonempty_a": bool(namespace.get("outA")),
                "nonempty_a3": bool(namespace.get("outA3")),
                "a3_normal_finish": last.get("finish") in ("stop", "length"),
                "reuse_floor": (
                    isinstance(namespace.get("reused3"), int)
                    and isinstance(namespace.get("want"), int)
                    and namespace["reused3"] >= namespace["want"]
                ),
                "native_exit_zero": receipt["native_exit_code"] == 0,
            }
            receipt["checks"] = checks
            receipt["capture_pass"] = all(checks.values())
            json.dump(receipt, output, indent=2)
            output.write("\n")
            output.flush()


if __name__ == "__main__":
    main()
