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
import time
import traceback


CASE_SHA256 = "1451ed33cb913d193ab09125cedb48b243da3324959ab7e51814f9fa58348921"
CAPTURE_KEYS = (
    "A1", "B1", "A2", "A3", "outA", "outB", "outA3", "r0", "rb",
    "reused2", "fin2", "stopped", "reused3", "want",
)


def capture_native_modules(engine, config, record):
    """Observe the constructor's actual child once; no discovery, retry or process control."""
    import psutil

    def normalized(path):
        return os.path.normcase(os.path.abspath(path))

    process = psutil.Process(engine.proc.pid)
    created = process.create_time()
    record.update(pid=process.pid, creation_time=created, exe=process.exe(),
                  cmdline=process.cmdline(), parent_pid=process.ppid())
    if normalized(record["exe"]) != normalized(config["exe"]):
        raise RuntimeError("Actual native executable differs from config")
    if record["parent_pid"] != os.getpid():
        raise RuntimeError("Native process is not this helper's direct child")
    paths = sorted({entry.path for entry in process.memory_maps(grouped=True)
                    if entry.path.lower().endswith((".dll", ".exe"))})
    record["actual_mapped_image_paths"] = paths
    after = psutil.Process(process.pid)
    if after.create_time() != created or engine.proc.poll() is not None:
        raise RuntimeError("Native identity changed or exited during module capture")
    allowed_dirs = {normalized(path) for path in config.get("lib_dirs", [])}
    for name in ("cublas64_13.dll", "cublasLt64_13.dll"):
        found = [path for path in paths if Path(path).name.casefold() == name.casefold()]
        if len(found) != 1 or normalized(Path(found[0]).parent) not in allowed_dirs:
            raise RuntimeError("Actual CUDA library path differs from config: " + name)
    record["identity_and_module_paths_verified"] = True


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
    config = json.loads(config_bytes)
    # Reserve the output before any native process can start; never overwrite evidence.
    with args.dump.open("x", encoding="utf-8") as output:
        os.environ["STRATA_START_RETRIES"] = "0"
        os.environ["STRATA_KEEP_LEFTOVERS"] = "1"
        namespace = {"__name__": "__main__", "__file__": str(case)}
        previous_argv = sys.argv
        previous_path = list(sys.path)
        engine_class = original_init = None
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
            "timing_mode": "diagnostic_with_synchronous_module_capture_not_scored",
            "native_module_capture": {
                "boundary": "Constructor returned READY, before the original test receives its engine",
                "identity_and_module_paths_verified": False,
                "error": None,
            },
        }
        try:
            sys.path.insert(0, str(case.parent))
            sys.path.insert(0, str(case.parent.parent))
            from serve.server import StrataEngine
            engine_class, original_init = StrataEngine, StrataEngine.__init__

            def observed_init(engine, *constructor_args, **constructor_kwargs):
                original_init(engine, *constructor_args, **constructor_kwargs)
                record = receipt["native_module_capture"]
                started = time.monotonic()
                record["started_unix"] = time.time()
                try:
                    capture_native_modules(engine, config, record)
                except Exception as exc:
                    record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                finally:
                    record["elapsed_seconds"] = time.monotonic() - started

            StrataEngine.__init__ = observed_init
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
            if engine_class is not None:
                engine_class.__init__ = original_init
            sys.argv = previous_argv
            sys.path[:] = previous_path
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
                "native_modules_verified": receipt["native_module_capture"]["identity_and_module_paths_verified"],
            }
            receipt["checks"] = checks
            receipt["capture_pass"] = all(checks.values())
            json.dump(receipt, output, indent=2)
            output.write("\n")
            output.flush()


if __name__ == "__main__":
    main()
