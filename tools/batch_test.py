#!/usr/bin/env python3
"""Batching test: the same prompts decoded alone (GEN, the usual path with drafts) and together in the batch
windows (BGEN), greedy.  Every slot's tokens must equal its solo tokens; prints the aggregate decode rate.

  python3 tools/batch_test.py --exe engine/strata --config strata-<model>.json --batch 4 --n 4 \
      --extra "--layer-split 12,24,36 --trim-stage-weights --pcie-frac 0 --adapt-every 1000000"
"""
import argparse, hashlib, json, math, os, subprocess, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_tokenizer as ST  # noqa: E402

QUESTIONS = [
    "Explique en detail le fonctionnement d'un B-tree.",
    "Write a Python function that merges overlapping intervals, then explain it.",
    "Compare TCP and QUIC: handshake, congestion control, multiplexing.",
    "Raconte l'histoire du calcul de pi, d'Archimede a nos jours.",
    "What are the main causes of the French Revolution?",
    "Explain how a transformer's attention works, step by step.",
    "Donne une recette de tarte aux pommes, etape par etape.",
    "Describe the life cycle of a star like the Sun.",
    "Erklaere ausfuehrlich, wie ein Dieselmotor funktioniert.",
    "Write a Rust function that parses a CSV line with quoted fields, then explain it.",
    "What were the main consequences of the invention of the printing press?",
    "Explique le theoreme de Pythagore avec trois demonstrations differentes.",
    "Describe how the immune system responds to a viral infection.",
    "Explain the difference between processes and threads in an operating system.",
    "Beschreibe die Geschichte des Roemischen Reiches in Grundzuegen.",
    "How does public-key cryptography work? Give an example with RSA.",
]
# more than 16 slots: the same questions again, asked to answer in German
QUESTIONS = QUESTIONS + [q + " Answer in German." for q in QUESTIONS]


def tokenizer(path):
    t = Path(path)
    vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for s, i in vocab.items():
        tokens[i] = s
    merges = (t / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((t / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


class Engine:
    def __init__(self, exe, cfg, batch, extra_env, extra_args, *, owner_deadline=None):
        if owner_deadline is None:
            args = list(cfg["args"])
            if len(cfg.get("gpu") or []) > 1:
                args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
            if batch:
                args += ["--batch", str(batch)]
            args += extra_args
            env = dict(os.environ, **extra_env)
            env["LD_LIBRARY_PATH"] = ":".join(cfg.get("lib_dirs", []) + [env.get("LD_LIBRARY_PATH", "")])
            if os.name == "nt":
                env["PATH"] = os.pathsep.join(cfg.get("lib_dirs", []) + [env.get("PATH", "")])
        else:
            # This opt-in diagnostic uses the normalized profile unchanged, including MTP and numerical settings.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from serve.server import child_env
            import psutil
            args, env = list(cfg["args"]), child_env(cfg)
        self.log_path = os.environ.get("BATCH_TEST_LOG") or os.path.join(tempfile.gettempdir(), "batch_test_engine.log")
        self.log = open(self.log_path, "x" if owner_deadline is not None else "w")
        self.timer, self.expired, self.info = None, False, {}
        self.ram_thread, self.ram_stop, self.ram_guard = None, None, None
        try:
            self.p = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=env)
        except BaseException:
            self.log.close()
            raise
        try:
            if owner_deadline is not None:
                def expire():
                    self.expired = True
                    try:
                        self.p.kill()  # The Popen owns this creation-qualified child handle.
                    except OSError:
                        pass
                self.timer = threading.Timer(max(0, owner_deadline - time.monotonic()), expire)
                self.timer.daemon = True
                self.timer.start()
                self.ram_stop = threading.Event()
                self.ram_guard = {"floor_bytes": 7 * 1024 ** 3, "minimum_available_bytes": None,
                                  "samples": 0, "breach": False, "sampler_error": None}

                def watch_ram():
                    while not self.ram_stop.is_set():
                        try:
                            available = int(psutil.virtual_memory().available)
                            minimum = self.ram_guard["minimum_available_bytes"]
                            self.ram_guard["minimum_available_bytes"] = available if minimum is None else min(minimum, available)
                            self.ram_guard["samples"] += 1
                            self.ram_guard["breach"] |= available < self.ram_guard["floor_bytes"]
                        except Exception as exc:
                            self.ram_guard["sampler_error"] = f"{type(exc).__name__}: {exc}"
                        if self.ram_guard["breach"] or self.ram_guard["sampler_error"] is not None:
                            try:
                                self.p.kill()
                            except OSError as exc:
                                self.ram_guard["stop_error"] = str(exc)
                            return
                        self.ram_stop.wait(1.0)

                self.ram_thread = threading.Thread(target=watch_ram, daemon=True)
                self.ram_thread.start()
            for line in self.p.stdout:
                if line.startswith("INFO "):
                    self.info.update(field.split("=", 1) for field in line.split()[1:] if "=" in field)
                if line.startswith("READY"):
                    return
            raise SystemExit(f"the engine ended before READY - see {self.log_path}")
        except BaseException as exc:
            if owner_deadline is not None:
                closure = self.close_owned()
                failure = RuntimeError(f"Startup failed: {exc}; child closure={closure}")
                failure.closure = closure
                raise failure from exc
            raise

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def lines(self):
        for line in self.p.stdout:
            yield line.rstrip("\n")


    def close_owned(self):
        forced = False
        receipt = {}
        try:
            if self.p.poll() is None:
                try:
                    self.send("QUIT")
                    self.p.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    forced = True
                    self.p.kill()
                    self.p.wait(timeout=10)
            receipt.update(exit_code=self.p.returncode, forced=forced, deadline_expired=self.expired)
            return receipt
        finally:
            if self.ram_stop is not None:
                self.ram_stop.set()
                if self.ram_thread is not None and self.ram_thread.is_alive():
                    self.ram_thread.join(timeout=2)
                receipt["ram_guard"] = {**self.ram_guard,
                                        "sampler_stopped": self.ram_thread is not None and not self.ram_thread.is_alive()}
            if self.timer is not None:
                self.timer.cancel()
                if self.timer.is_alive():
                    self.timer.join(timeout=1)
            for stream in (self.p.stdin, self.p.stdout, self.log):
                try:
                    stream.close()
                except OSError as exc:
                    receipt.setdefault("close_errors", []).append(str(exc))


def owner_reversal(a, cfg, prompts):
    """Two native slot assignments, reversed after a complete drain; no profile or arithmetic overrides."""
    result = {"mode": "owner_reversal", "pass": False, "solo": [], "rounds": [],
              "config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
              "args": list(cfg["args"]), "profile_env": cfg.get("env", {}),
              "limitations": ["Short text only: capacity and vision remain configured, not exercised here.",
                              "No active-peer matching-prefix fork, cancellation or BYIELD in this case.",
                              "Exact token parity may expose existing solo/batch arithmetic differences."]}
    eng = None
    deadline = time.monotonic() + a.deadline_seconds
    try:
        eng = Engine(a.exe, cfg, 2, {}, [], owner_deadline=deadline)
        result["engine_info"] = eng.info
        if eng.info.get("batch_slots") != "2":
            raise ValueError("UNSUPPORTED: native engine did not provide exactly two batch slots")
        from serve.server import StrataEngine
        sampling = {**cfg.get("sampling", {}), "temperature": 0, "seed": 1}
        keys = "temperature=0" + StrataEngine.sampling_keys(sampling)
        result.update(effective_sampling=sampling, native_sampling_keys=keys)
        out = eng.lines()
        for i, ids in enumerate(prompts):
            record = {"prompt": i, "tokens": [], "reused": None, "terminal": None}
            result["solo"].append(record)
            eng.send(f"GEN {a.max_new} {keys} " + ",".join(map(str, ids)))
            for line in out:
                fields = line.split()
                if line.startswith("T "):
                    record["tokens"].append(int(fields[1]))
                elif line.startswith("DONE "):
                    record.update(terminal=line, reused=int(fields[8]))
                    if (fields[5] not in ("stop", "length") or not record["tokens"]
                            or int(fields[1]) != len(record["tokens"])):
                        raise ValueError("Solo reference did not complete normally")
                    break
                elif line.startswith(("ERR", "BT ", "BDONE ", "BADM ")):
                    raise ValueError("Unexpected solo protocol: " + line)
            if record["terminal"] is None:
                raise ValueError("Engine ended before solo DONE")
        for assignment in ((0, 1), (1, 0)):
            records = [{"prompt": i, "slot": s, "tokens": [], "reused": None,
                        "admission": None, "terminal": None} for i, s in enumerate(assignment)]
            by_slot = {r["slot"]: r for r in records}
            round_record = {"records": records, "both_slots_active": False}
            result["rounds"].append(round_record)
            active = set()

            def receive(line, admitting=None):
                fields = line.split()
                if line.startswith("ERR"):
                    raise ValueError("Native engine error: " + line)
                if line.startswith("T "):
                    if admitting is None:
                        raise ValueError("Unattributed admission token")
                    admitting["tokens"].append(int(fields[1]))
                elif line.startswith("DONE "):
                    if (admitting is None or fields[5] not in ("stop", "length")
                            or admitting["reused"] is not None or int(fields[1]) != len(admitting["tokens"])):
                        raise ValueError("Unexpected admission DONE: " + line)
                    admitting.update(reused=int(fields[8]), first_window_done=line)
                elif line.startswith(("BT ", "BDONE ")):
                    slot = int(fields[1])
                    if slot not in active:
                        raise ValueError("Token/terminal for an inactive slot: " + line)
                    row = by_slot[slot]
                    if fields[0] == "BT":
                        row["tokens"].append(int(fields[2]))
                    else:
                        if fields[3] not in ("stop", "length") or int(fields[2]) != len(row["tokens"]):
                            raise ValueError("Abnormal slot completion: " + line)
                        row["terminal"] = line
                        active.remove(slot)
                elif line.startswith("BADM "):
                    if admitting is None or int(fields[1]) != admitting["slot"] or fields[2] != "1":
                        raise ValueError("UNSUPPORTED: requested slot was not admitted; no sequential fallback")
                    if admitting["reused"] is None:
                        raise ValueError("Admission lacks DONE reuse evidence")
                    admitting["admission"] = line
                    active.add(admitting["slot"])
                    round_record["both_slots_active"] |= len(active) == 2
                    return True
                return False

            for row in records:
                eng.send(f"BGEN {row['slot']} {a.max_new} {keys} " + ",".join(map(str, prompts[row["prompt"]])))
                for line in out:
                    if receive(line, row):
                        break
                if row["admission"] is None:
                    raise ValueError("Engine ended before requested BADM")
            while active:
                receive(next(out))
            if not round_record["both_slots_active"]:
                raise ValueError("UNSUPPORTED: no overlap of the two admitted slots")
            for row in records:
                row["matches_solo"] = row["tokens"] == result["solo"][row["prompt"]]["tokens"]
                if not row["matches_solo"] or not row["tokens"] or row["terminal"] is None:
                    raise ValueError("Slot output differs from its solo reference")
                if assignment == (1, 0) and row["reused"] <= 0:
                    raise ValueError("Reversed assignment did not exercise prefix reuse")
        result["pass"] = True
    except (Exception, SystemExit) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, "closure"):
            result["closure"] = exc.closure
    finally:
        if eng is not None:
            try:
                result["closure"] = eng.close_owned()
                closure = result["closure"]
                ram = closure["ram_guard"]
                if (closure["exit_code"] != 0 or closure["forced"] or closure["deadline_expired"]
                        or closure.get("close_errors") or not ram["sampler_stopped"] or ram["samples"] == 0
                        or ram["breach"] or ram["sampler_error"] is not None
                        or ram["minimum_available_bytes"] < ram["floor_bytes"]):
                    result["pass"] = False
            except Exception as exc:
                result.update({"pass": False, "cleanup_error": f"{type(exc).__name__}: {exc}"})
        native_log = Path(os.environ["BATCH_TEST_LOG"])
        result["native_log"] = str(native_log)
        if native_log.exists():
            result["owner_diagnostics"] = [line for line in native_log.read_text(errors="replace").splitlines()
                                           if "host KV borrow" in line or "actual=host-kv-borrow" in line]
        if hashlib.sha256(Path(a.config).read_bytes()).hexdigest() != result["config_sha256"]:
            result.update({"pass": False, "error": "Profile changed during owner reversal"})
        with Path(a.dump).open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    print(f"{'PASS' if result['pass'] else 'FAIL'}: two-slot owner reversal; evidence {a.dump}", flush=True)
    return 0 if result["pass"] else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=4, help="prompts (<= --batch)")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-solo", action="store_true")
    ap.add_argument("--keys", default="", help='sampling keys for every request, e.g. "temperature=0.7 top_k=20"')
    ap.add_argument("--extra", default="", help='more engine arguments in one string, e.g. "--adapt-every 1000000"')
    ap.add_argument("--dump", default="", help="write every slot's tokens (solo and batch) as JSON: an A/B of two engine builds")
    ap.add_argument("--long-tokens", type=int, default=0,
                    help="the LAST prompt is read from about this many tokens of filler text first: its chunks run "
                         "beside the earlier slots' windows (the prompt path lends experts then)")
    ap.add_argument("--mt-min", default="1", help="STRATA_IQ_MT_MIN for the engine (1: exact; empty: the default)")
    ap.add_argument("--owner-reversal", action="store_true",
                    help="two prompts: solo references, A0/B1 then A1/B0; preserve normalized profile unchanged")
    ap.add_argument("--deadline-seconds", type=float, default=600,
                    help="owner-reversal only: absolute startup/work deadline for its private engine")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.owner_reversal:
        if (a.batch != 2 or a.n != 2 or a.skip_solo or a.extra or a.keys or a.long_tokens
                or a.mt_min != "" or not 2 <= a.max_new <= 4096
                or not math.isfinite(a.deadline_seconds) or not 1 <= a.deadline_seconds <= 3600):
            ap.error("owner-reversal requires --batch 2 --n 2 --mt-min=, solos, no --extra/--keys/--long-tokens, "
                     "2..4096 output tokens and a finite 1..3600-second deadline")
        for flag, expected in (("--batch", "2"), ("--max-context", "262144"), ("--kv", "fp16"), ("--spec", "4")):
            positions = [i for i, value in enumerate(cfg["args"]) if value == flag]
            if len(positions) != 1 or cfg["args"][positions[0] + 1:positions[0] + 2] != [expected]:
                ap.error(f"UNSUPPORTED: owner-reversal requires normalized profile {flag} {expected}")
        if "--vision" not in cfg["args"] or "--mtp" not in cfg["args"]:
            ap.error("UNSUPPORTED: owner-reversal requires the profile's vision and MTP assets")
        if not a.dump or Path(a.dump).exists() or not os.environ.get("BATCH_TEST_LOG") or Path(os.environ["BATCH_TEST_LOG"]).exists():
            ap.error("owner-reversal requires fresh --dump and BATCH_TEST_LOG paths")
    tok = tokenizer(cfg["tokenizer"])
    prompts = []
    for q in QUESTIONS[: a.n]:
        text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if a.long_tokens and q is QUESTIONS[a.n - 1]:
            para = ("The committee reviewed item %d of the long agenda, noting the budget, the schedule and the open "
                    "risks, and asked the staff to report back next quarter. ")
            per = max(1, len(tok.encode(para % 100)))
            filler = "".join(para % i for i in range(a.long_tokens // per + 1))
            text = f"<|im_start|>user\n{filler}\n\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prompts.append(tok.encode(text, parse_special=True))
    if a.owner_reversal:
        return owner_reversal(a, cfg, prompts)
    env = {"STRATA_DECODE_TIMING": "1", **({"STRATA_VERIFY_PROFILE": "1"} if os.environ.get("PROF") else {})}
    if a.mt_min:
        env["STRATA_IQ_MT_MIN"] = a.mt_min
    eng = Engine(a.exe, cfg, a.batch, env, a.extra.split())
    out = eng.lines()

    solo = []
    if not a.skip_solo:
        for i, ids in enumerate(prompts):
            eng.send(" ".join(x for x in ("GEN", str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
            got, t0 = [], time.time()
            for line in out:
                if line.startswith("T "):
                    got.append(int(line.split()[1]))
                elif line.startswith("DONE") or line.startswith("ERR"):
                    print(f"solo {i}: {len(got)} tokens in {time.time() - t0:.1f}s  {line[:60]}", flush=True)
                    break
            solo.append(got)

    # batch: admit every prompt, then the slots decode together
    got = {i: [] for i in range(len(prompts))}
    done = {}
    t_admit = time.time()
    first_bt = None
    for i, ids in enumerate(prompts):
        eng.send(" ".join(x for x in ("BGEN", str(i), str(a.max_new), a.keys, ",".join(map(str, ids))) if x))
        for line in out:
            if line.startswith("T "):
                got[i].append(int(line.split()[1]))
            elif line.startswith("BT "):
                _, s, y = line.split()
                got[int(s)].append(int(y))
                first_bt = first_bt or time.time()
            elif line.startswith("BDONE "):
                done[int(line.split()[1])] = line
            elif line.startswith("ERR"):
                print("batch:", line)
                return 1
            elif line.startswith("BADM "):
                if line.split()[2] == "0":
                    done[i] = line
                break
    t_admitted = time.time()
    for line in out:
        if line.startswith("BT "):
            _, s, y = line.split()
            got[int(s)].append(int(y))
            first_bt = first_bt or time.time()
        elif line.startswith("BDONE "):
            done[int(line.split()[1])] = line
            if len(done) == len(prompts):
                break
        elif line.startswith("ERR"):
            print("batch:", line)
            return 1
    t_end = time.time()
    total = sum(len(v) for v in got.values())
    print(f"batch: {len(prompts)} slots, {total} tokens; admissions {t_admitted - t_admit:.1f}s; "
          f"then {t_end - t_admitted:.1f}s -> aggregate {total / max(t_end - t_admit, 1e-9):.1f} tok/s overall, "
          f"{sum(len(v) for v in got.values()) / max(t_end - (first_bt or t_admit), 1e-9):.1f} tok/s from the first batch token",
          flush=True)
    ok = True
    for i in range(len(prompts)):
        b = got[i]
        if solo:
            same = b == solo[i]
            ok &= same
            first_diff = next((k for k in range(min(len(b), len(solo[i]))) if b[k] != solo[i][k]), None)
            print(f"slot {i}: {len(b)} tokens, solo {len(solo[i])}: {'IDENTICAL' if same else f'DIFFERS at {first_diff}'}")
        print("   ", ascii(tok.decode(b)[:160]))
    if a.dump:
        Path(a.dump).write_text(json.dumps({"solo": solo, "batch": [got[i] for i in range(len(prompts))]}), encoding="utf-8")
    eng.send("QUIT")
    eng.p.wait(timeout=180)   # the next run needs the GPUs back
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
