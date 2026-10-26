"""Protocol-only checks for owner-reversal collection; never starts an engine."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).with_name("batch_test.py")
SPEC = importlib.util.spec_from_file_location("batch_owner_collection_under_test", SOURCE)
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def done(count, reused=0):
    return f"DONE {count} 0 0 0 length 0 0 {reused}"


def protocol(rounds, *, reversed_reuse=4):
    """Two solos and two drained, overlapping assignments; replies follow each send."""
    replies = [["T 10", "T 11", done(2)], ["T 20", "T 21", done(2)]]
    for round_index, assignment in enumerate(((0, 1), (1, 0))):
        tokens = rounds[round_index]
        for prompt, slot in enumerate(assignment):
            first = tokens[prompt][:1]
            reply = [*(f"T {token}" for token in first),
                     done(len(first), reversed_reuse if round_index else 0), f"BADM {slot} 1"]
            if prompt == 1:
                for other_prompt, other_slot in enumerate(assignment):
                    reply += [f"BT {other_slot} {token}" for token in tokens[other_prompt][1:]]
                    reply += [f"BDONE {other_slot} {len(tokens[other_prompt])} length"]
            replies.append(reply)
    return replies


class OwnerCollectionTests(unittest.TestCase):
    def run_case(self, replies, *, closure_changes=None, mutate_profile=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"args": [], "sampling": {}}), encoding="utf-8")
            args = SimpleNamespace(config=str(config), exe="not-launched", max_new=2,
                                   deadline_seconds=600, dump=str(root / "result.json"))
            observations = {"commands": [], "closures": 0}
            closure = {"exit_code": 0, "forced": False, "deadline_expired": False,
                       "ram_guard": {"sampler_stopped": True, "samples": 2,
                                     "breach": False, "sampler_error": None,
                                     "minimum_available_bytes": 8 * 1024 ** 3,
                                     "floor_bytes": 7 * 1024 ** 3}}
            closure.update(closure_changes or {})

            class FakeEngine:
                def __init__(self, exe, cfg, batch, env, extra, *, owner_deadline):
                    observations["deadline"] = owner_deadline
                    self.info = {"batch_slots": "2"}
                    self.queue = []

                def send(self, command):
                    index = len(observations["commands"])
                    observations["commands"].append(command)
                    if self.queue:
                        raise AssertionError("Previous assignment was not drained before next send")
                    self.queue.extend(replies[index])

                def lines(self):
                    while True:
                        if not self.queue:
                            raise RuntimeError("Protocol exhausted before next command")
                        yield self.queue.pop(0)

                def close_owned(self):
                    observations["closures"] += 1
                    if mutate_profile:
                        config.write_text('{"changed": true}', encoding="utf-8")
                    return closure

            sampling = SimpleNamespace(sampling_keys=lambda value: " seed=1")
            server = SimpleNamespace(StrataEngine=sampling)
            with patch.object(BATCH, "Engine", FakeEngine), \
                    patch.object(BATCH.time, "monotonic", return_value=100), \
                    patch.dict("sys.modules", {"serve.server": server}), \
                    patch.dict(os.environ, {"BATCH_TEST_LOG": str(root / "native.log")}):
                status = BATCH.owner_reversal(args, {"args": []}, [[100], [200]])
            result = json.loads(Path(args.dump).read_text(encoding="utf-8"))
            self.assertEqual(observations["closures"], 1)
            self.assertEqual(observations["deadline"], 700)
            return status, result, observations

    def test_all_mismatches_collected_after_both_assignments_and_still_fail(self):
        status, result, observed = self.run_case(protocol([[[30, 31], [40, 41]], [[50, 51], [60, 61]]]))
        self.assertEqual(status, 2)
        self.assertFalse(result["pass"])
        self.assertTrue(result["collection_complete"])
        self.assertEqual(result["solo_mismatches"], [
            {"round": 0, "prompt": 0, "slot": 0}, {"round": 0, "prompt": 1, "slot": 1},
            {"round": 1, "prompt": 0, "slot": 1}, {"round": 1, "prompt": 1, "slot": 0}])
        self.assertEqual([command.split()[:2] for command in observed["commands"][2:]],
                         [["BGEN", "0"], ["BGEN", "1"], ["BGEN", "1"], ["BGEN", "0"]])
        self.assertTrue(all(row["terminal"] for turn in result["rounds"] for row in turn["records"]))
        self.assertEqual(result["error"], "ValueError: Slot output differs from its solo reference")

    def test_matching_rounds_remain_pass(self):
        status, result, observed = self.run_case(protocol([[[10, 11], [20, 21]]] * 2))
        self.assertEqual(status, 0)
        self.assertTrue(result["pass"])
        self.assertTrue(result["collection_complete"])
        self.assertEqual(result["solo_mismatches"], [])
        self.assertEqual(len(observed["commands"]), 6)

    def test_fatal_protocol_stops_immediately_and_closes(self):
        replies = protocol([[[10, 11], [20, 21]]] * 2)
        replies[2] = ["ERR injected failure"]
        status, result, observed = self.run_case(replies)
        self.assertEqual(status, 2)
        self.assertFalse(result["collection_complete"])
        self.assertEqual(len(observed["commands"]), 3)
        self.assertIn("Native engine error", result["error"])

    def test_empty_slot_is_fatal_before_reversal(self):
        status, result, observed = self.run_case(protocol([[[], [20, 21]], [[10, 11], [20, 21]]]))
        self.assertEqual(status, 2)
        self.assertFalse(result["collection_complete"])
        self.assertEqual(len(observed["commands"]), 4)
        self.assertIn("did not complete normally", result["error"])

    def test_reversed_reuse_gate_is_not_softened(self):
        status, result, _ = self.run_case(protocol([[[30, 31], [40, 41]]] * 2, reversed_reuse=0))
        self.assertEqual(status, 2)
        self.assertFalse(result["collection_complete"])
        self.assertIn("did not exercise prefix reuse", result["error"])

    def test_expired_deadline_still_fails_completed_collection(self):
        status, result, _ = self.run_case(protocol([[[10, 11], [20, 21]]] * 2),
                                          closure_changes={"deadline_expired": True})
        self.assertEqual(status, 2)
        self.assertFalse(result["pass"])
        self.assertTrue(result["collection_complete"])

    def test_changed_profile_still_fails_completed_collection(self):
        status, result, _ = self.run_case(protocol([[[10, 11], [20, 21]]] * 2), mutate_profile=True)
        self.assertEqual(status, 2)
        self.assertFalse(result["pass"])
        self.assertEqual(result["error"], "Profile changed during owner reversal")


if __name__ == "__main__":
    unittest.main()
