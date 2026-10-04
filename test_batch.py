"""Deterministic protocol tests for Strata's two-slot batch mode."""
import ast
import queue
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name("strata_grpc_backend.py")
MODULE = ast.parse(SOURCE.read_text())
NODES = [node for node in MODULE.body if getattr(node, "name", "") in ("EngineError", "Engine")]
NS = {
    "os": __import__("os"),
    "queue": queue,
    "subprocess": subprocess,
    "threading": threading,
    "time": time,
    "Path": Path,
    "READY_TIMEOUT_S": 1,
    "GEN_TIMEOUT_S": 2,
}
exec(compile(ast.Module(body=NODES, type_ignores=[]), str(SOURCE), "exec"), NS)
Engine = NS["Engine"]
EngineError = NS["EngineError"]


class FakeProc:
    def poll(self):
        return None


class ProtocolHarness:
    def __init__(self):
        self.engine = Engine({"parallel": 2})
        self.engine.proc = FakeProc()
        self.engine.ended = False
        self.engine.can_stop = True
        self.commands = []
        self.admitted = []
        self.lock = threading.Lock()
        self.both_admitted = threading.Event()
        self.engine._send = self.send

    def route(self, line):
        self.engine._route_line(line + "\n")

    def send(self, line):
        with self.lock:
            self.commands.append(line)
            parts = line.split()
            if parts[0] in ("BGEN", "BGENI"):
                slot = int(parts[1])
                # Admission has legacy-shaped output that belongs to no active slot.
                self.route("T 999")
                self.route("DONE")
                self.route(f"BADM {slot} 1")
                self.admitted.append(slot)
                if len(self.admitted) == 2:
                    first, second = self.admitted
                    self.route(f"BT {second} 201")
                    self.route(f"BT {first} 101")
                    self.route(f"BT {second} 202")
                    self.route(f"BDONE {second} 2 stop 4.25")
                    self.route(f"BT {first} 102")
                    self.route(f"BDONE {first} 2 stop 5")
                    self.both_admitted.set()


class BatchProtocolTests(unittest.TestCase):
    def test_parallel_defaults_to_legacy_serial_protocol(self):
        engine = Engine({})
        self.assertEqual(engine.parallel, 1)
        self.assertEqual(engine.command_args(["strata", "--serve"]), ["strata", "--serve"])

    def test_parallel_two_adds_batch_flag_only_without_upstream_batch_syntax(self):
        engine = Engine({"parallel": 2})
        base = ["strata", "--serve"]
        self.assertEqual(engine.command_args(base), base + ["--batch", "2"])
        explicit_batch = base + ["--batch", "4"]
        explicit_slots = base + ["--slots", "4"]
        explicit_equals = base + ["--batch=4"]
        self.assertEqual(engine.command_args(explicit_batch), explicit_batch)
        self.assertEqual(engine.command_args(explicit_slots), explicit_slots)
        self.assertEqual(engine.command_args(explicit_equals), explicit_equals)
        with self.assertRaisesRegex(ValueError, "parallel must be 1 or 2"):
            Engine({"parallel": 3})

    def test_two_sessions_route_interleaved_tokens_to_their_own_slots(self):
        harness = ProtocolHarness()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(lambda: list(harness.engine.batch_generate(8, "", [11])))
            b = pool.submit(lambda: list(harness.engine.batch_generate(8, "", [22])))
            self.assertEqual(a.result(timeout=2), [999, 101, 102])
            self.assertEqual(b.result(timeout=2), [999, 201, 202])
        self.assertEqual(len([c for c in harness.commands if c.startswith("BGEN ")]), 2)
        self.assertEqual(harness.engine.available_slots(), 2)

    def test_vision_uses_bgeni_with_request_slot_and_embedding_path(self):
        engine = Engine({"parallel": 2})
        engine.proc = FakeProc()
        engine.ended = False
        sent = []
        def send(line):
            sent.append(line)
            slot = int(line.split()[1])
            engine._route_line(f"BADM {slot} 1\n")
            engine._route_line(f"BDONE {slot} 0 stop 1\n")
        engine._send = send
        self.assertEqual(list(engine.batch_generate(9, " temperature=0.2", [1, 2], Path("/tmp/request.sve"))), [])
        self.assertRegex(sent[0], r"^BGENI [01] 9 temperature=0\.2 /tmp/request\.sve 1,2$")

    def test_cancel_stops_and_drains_only_its_slot_before_reuse(self):
        engine = Engine({"parallel": 2})
        engine.proc = FakeProc()
        engine.ended = False
        engine.can_stop = True
        sent = []
        generations = 0
        def send(line):
            nonlocal generations
            sent.append(line)
            parts = line.split()
            if parts[0] == "BGEN":
                generations += 1
                slot = int(parts[1])
                engine._route_line(f"BADM {slot} 1\n")
                engine._route_line(f"BT {slot} {100 + generations}\n")
                if generations == 2:
                    engine._route_line(f"BDONE {slot} 1 stop 1\n")
            elif parts[0] == "BSTOP":
                slot = int(parts[1])
                engine._route_line(f"BT {slot} 999\n")
                engine._route_line(f"BDONE {slot} 2 cancelled 1\n")
        engine._send = send

        first = engine.batch_generate(8, "", [1])
        self.assertEqual(next(first), 101)
        first.close()
        self.assertTrue(any(command.startswith("BSTOP ") for command in sent))
        self.assertEqual(list(engine.batch_generate(8, "", [2])), [102])
        self.assertEqual(engine.available_slots(), 2)

    def test_non_admitted_request_returns_its_serial_fallback_tokens(self):
        engine = Engine({"parallel": 2})
        engine.proc = FakeProc()
        engine.ended = False
        def send(line):
            slot = int(line.split()[1])
            engine._route_line("T 301\n")
            engine._route_line("T 302\n")
            engine._route_line("DONE\n")
            engine._route_line(f"BADM {slot} 0\n")
        engine._send = send
        self.assertEqual(list(engine.batch_generate(8, "", [7])), [301, 302])
        self.assertEqual(engine.available_slots(), 2)

    def test_batch_drain_never_reuses_a_slot_without_bdone(self):
        engine = Engine({"parallel": 2})
        engine.proc = FakeProc()
        engine.ended = False
        with self.assertRaisesRegex(EngineError, "without BDONE"):
            engine.drain(timeout_s=0.01, slot=0)

    def test_engine_must_report_at_least_two_batch_slots(self):
        engine = Engine({"parallel": 2})
        engine.info["batch_slots"] = 1
        with self.assertRaisesRegex(EngineError, "batch_slots=1"):
            engine.validate_batch_slots()
        engine.info["batch_slots"] = 4
        engine.validate_batch_slots()


if __name__ == "__main__":
    unittest.main()
