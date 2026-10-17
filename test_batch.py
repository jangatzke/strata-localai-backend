"""Deterministic protocol tests for Strata's batch slots (up to eight)."""
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
    "PROGRESS_STALL_S": 180,
    "DECODE_PROGRESS_S": 1.0,
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
    def test_pending_admission_cancel_retains_owner_until_reconciled(self):
        for reason in ('cancelled', 'stopping'):
            for accepted in (True, False):
                with self.subTest(reason=reason, accepted=accepted):
                    engine = Engine({'parallel': 2})
                    engine.proc = FakeProc()
                    engine.ended = False
                    engine.can_stop = True
                    sent = threading.Event()
                    cancelled = threading.Event()
                    stopped = threading.Event()
                    peer_sent = threading.Event()
                    commands = []
                    slots = []
                    def route(text):
                        engine._route_line(text + '\n')
                    def send(line):
                        commands.append(line)
                        parts = line.split()
                        slot = int(parts[1])
                        if parts[0] == 'BGEN':
                            slots.append(slot)
                            if len(slots) == 1:
                                sent.set()
                            else:
                                route(f'BADM {slot} 1')
                                route(f'BT {slot} 202')
                                route(f'BDONE {slot} 1 stop 1')
                                peer_sent.set()
                        elif parts[0] == 'BSTOP':
                            stopped.set()
                    engine._send = send
                    def check():
                        if cancelled.is_set():
                            raise EngineError(reason)
                    pool = ThreadPoolExecutor(max_workers=2)
                    a = pool.submit(lambda: list(engine.batch_generate(8, '', [1], check_active=check)))
                    try:
                        self.assertTrue(sent.wait(1))
                        cancelled.set()
                        peer = pool.submit(lambda: list(engine.batch_generate(8, '', [2])))
                        observed_stop = stopped.wait(1.5)
                        early_peer = peer_sent.is_set()
                        owner_finished = a.done()
                        # Simulate delayed prefill finishing only AFTER cancellation.
                        route('T 999')
                        route('DONE')
                        route(f'BADM {slots[0]} {int(accepted)}')
                        if accepted:
                            route(f'BT {slots[0]} 666')
                            route(f'BDONE {slots[0]} 1 cancelled 1')
                        with self.assertRaisesRegex(EngineError, reason):
                            a.result(3)
                        self.assertEqual(peer.result(3), [202])
                        self.assertTrue(observed_stop, 'pending BGEN cancellation was not polled')
                        self.assertFalse(early_peer, 'unreconciled admission owner was reused')
                        self.assertFalse(owner_finished, 'pending numeric slot was released')
                        self.assertEqual(engine.available_slots(), 2)
                        self.assertEqual([c for c in commands if c.startswith('BSTOP')], [f'BSTOP {slots[0]}'])
                        self.assertTrue(engine.lines.empty())
                    finally:
                        pool.shutdown(wait=True)

    def test_unreconciled_admission_quarantines_slot_without_stopping_active_peer(self):
        engine = Engine({'parallel': 2})
        engine.proc = FakeProc()
        engine.ended = False
        engine.can_stop = True
        commands = []
        clock = [0.0]
        generations = []
        def send(line):
            commands.append(line)
            if line.startswith('BGEN '):
                slot = int(line.split()[1])
                generations.append(slot)
                if len(generations) == 1:
                    engine._route_line(f'BADM {slot} 1\n')
                    engine._route_line(f'BT {slot} 101\n')
            elif line.startswith('BSTOP '):
                # Advance across the cleanup bound after the next empty read.
                engine.lines = ExpiringQueue()
        class ExpiringQueue(queue.Queue):
            def get(self, *args, **kwargs):
                clock[0] += 121
                raise queue.Empty
        saved_time, saved_timeout = NS['time'], NS['GEN_TIMEOUT_S']
        NS['time'] = SimpleNamespace(time=lambda: clock[0])
        NS['GEN_TIMEOUT_S'] = 3600
        peer = engine.batch_generate(8, '', [1])
        engine._send = send
        try:
            self.assertEqual(next(peer), 101)
            checks = []
            def cancel_after_send():
                checks.append(1)
                if len(generations) == 2:
                    raise EngineError('cancelled')
            with self.assertRaisesRegex(EngineError, 'unreconciled'):
                list(engine.batch_generate(8, '', [2], check_active=cancel_after_send))
            self.assertEqual(engine.available_slots(), 0, 'unresolved slot must be quarantined')
            with self.assertRaisesRegex(EngineError, 'unreconciled'):
                list(engine.batch_generate(8, '', [3]))
            engine._route_line(f'BT {generations[0]} 102\n')
            engine._route_line(f'BDONE {generations[0]} 2 stop 1\n')
            self.assertEqual(list(peer), [102])
            self.assertEqual(engine.available_slots(), 1)
            self.assertEqual([c for c in commands if c.startswith('BSTOP')], [f'BSTOP {generations[1]}'])
        finally:
            peer.close()
            NS['time'], NS['GEN_TIMEOUT_S'] = saved_time, saved_timeout

    def test_admission_timeout_reconciles_late_badm_before_raising(self):
        engine = Engine({'parallel': 2})
        engine.proc = FakeProc()
        engine.ended = False
        engine.can_stop = True
        clock = [0.0]
        commands = []
        class DelayedQueue(queue.Queue):
            def get(self, *args, **kwargs):
                if self.empty():
                    clock[0] += 3
                    raise queue.Empty
                return super().get(*args, **kwargs)
        engine.lines = DelayedQueue()
        def send(line):
            commands.append(line)
            if line.startswith('BSTOP '):
                slot = int(line.split()[1])
                engine._route_line(f'BADM {slot} 1\n')
                engine._route_line(f'BDONE {slot} 0 cancelled 1\n')
        engine._send = send
        saved = NS['time']
        NS['time'] = SimpleNamespace(time=lambda: clock[0])
        try:
            with self.assertRaisesRegex(EngineError, 'timed out'):
                list(engine.batch_generate(8, '', [1]))
            self.assertFalse(engine.admission_broken)
            self.assertEqual(engine.available_slots(), 2)
            self.assertEqual(commands, ['BGEN 0 8 1', 'BSTOP 0'])
            self.assertTrue(engine.lines.empty())
        finally:
            NS['time'] = saved

    def test_parallel_defaults_to_legacy_serial_protocol(self):
        engine = Engine({})
        self.assertEqual(engine.parallel, 1)
        self.assertEqual(engine.command_args(["strata", "--serve"]), ["strata", "--serve"])

    def test_parallel_two_to_eight_add_batch_flag_without_overriding_engine_args(self):
        base = ["strata", "--serve"]
        for count in range(2, 9):
            with self.subTest(parallel=count):
                engine = Engine({"parallel": count})
                self.assertEqual(engine.command_args(base), base + ["--batch", str(count)])
                self.assertEqual(engine.available_slots(), count)
                self.assertEqual(set(engine.slot_lines), set(range(count)))
                for explicit in (["--batch", str(count)], ["--slots", str(count)],
                                 [f"--batch={count}"], [f"--slots={count}"]):
                    self.assertEqual(engine.command_args(base + explicit), base + explicit)
        for count in (0, 9):
            with self.assertRaisesRegex(ValueError, "parallel must be between 1 and 8"):
                Engine({"parallel": count})

    def test_eight_slots_route_interleaved_output_and_reuse_each_slot(self):
        engine = Engine({'parallel': 8})
        engine.info['batch_slots'] = 8
        engine.validate_batch_slots()
        engine.proc = FakeProc()
        engine.ended = False
        assigned = {}
        active = []
        guard = threading.Lock()

        def send(line):
            with guard:
                parts = line.split()
                slot, marker = int(parts[1]), int(parts[-1])
                assigned[marker] = slot
                engine._route_line(f'BADM {slot} 1\n')
                active.append(slot)
                if len(active) == 8:
                    for slot_id in reversed(active):
                        engine._route_line(f'BT {slot_id} {100 + slot_id}\n')
                        engine._route_line(f'BDONE {slot_id} 1 stop 1\n')
                    active.clear()
        engine._send = send
        for start in (11, 21):
            markers = list(range(start, start + 8))
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda marker: list(engine.batch_generate(8, '', [marker])), markers))
            self.assertEqual(set(assigned[m] for m in markers), set(range(8)))
            self.assertEqual(results, [[100 + assigned[m]] for m in markers])
            self.assertEqual(engine.available_slots(), 8)

    def test_requested_slot_count_must_fit_engine_report(self):
        for count in range(2, 9):
            with self.subTest(parallel=count):
                engine = Engine({'parallel': count})
                for actual in (None, count - 1):
                    engine.info['batch_slots'] = actual
                    with self.assertRaisesRegex(EngineError, f'parallel={count}.*batch_slots={actual}'):
                        engine.validate_batch_slots()
                engine.info['batch_slots'] = count
                engine.validate_batch_slots()

    def test_three_slots_route_interleaved_tokens_to_their_owners(self):
        engine = Engine({'parallel': 3})
        self.assertEqual(engine.command_args(['strata', '--serve']),
                         ['strata', '--serve', '--batch', '3'])
        engine.info['batch_slots'] = 3
        engine.validate_batch_slots()
        engine.proc = FakeProc()
        engine.ended = False
        admitted = []
        assignments = {}
        commands = []
        guard = threading.Lock()
        def send(line):
            with guard:
                commands.append(line)
                slot = int(line.split()[1])
                assignments[int(line.split()[-1])] = slot
                admitted.append(slot)
                engine._route_line('T 999\n')
                engine._route_line('DONE\n')
                engine._route_line(f'BADM {slot} 1\n')
                if len(admitted) == 3:
                    for slot_id in reversed(admitted):
                        engine._route_line(f'BT {slot_id} {100 + slot_id}\n')
                    for slot_id in admitted:
                        engine._route_line(f'BDONE {slot_id} 2 stop 4.25\n')
        engine._send = send
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda token: list(engine.batch_generate(8, '', [token])), (11, 22, 33)))
        self.assertEqual(len(admitted), 3)
        self.assertEqual(set(admitted), {0, 1, 2})
        self.assertEqual(results, [[999, 100 + assignments[token]] for token in (11, 22, 33)])
        self.assertEqual(engine.available_slots(), 3)
        self.assertEqual(len([c for c in commands if c.startswith('BGEN ')]), 3)

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
