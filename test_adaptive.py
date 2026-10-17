"""Deterministic, CPU-only adaptive Strata protocol tests."""
import queue
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from test_batch import Engine, EngineError, FakeProc


class AdaptiveHarness:
    def __init__(self):
        self.engine = Engine({'parallel': 2})
        self.engine.proc = FakeProc()
        self.engine.ended = False
        self.engine.can_stop = True
        self.commands = []
        self.solo_started = threading.Event()
        self.stop_sent = threading.Event()
        self.allow_done = threading.Event()
        self.auto_done = True
        self.lock = threading.Lock()
        self.engine._send = self.send

    def route(self, text):
        self.engine._route_line(text + '\n')

    def send(self, command):
        with self.lock:
            self.commands.append(command)
            parts = command.split()
            if parts[0] in ('GEN', 'GENI'):
                self.route('T 11')
                if self.auto_done:
                    self.route('DONE')
                self.solo_started.set()
            elif parts[0] == 'STOP':
                self.stop_sent.set()
            elif parts[0] in ('BGEN', 'BGENI'):
                slot = int(parts[1])
                marker = int(parts[-1].split(',')[0])
                self.route(f'T {100 + marker}')
                self.route('DONE')
                self.route(f'BADM {slot} 1')
                self.route(f'BT {slot} {110 + marker}')
                self.route(f'BDONE {slot} 1 stop 1.5')
            elif parts[0] == 'BSTOP':
                slot = int(parts[1])
                self.route(f'BDONE {slot} 0 cancelled 1')

    def finish_solo(self):
        self.allow_done.set()
        self.route('DONE')


class AdaptiveTests(unittest.TestCase):
    def test_lone_vision_uses_geni_and_preserves_image_argument(self):
        h = AdaptiveHarness()
        embedding = Path('/safe/request.sve')
        self.assertEqual(list(h.engine.adaptive_generate(3, '', [1, 2], embedding)), [11])
        self.assertEqual(h.commands, ['GENI 3 /safe/request.sve 1,2'])

    def test_active_batch_peer_survives_slot_scoped_cancel(self):
        h = AdaptiveHarness()
        h.engine.adaptive_requests = 1  # an existing request already holds the solo path
        slots = {}
        def send(command):
            h.commands.append(command)
            parts = command.split()
            if parts[0] == 'BGEN':
                slot, marker = int(parts[1]), int(parts[-1])
                slots[marker] = slot
                h.route(f'T {100 + marker}')
                h.route('DONE')
                h.route(f'BADM {slot} 1')
            elif parts[0] == 'BSTOP':
                h.route(f'BDONE {parts[1]} 0 cancelled 1')
        h.engine._send = send
        a = h.engine.adaptive_generate(4, '', [1])
        b = h.engine.adaptive_generate(4, '', [2])
        self.assertEqual(next(a), 101)
        self.assertEqual(next(b), 102)
        a.close()
        self.assertEqual([c for c in h.commands if c.startswith('BSTOP ')],
                         [f'BSTOP {slots[1]}'])
        h.route(f'BT {slots[2]} 202')
        h.route(f'BDONE {slots[2]} 1 stop 2.5')
        self.assertEqual(list(b), [202])
        self.assertEqual(h.engine.available_slots(), 2)

    def test_lone_gen_completes_without_bgen_and_next_lone_gen(self):
        h = AdaptiveHarness()
        self.assertEqual(list(h.engine.adaptive_generate(3, '', [1])), [11])
        self.assertEqual(list(h.engine.adaptive_generate(3, '', [2])), [11])
        self.assertEqual([c.split()[0] for c in h.commands], ['GEN', 'GEN'])
        self.assertEqual(h.engine.available_slots(), 2)

    def test_promotion_waits_for_done_and_preserves_prefix_and_budget(self):
        h = AdaptiveHarness()
        h.auto_done = False
        first = h.engine.adaptive_generate(4, ' seed=7', [1])
        self.assertEqual(next(first), 11)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(lambda: list(first))
            b = pool.submit(lambda: list(h.engine.adaptive_generate(4, '', [2])))
            try:
                self.assertTrue(h.stop_sent.wait(2))
                self.assertFalse(any(c.startswith('BGEN ') for c in h.commands),
                                 'no batch admission before solo DONE')
                h.finish_solo()
                self.assertEqual(a.result(3), [101, 111])
                self.assertEqual(b.result(3), [102, 112])
            finally:
                if not h.allow_done.is_set():
                    h.finish_solo()
        self.assertEqual(h.commands[0], 'GEN 4 seed=7 1')
        self.assertTrue(any(c.startswith('BGEN ') and c.endswith(' 3 seed=7 1,11')
                            for c in h.commands))
        self.assertEqual(h.engine.available_slots(), 2)
        h.auto_done = True
        self.assertEqual(list(h.engine.adaptive_generate(3, '', [3])), [11],
                         'next uncontended turn returns to GEN')
        self.assertEqual(h.commands[-1], 'GEN 3 3')

    def test_promotion_during_prefill_without_any_solo_token(self):
        h = AdaptiveHarness()
        original = h.send
        def send(command):
            if command.startswith('GEN '):
                h.commands.append(command)
                h.solo_started.set()
            else:
                original(command)
        h.engine._send = send
        with ThreadPoolExecutor(max_workers=2) as pool:
            owner = pool.submit(lambda: list(h.engine.adaptive_generate(4, '', [1])))
            self.assertTrue(h.solo_started.wait(2))
            peer = pool.submit(lambda: list(h.engine.adaptive_generate(4, '', [2])))
            try:
                self.assertTrue(h.stop_sent.wait(2))
                self.assertFalse(any(c.startswith('BGEN ') for c in h.commands))
                h.route('DONE')
                self.assertEqual(owner.result(3), [101, 111])
                self.assertEqual(peer.result(3), [102, 112])
            finally:
                if not owner.done():
                    h.route('DONE')
        self.assertTrue(any(c.startswith('BGEN ') and c.endswith(' 4 1')
                            for c in h.commands))

    def test_cancel_solo_waits_for_done_before_peer_admission(self):
        h = AdaptiveHarness()
        h.auto_done = False
        first = h.engine.adaptive_generate(4, '', [1])
        self.assertEqual(next(first), 11)
        with ThreadPoolExecutor(max_workers=2) as pool:
            peer = pool.submit(lambda: list(h.engine.adaptive_generate(4, '', [2])))
            closing = pool.submit(first.close)
            try:
                self.assertTrue(h.stop_sent.wait(2))
                self.assertFalse(any(c.startswith('BGEN ') for c in h.commands))
                h.finish_solo()
                closing.result(3)
                self.assertEqual(peer.result(3), [102, 112])
            finally:
                if not h.allow_done.is_set():
                    h.finish_solo()
        self.assertEqual(h.engine.available_slots(), 2)

    def test_solo_err_is_terminal_and_next_request_not_offset(self):
        h = AdaptiveHarness()
        h.engine._send = lambda cmd: (h.commands.append(cmd), h.route('ERR refused')) if cmd.startswith('GEN ') else h.send(cmd)
        with self.assertRaisesRegex(EngineError, 'refused'):
            list(h.engine.adaptive_generate(4, '', [1]))
        self.assertEqual(h.commands, ['GEN 4 1'])
        self.assertFalse(h.engine.admission_broken)
        h.engine._send = h.send
        self.assertEqual(list(h.engine.adaptive_generate(4, '', [2])), [11])
        self.assertEqual(h.commands[-1], 'GEN 4 2')

    def test_stop_finish_does_not_promote_completed_solo(self):
        h = AdaptiveHarness()
        h.auto_done = False
        first = h.engine.adaptive_generate(4, '', [1])
        self.assertEqual(next(first), 11)
        with ThreadPoolExecutor(max_workers=2) as pool:
            owner = pool.submit(lambda: list(first))
            peer = pool.submit(lambda: list(h.engine.adaptive_generate(4, '', [2])))
            try:
                self.assertTrue(h.stop_sent.wait(2))
                h.route('DONE 1 1 1 0 stop')
                self.assertEqual(owner.result(3), [])
                self.assertEqual(peer.result(3), [102, 112])
            finally:
                if not owner.done():
                    h.route('DONE 1 1 1 0 stop')
        self.assertFalse(any(c.startswith('BGEN ') and c.endswith(' 1,11')
                             for c in h.commands))


if __name__ == '__main__':
    unittest.main()
