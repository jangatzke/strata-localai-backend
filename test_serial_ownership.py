"""CPU-only deterministic reproduction against unchanged deployed source.
No GPU, subprocess, network, dependencies or repository modifications.
Real AST-extracted serial generation/stream/drain, virtual clock and engine queue.
"""
import ast
import codecs
import json
import queue
import re
import sys
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('strata_grpc_backend.py')
root = ast.parse(SOURCE.read_text())
names = {'EngineError', 'Engine', 'ToolFormatError', 'ToolCallStreamParser',
         '_tool_system_block', '_generation_limit'}
nodes = [n for n in root.body if getattr(n, 'name', '') in names]
backend = next(n for n in root.body if isinstance(n, ast.ClassDef) and n.name == 'StrataBackend')
methods = [n for n in backend.body if getattr(n, 'name', '') in ('_stream', '_generate')]
nodes += methods

class Clock:
    now = 0.0
    def time(self): return self.now
    def monotonic(self): return self.now
clock = Clock()
PB = SimpleNamespace(Reply=lambda **kw: SimpleNamespace(**kw),
                     ChatDelta=lambda **kw: SimpleNamespace(**kw),
                     ToolCallDelta=lambda **kw: SimpleNamespace(**kw))
ns = dict(queue=queue, threading=threading, time=clock, codecs=codecs, json=json,
          re=re, uuid=uuid, Path=Path, pb=PB, os=SimpleNamespace(_exit=lambda code: None),
          GEN_TIMEOUT_S=3600, PROGRESS_STALL_S=180,
          IM_START='<|' + 'im_start' + '|>', IM_END='<|' + 'im_end' + '|>')
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)

class ScheduledQueue:
    def __init__(self): self.events = []
    def put_at(self, at, line): self.events.append((at, line)); self.events.sort(key=lambda x: x[0])
    def get(self, timeout):
        if self.events and self.events[0][0] <= clock.now + timeout:
            at, line = self.events.pop(0)
            clock.now = max(clock.now, at)
            return line
        clock.now += timeout
        raise queue.Empty

class Engine(ns['Engine']):
    def __init__(self):
        super().__init__({'parallel': 1})
        self.ended = False
        self.can_stop = True
        self.lines = ScheduledQueue()
        self.commands = []
        self.owner = None
        self.context = None
        self.completed = []
    def alive(self): return True
    def _send(self, command):
        self.commands.append((clock.now, command))
        if command.startswith('GEN '):
            owner = int(command.split()[-1])
            # Engine continues old prefill despite STOP until its next checkpoint.
            at = max(4.0, clock.now + 1.0)
            if self.lines.events:
                at = max(at, self.lines.events[-1][0] + 1.0)
            self.lines.put_at(at, 'PP 80941 80946 100 1000' if owner == 1 else 'PP 1 1 1 1000')
            self.lines.put_at(at, 'T ' + str(owner + 10))
            self.lines.put_at(at, 'T 99')
            self.lines.put_at(at, 'DONE')
            if owner == 1: self.context.active = False
        # STOP is not an acknowledgement: actual terminal output remains scheduled.

class Backend:
    _generate = ns['_generate']
    _stream = ns['_stream']
    stop_ids = {99}
    think_open_ids = set()
    think_close_ids = set()
    cfg = {}
    stopping = False
    def __init__(self):
        self.engine = Engine()
        self.gen_lock = threading.Semaphore(1)
        self.tok = SimpleNamespace(encode=lambda p, parse_special: [int(p)],
                                   token_bytes=lambda tid: ('answer-' + str(tid - 10)).encode())
    def _ensure_loaded(self): pass
    def _check_identity(self, identity): pass
    def _sampling_keys(self, opts): return ''

def opts(owner):
    return SimpleNamespace(Prompt=str(owner), Tools='', ModelIdentity='', Tokens=5, StopPrompts=[])

class Context:
    active = True
    def is_active(self): return self.active



class Repro(unittest.TestCase):
    def setUp(self): clock.now = 0.0
    def test_serial_timeout_quarantines_engine_before_next_gen(self):
        b = Backend()
        with self.assertRaisesRegex(ns['EngineError'], 'drain'):
            b.engine.drain(timeout_s=2)
        with self.assertRaisesRegex(ns['EngineError'], 'unreconciled'):
            next(b._generate([2], 5, ''))
        self.assertEqual(b.engine.commands, [])

    def test_quarantined_stream_rejects_without_another_stop_drain(self):
        b = Backend()
        b.engine.admission_broken = True
        with self.assertRaisesRegex(ns['EngineError'], 'unreconciled'):
            list(b._stream(opts(2)))
        self.assertEqual(b.engine.commands, [])
        self.assertEqual(clock.now, 0, 'quarantined engine must not wait through another drain')

    def test_serial_error_does_not_replace_terminal_acknowledgement(self):
        e = Engine()
        e.lines.put_at(0, 'ERR failed request')
        e.lines.put_at(4, 'DONE')
        e.drain()
        self.assertEqual(e.lines.events, [])
        self.assertEqual(clock.now, 4)

    def test_serial_drain_waits_for_delayed_terminal(self):
        e = Engine()
        e.lines.put_at(4.0, 'DONE')
        e.drain()
        self.assertEqual(clock.now, 4.0)
        self.assertEqual(e.lines.events, [], 'serial drain must consume delayed DONE')
    def test_cancelled_prefill_preserves_successor_prompt_ownership(self):
        b = Backend()
        context = Context()
        b.engine.context = context
        with self.assertRaisesRegex(ns['EngineError'], 'cancelled'):
            list(b._stream(opts(1), context=context))
        self.assertEqual(clock.now, 4.0)
        self.assertTrue(b.gen_lock.acquire(blocking=False))
        b.gen_lock.release()
        answers = []
        for owner in (2, 3):
            replies = list(b._stream(opts(owner)))
            answers.append(b''.join(r.message for r in replies).decode())
        self.assertEqual(answers, ['answer-2', 'answer-3'])
        self.assertEqual([cmd.split()[0] for _, cmd in b.engine.commands], ['GEN', 'STOP', 'GEN', 'STOP', 'GEN', 'STOP'])
        self.assertEqual(b.engine.lines.events, [], 'no generation may remain pending')
        print('Successor answers remain aligned:', answers)
        print('COMMAND TIMES:', [(at, cmd.split()[0]) for at, cmd in b.engine.commands])


if __name__ == '__main__': unittest.main(verbosity=2)