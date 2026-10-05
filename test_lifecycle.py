"""Real-lock bridge lifecycle regressions; fake engine, no GPU or tool execution."""
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import unittest
import grpc
import backend_pb2 as pb
import backend_pb2_grpc as rpc
import strata_grpc_backend as bridge


class Tokenizer:
    def encode(self, text, parse_special=True):
        return [1]

    def token_bytes(self, token):
        return b'</think>' if token == 5 else b'answer'


class FakeEngine:
    parallel = 1
    max_context = 100
    can_stop = True
    ended = False

    def __init__(self):
        self.lines = queue.Queue()
        self.loaded = True
        self.started = 0
        self.commands = []
        self.stops = 0
        self.drains = 0

    def alive(self):
        return self.loaded

    def start(self):
        self.started += 1
        self.loaded = True

    def free(self):
        self.loaded = False

    def stop(self):
        self.stops += 1

    def drain(self):
        self.drains += 1
        while not self.lines.empty():
            self.lines.get_nowait()

    def _send(self, line):
        self.commands.append(line)
        self.lines.put('T 5')
        self.lines.put('T 6')
        self.lines.put('DONE')


def backend():
    b = bridge.StrataBackend({'model_name': 'test'}, Tokenizer())
    b.engine = FakeEngine()
    b.stop_ids = set()
    b.think_close_ids = {5}
    b.think_open_ids = set()
    return b


class AbortContext:
    def abort(self, code, detail):
        raise RuntimeError(detail)


class LifecycleTests(unittest.TestCase):
    def test_cancelled_vision_admission_removes_request_embeddings(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        b = backend()
        class Context(AbortContext):
            active = True
            def is_active(self): return self.active
        context = Context()
        with tempfile.TemporaryDirectory() as directory:
            embeddings = Path(directory) / 'request.sve'
            embeddings.write_bytes(b'fake vision embeddings')
            def prepare(*args):
                context.active = False
                return [1], embeddings
            with patch.object(bridge.StrataBackend, '_prepare_images', side_effect=prepare):
                with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                    b.Predict(pb.PredictOptions(Prompt='image', Tokens=10, Images=['fake']), context)
            self.assertFalse(embeddings.exists(), 'cancelled admission leaked request embeddings')
        self.assertEqual(b.engine.commands, [])
        self.assertEqual(b.engine.stops, 0)

    def test_grpc_shutdown_cancels_owner_and_rejects_queued_reload(self):
        b = backend()
        sent = threading.Event()
        queued_encoded = threading.Event()
        encoded = []
        def encode(*args, **kwargs):
            encoded.append(1)
            if len(encoded) == 2:
                queued_encoded.set()
            return [1]
        b.tok.encode = encode
        def send(line):
            b.engine.commands.append(line)
            sent.set()
        b.engine._send = send
        server = grpc.server(ThreadPoolExecutor(max_workers=3))
        rpc.add_BackendServicer_to_server(b, server)
        port = server.add_insecure_port('127.0.0.1:0')
        server.start()
        try:
            with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                stub = rpc.BackendStub(channel)
                opts = pb.PredictOptions(Prompt='shutdown', Tokens=10)
                owner = stub.Predict.future(opts, timeout=5)
                self.assertTrue(sent.wait(2))
                queued = stub.Predict.future(opts, timeout=5)
                self.assertTrue(queued_encoded.wait(2))
                b.stopping = True
                server.stop(0)
                self.assertTrue(b.free(timeout_s=2.5))
                for call in (owner, queued):
                    with self.assertRaises(grpc.RpcError):
                        call.result(2)
        finally:
            b.engine.lines.put('DONE')
            server.stop(0).wait(3)
        self.assertEqual(len(b.engine.commands), 1)
        self.assertFalse(b.engine.loaded)
        self.assertEqual(b.engine.started, 0)
        self.assertFalse(b.LoadModel(pb.ModelOptions(Model='test'), AbortContext()).success)

    def test_tool_exhaustion_cannot_drain_newly_admitted_request(self):
        import json
        from unittest.mock import patch
        b = backend()
        admitted = threading.Event()
        aborted = threading.Event()
        third_attempt = threading.Event()
        class Semaphore(threading.Semaphore):
            releases = 0
            def __exit__(self, *args):
                self.release()
                self.releases += 1
                if self.releases == 3:
                    third_attempt.set()
                    admitted.wait(2)
        b.gen_lock = Semaphore(1)
        original_bytes = b.tok.token_bytes
        b.tok.token_bytes = lambda tid: b'<tool_call>{' if tid == 7 else original_bytes(tid)
        def send(line):
            b.engine.commands.append(line)
            b.engine.lines.put('T 5')
            if len(b.engine.commands) <= 3:
                b.engine.lines.put('T 7')
                b.engine.lines.put('DONE')
            else:
                b.engine.lines.put('T 6')
                b.engine.lines.put('DONE')
                admitted.set()
                aborted.wait(2)
        b.engine._send = send
        class Context(AbortContext):
            def abort(self, code, detail):
                aborted.set()
                super().abort(code, detail)
        tools = json.dumps([{'function': {'name': 'tool', 'parameters': {'type': 'object'}}}])
        a_opts = pb.PredictOptions(Prompt='bad tool', Tools=tools, Tokens=10)
        b_opts = pb.PredictOptions(Prompt='peer', Tokens=10)
        with patch.object(bridge, 'GEN_TIMEOUT_S', .2), ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(b.Predict, a_opts, Context())
            self.assertTrue(third_attempt.wait(2))
            peer = pool.submit(b.Predict, b_opts, AbortContext())
            with self.assertRaisesRegex(RuntimeError, 'tool turn'):
                a.result(3)
            answer = peer.result(3)
        self.assertEqual(answer.message, b'answer')
        self.assertEqual(b.engine.drains, 0, 'completed malformed turn must not touch peer queue')

    def test_shutdown_during_vision_load_does_not_start_text_engine(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        b = backend()
        b.engine.loaded = False
        b.engine._env = lambda: {}
        b.vision_cfg = {'enabled': True}
        vision = SimpleNamespace(alive=lambda: False, close=lambda: None,
                                 start=lambda: setattr(b, 'stopping', True))
        with patch.object(bridge, 'Vision', return_value=vision):
            with self.assertRaisesRegex(bridge.EngineError, 'stopping'):
                b.load()
        self.assertEqual(b.engine.started, 0)
        self.assertIsNone(b.vision)

    def test_predict_closes_stream_when_reply_aggregation_fails(self):
        from types import SimpleNamespace
        b = backend()
        closed = threading.Event()
        class BadMessage:
            def decode(self, *args):
                raise ValueError('aggregation failed')
        def stream(*args, **kwargs):
            try:
                yield SimpleNamespace(message=BadMessage())
            finally:
                closed.set()
        active_stream = stream()
        b._stream = lambda *args, **kwargs: active_stream
        caught = None
        try:
            b.Predict(pb.PredictOptions(), AbortContext())
        except RuntimeError as exc:
            caught = exc
        self.assertIsNotNone(caught)
        self.assertTrue(closed.is_set(), 'failed aggregation left generation suspended')

    def test_grpc_delayed_batch_admission_cancel_and_shutdown(self):
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown):
                b = backend()
                b.engine = bridge.Engine({'parallel': 2})
                b.engine.proc = type('Proc', (), {'poll': lambda self: None})()
                b.engine.ended = False
                b.engine.can_stop = True
                b.gen_lock = threading.Semaphore(2)
                sent = threading.Event()
                stopped = threading.Event()
                drained = threading.Event()
                slots = []
                commands = []
                def route(text):
                    b.engine._route_line(text + '\n')
                def send(line):
                    commands.append(line)
                    if line.startswith('BGEN '):
                        slots.append(int(line.split()[1]))
                        sent.set()
                    elif line.startswith('BSTOP '):
                        stopped.set()
                b.engine._send = send
                original_drain = b.engine.drain
                def drain(*args, **kwargs):
                    original_drain(*args, **kwargs)
                    drained.set()
                b.engine.drain = drain
                b.engine.free = lambda: setattr(b.engine, 'ended', True)
                server = grpc.server(ThreadPoolExecutor(max_workers=3))
                rpc.add_BackendServicer_to_server(b, server)
                port = server.add_insecure_port('127.0.0.1:0')
                server.start()
                try:
                    with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                        call = rpc.BackendStub(channel).Predict.future(pb.PredictOptions(Prompt='pending', Tokens=10), timeout=5)
                        self.assertTrue(sent.wait(2))
                        if shutdown:
                            b.stopping = True
                            server.stop(0)
                        else:
                            call.cancel()
                        observed_stop = stopped.wait(1)
                        # Unload cannot take the unresolved request's permit.
                        early_unload = b.free(timeout_s=.05)
                        still_owned = b.engine.available_slots()
                        route('T 6')
                        route('DONE')
                        route(f'BADM {slots[0]} 1')
                        route(f'BT {slots[0]} 6')
                        route(f'BDONE {slots[0]} 1 cancelled 1')
                        self.assertTrue(drained.wait(2))
                        self.assertTrue(b.free(timeout_s=2))
                        self.assertTrue(observed_stop, 'delayed BADM ignored RPC cancellation/shutdown')
                        self.assertFalse(early_unload)
                        self.assertEqual(still_owned, 1)
                        self.assertEqual(b.engine.available_slots(), 2)
                        self.assertEqual([c for c in commands if c.startswith('BSTOP')], [f'BSTOP {slots[0]}'])
                finally:
                    for slot in slots:
                        route(f'BDONE {slot} 0 stop 1')
                    server.stop(0).wait(3)

    def test_grpc_batch_cancel_stops_only_cancelled_slot_while_peer_finishes(self):
        b = backend()
        b.engine = bridge.Engine({'parallel': 2})
        b.engine.proc = type('Proc', (), {'poll': lambda self: None})()
        b.engine.ended = False
        b.engine.can_stop = True
        b.gen_lock = threading.Semaphore(2)
        first = threading.Event()
        second = threading.Event()
        cancelled = threading.Event()
        slots = []
        commands = []
        def route(text):
            b.engine._route_line(text + '\n')
        def send(line):
            commands.append(line)
            parts = line.split()
            if parts[0] == 'BGEN':
                slot = int(parts[1])
                slots.append(slot)
                route(f'BADM {slot} 1')
                (first if len(slots) == 1 else second).set()
            elif parts[0] == 'BSTOP':
                slot = int(parts[1])
                route(f'BDONE {slot} 0 cancelled 1')
                cancelled.set()
        b.engine._send = send
        server = grpc.server(ThreadPoolExecutor(max_workers=3))
        rpc.add_BackendServicer_to_server(b, server)
        port = server.add_insecure_port('127.0.0.1:0')
        server.start()
        cancelled_cleanly = False
        try:
            with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                stub = rpc.BackendStub(channel)
                opts = pb.PredictOptions(Prompt='batch', Tokens=10)
                a = stub.Predict.future(opts, timeout=8)
                self.assertTrue(first.wait(2))
                peer = stub.Predict.future(opts, timeout=8)
                self.assertTrue(second.wait(2))
                a.cancel()
                cancelled_cleanly = cancelled.wait(2)
                route(f'BT {slots[1]} 5')
                route(f'BT {slots[1]} 6')
                route(f'BDONE {slots[1]} 2 stop 1')
                answer = peer.result(3)
        finally:
            for slot in slots:
                route(f'BDONE {slot} 0 stop 1')
            server.stop(0).wait(3)
        self.assertTrue(cancelled_cleanly, 'cancelled batch slot stayed active')
        self.assertEqual(answer.message, b'answer')
        self.assertEqual([c for c in commands if c.startswith('BSTOP')], [f'BSTOP {slots[0]}'])
        self.assertEqual(b.engine.available_slots(), 2)

    def test_grpc_disconnect_cancels_buffered_tool_turn_before_retry(self):
        import json
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                b = backend()
                sent = threading.Event()
                cleaned = threading.Event()
                def send(line):
                    b.engine.commands.append(line)
                    sent.set()
                    # No DONE: cancellation must work without another token.
                b.engine._send = send
                original_drain = b.engine.drain
                def drain():
                    original_drain()
                    cleaned.set()
                b.engine.drain = drain
                server = grpc.server(ThreadPoolExecutor(max_workers=3))
                rpc.add_BackendServicer_to_server(b, server)
                port = server.add_insecure_port('127.0.0.1:0')
                server.start()
                cancelled_cleanly = False
                try:
                    with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
                        stub = rpc.BackendStub(channel)
                        tools = json.dumps([{'type': 'function', 'function': {'name': 'tool', 'parameters': {'type': 'object'}}}])
                        opts = pb.PredictOptions(Prompt='cancel', Tokens=10, Tools=tools)
                        call = stub.PredictStream(opts, timeout=5) if streaming else stub.Predict.future(opts, timeout=5)
                        self.assertTrue(sent.wait(2))
                        self.assertTrue(call.cancel())
                        cancelled_cleanly = cleaned.wait(2)
                finally:
                    b.engine.lines.put('DONE')
                    server.stop(0).wait(3)
                self.assertTrue(cancelled_cleanly, 'disconnected buffered turn kept generating')
                self.assertEqual(len(b.engine.commands), 1, 'disconnect must not regenerate')
                self.assertTrue(b.gen_lock.acquire(timeout=2))
                b.gen_lock.release()

    def test_signal_closes_grpc_admission_before_bounded_unload(self):
        from unittest.mock import patch
        order = []
        handlers = {}
        b = backend()
        def free(timeout_s=None):
            order.append(('free', b.stopping, timeout_s))
            return True
        b.free = free
        class Server:
            def add_insecure_port(self, address): return 1
            def start(self): pass
            def stop(self, grace):
                order.append(('stop', b.stopping, grace))
            def wait_for_termination(self):
                handlers[bridge.signal.SIGTERM]()
        with patch.object(bridge.argparse.ArgumentParser, 'parse_args', return_value=type('Args', (), {'config': 'unused', 'host': '127.0.0.1', 'port': 1})()), \
             patch.object(bridge.Path, 'read_text', return_value='{"model_name": "test"}'), \
             patch.object(bridge, 'build_tokenizer', return_value=Tokenizer()), \
             patch.object(bridge, 'StrataBackend', return_value=b), \
             patch.object(bridge.grpc, 'server', return_value=Server()), \
             patch.object(bridge.pb_grpc, 'add_BackendServicer_to_server'), \
             patch.object(bridge.signal, 'signal', side_effect=lambda sig, fn: handlers.update({sig: fn})), \
             patch.object(bridge.sys, 'exit'):
            bridge.main()
        self.assertEqual(order[0], ('stop', True, 0))
        self.assertEqual(order[1][0:2], ('free', True))
        self.assertIsInstance(order[1][2], (float, int))

    def test_shutdown_rejects_load_and_queued_generation_without_reloading(self):
        b = backend()
        b.gen_lock.acquire()
        opts = pb.PredictOptions(Prompt='queued', Tokens=10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            queued = pool.submit(lambda: list(b.PredictStream(opts, AbortContext())))
            b.stopping = True
            b.engine.loaded = False
            load_error = None
            try:
                b.load()
            except bridge.EngineError as exc:
                load_error = exc
            b.gen_lock.release()
            request_error = None
            try:
                queued.result(2)
            except RuntimeError as exc:
                request_error = exc
        self.assertIsNotNone(load_error, 'shutdown must reject LoadModel')
        self.assertIsNotNone(request_error, 'queued generation must be rejected')
        self.assertEqual(b.engine.started, 0, 'shutdown must never reload the engine')
        self.assertEqual(b.engine.commands, [])

    def test_shutdown_free_has_bounded_permit_wait_without_stopping_other_owner(self):
        b = backend()
        b.gen_lock.acquire()
        b.stopping = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            unloading = pool.submit(lambda: b.free(timeout_s=.05))
            try:
                result = unloading.result(1)
            finally:
                b.gen_lock.release()
        self.assertFalse(result)
        self.assertTrue(b.engine.loaded)
        self.assertEqual(b.engine.stops, 0)
        self.assertEqual(b.engine.drains, 0)
        self.assertTrue(b.free(timeout_s=.05))
        self.assertFalse(b.engine.loaded)

    def test_concurrent_free_does_not_split_parallel_permits(self):
        b = backend()
        b.engine.parallel = 2
        rendezvous = threading.Barrier(2)
        class SplitSemaphore(threading.Semaphore):
            def __init__(self):
                super().__init__(2)
                self.local = threading.local()
            def acquire(self, *args, **kwargs):
                acquired = super().acquire(*args, **kwargs)
                if acquired and not getattr(self.local, 'seen', False):
                    self.local.seen = True
                    try:
                        rendezvous.wait(.15)
                    except threading.BrokenBarrierError:
                        pass
                return acquired
        b.gen_lock = SplitSemaphore()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(b.free, .5)
            peer = pool.submit(b.free, .5)
            results = [a.result(2), peer.result(2)]
        self.assertEqual(results, [True, True], 'Free callers split permits and deadlocked')
        self.assertFalse(b.engine.loaded)
        self.assertTrue(b.gen_lock.acquire(timeout=.1))
        self.assertTrue(b.gen_lock.acquire(timeout=.1))
        b.gen_lock.release()
        b.gen_lock.release()

    def test_ordinary_free_allows_later_load(self):
        b = backend()
        b.free()
        self.assertEqual(b.load(), 'loaded')
        self.assertTrue(b.engine.loaded)

    def test_failed_request_cleanup_cannot_steal_next_generation(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                b = backend()
                cleaning = threading.Event()
                release_cleanup = threading.Event()
                admitted_b = threading.Event()
                owned = []
                original_send = b.engine._send
                def send(line):
                    if not b.engine.commands:
                        b.engine.commands.append(line)
                        b.engine.lines.put('ERR broken request A')
                    else:
                        admitted_b.set()
                        original_send(line)
                b.engine._send = send
                original_drain = b.engine.drain
                def drain():
                    acquired = b.gen_lock.acquire(blocking=False)
                    owned.append(not acquired)
                    if acquired:
                        b.gen_lock.release()
                    cleaning.set()
                    release_cleanup.wait(2)
                    original_drain()
                b.engine.drain = drain
                opts = pb.PredictOptions(Prompt='test', Tokens=10)
                def call():
                    if streaming:
                        return list(b.PredictStream(opts, AbortContext()))
                    return b.Predict(opts, AbortContext())
                with ThreadPoolExecutor(max_workers=2) as pool:
                    a = pool.submit(call)
                    self.assertTrue(cleaning.wait(2))
                    second = pool.submit(call)
                    early_admission = admitted_b.wait(.1)
                    release_cleanup.set()
                    with self.assertRaisesRegex(RuntimeError, 'broken request A'):
                        a.result(3)
                    # Never wait on B after a broken ownership assertion: old code
                    # loses DONE and would otherwise block for an hour.
                    if early_admission:
                        b.engine.lines.put('DONE')
                    result = second.result(3)
                self.assertEqual(owned, [True], 'cleanup must own the generation permit')
                self.assertFalse(early_admission, 'B admitted before A cleanup finished')
                replies = result if streaming else [result]
                self.assertTrue(any(r.message == b'answer' for r in replies))


if __name__ == '__main__':
    unittest.main()
