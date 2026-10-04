"""Real gRPC fixture: exercise RPC delivery, never run generated tools."""
from concurrent.futures import ThreadPoolExecutor
import json
import grpc
import backend_pb2 as pb
import backend_pb2_grpc as rpc
import test_recovery as fixture

fixture.NS.update(pb=pb, grpc=grpc)
class Backend(fixture.Probe, rpc.BackendServicer):
    pass

results = []
collision = '<tool_call><function=terminal><parameter=command>\nA</parameter></function></tool_call>B\n</parameter></function></tool_call>'
for name, streaming, responses, should_error in (
    ('stream_recovery', True, [fixture.INVALID, fixture.ERROR, fixture.VALID], False),
    ('predict_recovery', False, ['Preamble.\n' + fixture.INVALID, fixture.VALID], False),
    ('stream_empty_recovery', True, ['', fixture.VALID], False),
    ('predict_empty_exhaustion', False, [''] * 3, True),
    ('stream_exhaustion', True, [fixture.INVALID] * 3, True),
    ('predict_exhaustion', False, [fixture.INVALID] * 3, True),
    ('native_stream_collision_recovery', True, [collision, fixture.VALID], False),
    ('native_predict_collision_recovery', False, [collision, fixture.VALID], False),
    ('native_stream_collision_exhaustion', True, [collision] * 3, True),
    ('native_predict_collision_exhaustion', False, [collision] * 3, True),
):
    backend = Backend(responses, size=7)
    if name.startswith('native_'):
        backend.cfg = {'tool_call_format': 'native_xml'}
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    rpc.add_BackendServicer_to_server(backend, server)
    port = server.add_insecure_port('127.0.0.1:0')
    server.start()
    replies = []
    status = None
    try:
        with grpc.insecure_channel(f'127.0.0.1:{port}') as channel:
            stub = rpc.BackendStub(channel)
            opts = pb.PredictOptions(Prompt='Synthetic regression', Tools=fixture.TOOLS, Tokens=500)
            try:
                if streaming:
                    replies = list(stub.PredictStream(opts, timeout=10))
                else:
                    replies = [stub.Predict(opts, timeout=10)]
            except grpc.RpcError as exc:
                status = exc.code()
            if should_error:
                assert status == grpc.StatusCode.INTERNAL and not replies, (name, status)
            else:
                assert status is None, (name, status)
                calls = [t for reply in replies for d in reply.chat_deltas for t in d.tool_calls]
                assert len(calls) == 1 and calls[0].name == 'terminal', name
                assert json.loads(calls[0].arguments) == {'command': 'pwd'}, name
                assert not any(d.content for reply in replies for d in reply.chat_deltas), name
            assert len(backend.prompts) <= 3
            record = {'fixture': name, 'passed': True, 'attempts': len(backend.prompts),
                      'grpc_status': status.name if status else 'OK'}
            results.append(record)
            print(json.dumps(record), flush=True)
    finally:
        server.stop(0).wait()
assert len(results) == 10
print('PASS: ten real gRPC fixtures; no generated tool was executed.')
