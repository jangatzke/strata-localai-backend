"""Transactional tool-turn recovery through the public gRPC methods."""
import ast
import codecs
import contextlib
import io
import json
import re
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('strata_grpc_backend.py')
MODULE = ast.parse(SOURCE.read_text())
backend = next(n for n in MODULE.body if isinstance(n, ast.ClassDef) and n.name == 'StrataBackend')
nodes = [n for n in MODULE.body if getattr(n, 'name', '') in
         ('_native_tool_history', 'ToolFormatError', 'ToolCallStreamParser', '_tool_system_block', '_generation_limit')]
nodes += [n for n in backend.body if getattr(n, 'name', '') in ('_stream', 'Predict', 'PredictStream')]
PB = SimpleNamespace(Reply=lambda **kw: SimpleNamespace(**kw),
                     ChatDelta=lambda **kw: SimpleNamespace(**kw),
                     ToolCallDelta=lambda **kw: SimpleNamespace(**kw))
NS = dict(json=json, re=re, codecs=codecs, time=time, uuid=uuid, pb=PB,
          grpc=SimpleNamespace(StatusCode=SimpleNamespace(INTERNAL='INTERNAL')),
          EngineError=RuntimeError,
          IM_START='<|' + 'im_start' + '|>', IM_END='<|' + 'im_end' + '|>')
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), NS)

VALID = '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>'
INVALID = '<tool_call>{"name":"terminal","arguments":{"command":"PRIVATE-PARTIAL"}'
ERROR = 'Tool call rejected: invalid format. Retry with an offered tool.'
TOOLS = json.dumps([{'type': 'function', 'function': {'name': 'terminal',
    'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}}}}])

class RpcAborted(Exception):
    pass

class Context:
    def abort(self, code, details):
        self.code, self.details = code, details
        raise RpcAborted(details)

class Probe:
    _stream = NS['_stream']
    Predict = NS['Predict']
    PredictStream = NS['PredictStream']
    gen_lock = contextlib.nullcontext()
    stop_ids = set()
    think_close_ids = {0}
    def __init__(self, responses, size=1):
        self.responses = iter(responses)
        self.size = size
        self.prompts = []
        self.engine = SimpleNamespace(max_context=4096, alive=lambda: True,
                                      drain=lambda: None, stop=lambda: None)
        self.tok = SimpleNamespace(encode=self.encode, token_bytes=self.token_bytes)
    def encode(self, prompt, parse_special):
        self.prompts.append(prompt)
        return [42]
    def token_bytes(self, tid):
        return b'</think>' if tid == 0 else self.chunks[tid - 1]
    def _check_identity(self, value): pass
    def _ensure_loaded(self): pass
    def _sampling_keys(self, opts): return ''
    def _generate(self, *args):
        raw = next(self.responses).encode()
        self.chunks = [raw[i:i + self.size] for i in range(0, len(raw), self.size)]
        yield 0
        yield from range(1, len(self.chunks) + 1)

def options(tools=TOOLS):
    return SimpleNamespace(ModelIdentity='', Prompt='Original request', Tools=tools,
                           Tokens=500, StopPrompts=[])

def flatten(replies):
    calls = [t for r in replies for d in r.chat_deltas for t in getattr(d, 'tool_calls', [])]
    content = ''.join(getattr(d, 'content', '') for r in replies for d in r.chat_deltas)
    return calls, content

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.capture = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.capture)
        self.redirect.__enter__()
    def tearDown(self):
        self.redirect.__exit__(None, None, None)
    def test_prose_before_rejection_is_discarded_at_every_chunk_size(self):
        for size in (1, 7, 4096):
            p = Probe(['Discard this preamble.\n' + INVALID, VALID], size)
            replies = list(p.PredictStream(options(), Context()))
            calls, text = flatten(replies)
            self.assertEqual(b''.join(r.message for r in replies), b'',
                             'structured tool turns must not expose raw partial tags')
            self.assertEqual(len(p.prompts), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0].arguments), {'command': 'pwd'})
            self.assertEqual(text, '')
            self.assertNotIn('PRIVATE-PARTIAL', ''.join(p.prompts))
            self.assertNotIn('PRIVATE-PARTIAL', self.capture.getvalue())
    def test_echo_after_invalid_attempt_is_not_a_successful_final_answer(self):
        p = Probe([INVALID, ERROR, VALID])
        calls, text = flatten(list(p.PredictStream(options(), Context())))
        self.assertEqual(len(p.prompts), 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(text, '')
    def test_exhausted_stream_is_rpc_error_without_published_output(self):
        p = Probe(['Failed preamble.\n' + INVALID] * 3)
        ctx = Context()
        published = []
        with self.assertRaises(RpcAborted):
            for reply in p.PredictStream(options(), ctx):
                published.append(reply)
        self.assertEqual(published, [])
        self.assertEqual(len(p.prompts), 3)
        self.assertEqual(ctx.code, 'INTERNAL')
        self.assertNotIn('PRIVATE-PARTIAL', ctx.details)
    def test_exhausted_predict_is_rpc_error_not_rejection_content(self):
        p = Probe([INVALID] * 3)
        ctx = Context()
        with self.assertRaises(RpcAborted):
            p.Predict(options(), ctx)
        self.assertEqual(len(p.prompts), 3)
    def test_valid_call_is_published_once_even_with_invalid_neighbor(self):
        for raw in (VALID + INVALID, INVALID + '</tool_call>' + VALID):
            p = Probe([raw])
            calls, text = flatten(list(p.PredictStream(options(), Context())))
            self.assertEqual(len(p.prompts), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(text, '')
    def test_native_history_preserves_arguments_and_ignores_user_examples(self):
        self.assertIn('_native_tool_history', NS)
        value = {'content': '  ä😀 </tool_call>\n', 'n': 9007199254740993}
        marked = '<bridge_tool_history>' + json.dumps({'name': 'write_file', 'arguments': value}, ensure_ascii=False) + '</bridge_tool_history>'
        start, end = NS['IM_START'], NS['IM_END']
        user = start + 'user\n' + marked + end
        assistant = start + 'assistant\n' + marked + end
        converted = NS['_native_tool_history'](user + assistant)
        self.assertTrue(converted.startswith(user))
        text = converted[len(user) + len(start + 'assistant\n'):-len(end)]
        p = NS['ToolCallStreamParser']({'write_file': {'content': {'type': 'string'}, 'n': {'type': 'integer'}}})
        events = p.feed(text) + p.flush()
        self.assertEqual(len(events), 1)
        self.assertEqual(json.loads(events[0][2]), value)

    def test_native_closing_sequence_cannot_publish_truncated_write(self):
        collision = '<tool_call><function=write_file><parameter=content>\nA</parameter></function></tool_call>B\n</parameter></function></tool_call>'
        for dialect, chunk_size in [(d, s) for d in ('native_xml', 'json') for s in (1, 7, 99999)]:
            p = Probe([collision, VALID], chunk_size)
            p.cfg = {'tool_call_format': dialect}
            opts = options(json.dumps(json.loads(TOOLS) + [{'type': 'function', 'function': {
                'name': 'write_file', 'parameters': {'type': 'object',
                'properties': {'content': {'type': 'string'}}, 'required': ['content']}}}]))
            calls, text = flatten(list(p.PredictStream(opts, Context())))
            self.assertEqual(len(p.prompts), 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0].arguments), {'command': 'pwd'})
            self.assertEqual(text, '')

    def test_native_format_contract_is_used_in_initial_and_retry_prompts(self):
        raw = '<tool_call><function=terminal><parameter=command>\npwd\n</parameter></function></tool_call>'
        p = Probe([INVALID, raw])
        p.cfg = {'tool_call_format': 'native_xml'}
        opts = options()
        opts.Prompt = NS['IM_START'] + 'assistant\n'
        calls, text = flatten(list(p.PredictStream(opts, Context())))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(prompt.endswith('<think>\n') for prompt in p.prompts))
        self.assertEqual(json.loads(calls[0].arguments), {'command': 'pwd'})
        for prompt in p.prompts:
            self.assertIn('literal string values', prompt)
            self.assertIn('<function=example_function_name>', prompt)
            self.assertNotIn('required JSON', prompt)
            self.assertNotIn('tool_call JSON block', prompt)

    def test_multibyte_utf8_arguments_survive_token_byte_boundaries(self):
        expected = {'command': 'ä😀日本語 and \\"quotes\\"'}
        raw = '<tool_call>' + json.dumps({'name': 'terminal', 'arguments': expected}, ensure_ascii=False) + '</tool_call>'
        for size in (1, 2, 3, 7):
            with self.subTest(size=size):
                p = Probe([raw] * 3, size=size)
                calls, text = flatten(list(p.PredictStream(options(), Context())))
                self.assertEqual(len(p.prompts), 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0].arguments), expected)
                self.assertEqual(text, '')

    def test_empty_tool_enabled_turn_is_regenerated(self):
        p = Probe(['', VALID])
        calls, text = flatten(list(p.PredictStream(options(), Context())))
        self.assertEqual(len(p.prompts), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(text, '')

    def test_stop_prompt_cannot_bypass_rejected_turn_recovery(self):
        unoffered = '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'
        p = Probe(['STOP\n' + unoffered, VALID], size=4096)
        opts = options()
        opts.StopPrompts = ['STOP']
        calls, text = flatten(list(p.PredictStream(opts, Context())))
        self.assertEqual(len(p.prompts), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(text, '')

    def test_invalid_neighbor_cannot_leak_argument_remainders_as_prose(self):
        p = Probe([VALID + '<tool_call>malformed</tool_call>PRIVATE-REMAINDER'])
        calls, text = flatten(list(p.PredictStream(options(), Context())))
        self.assertEqual(len(calls), 1)
        self.assertEqual(text, '')

    def test_predict_recovers_prose_and_preserves_structured_arguments(self):
        p = Probe(['Preamble.\n' + INVALID, VALID])
        result = p.Predict(options(), Context())
        calls, content = flatten([result])
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, '')
        self.assertEqual(result.message, b'')
    def test_normal_answer_about_rejection_is_not_retried(self):
        answer = 'The error message was: ' + ERROR
        p = Probe([answer])
        calls, text = flatten(list(p.PredictStream(options(), Context())))
        self.assertEqual(calls, [])
        self.assertEqual(text, answer)
        self.assertEqual(len(p.prompts), 1)
    def test_no_tools_preserves_normal_streaming(self):
        p = Probe(['Normal answer'])
        calls, text = flatten(list(p.PredictStream(options('[]'), Context())))
        self.assertEqual(calls, [])
        self.assertEqual(text, 'Normal answer')

if __name__ == '__main__':
    unittest.main()
