"""Regression tests for the Strata bridge's token-stream tool parser."""
import ast
import codecs
import contextlib
import json
import re
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).with_name('strata_grpc_backend.py')
module = ast.parse(SOURCE.read_text())
parser_node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'ToolCallStreamParser')
namespace = {'json': json, 're': re}
exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
Parser = namespace['ToolCallStreamParser']


def events(text, size=3, specs=None):
    parser = Parser(specs or {})
    result = []
    for i in range(0, len(text), size):
        result.extend(parser.feed(text[i:i + size]))
    result.extend(parser.flush())
    return result


class OutputBudgetTests(unittest.TestCase):
    def test_unset_client_limit_uses_remaining_engine_context(self):
        node = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_generation_limit')
        scope = {'EngineError': RuntimeError}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
        limit = scope['_generation_limit']
        self.assertEqual(limit(0, 50000, 128000), 128000 - 50000 - 8)
        self.assertEqual(limit(-1, 50000, 128000), 128000 - 50000 - 8)
        self.assertEqual(limit(4096, 50000, 128000), 4096)
        with self.assertRaisesRegex(RuntimeError, 'no room'):
            limit(0, 127995, 128000)


class PredictReplyTests(unittest.TestCase):
    def test_predict_preserves_structured_deltas_without_raw_tool_fallback(self):
        methods = next(n for n in module.body if isinstance(n, ast.ClassDef)
                       and n.name == 'StrataBackend').body
        predict = next(n for n in methods if isinstance(n, ast.FunctionDef) and n.name == 'Predict')
        pb = SimpleNamespace(Reply=lambda **kw: SimpleNamespace(**kw))
        ns = {'pb': pb}
        exec(compile(ast.Module(body=[predict], type_ignores=[]), str(SOURCE), 'exec'), ns)
        for delta in (SimpleNamespace(content='', tool_calls=[SimpleNamespace(name='write_file', arguments='{}')]),
                      SimpleNamespace(content='Tool call rejected: invalid format. Retry with an offered tool.')):
            reply = SimpleNamespace(message=b'<tool_call>DO-NOT-PARSE</tool_call>',
                                    tokens=1, prompt_tokens=5, chat_deltas=[delta])
            probe = SimpleNamespace(_stream=lambda *a, **kw: iter([reply]))
            result = ns['Predict'](probe, None, None)
            self.assertEqual(getattr(result, 'chat_deltas', []), [delta])
            self.assertNotIn(b'DO-NOT-PARSE', result.message)


class ToolCallStreamParserTests(unittest.TestCase):
    def test_json_string_delimiters_roundtrip_without_rejection_or_leak(self):
        values = ['```json\n{"ok": true}\n```',
                  'Literal </tool_call> inside a string',
                  '```xml\n<tool_call>example</tool_call>\n```',
                  'Escaped quote " and slash \\ followed by </tool_call>',
                  'Two backslashes \\\\ then "</tool_call>" and Unicode ä']
        specs = {'write_file': {'content': {'type': 'string'}}}
        for value in values:
            raw = '<tool_call>' + json.dumps({'name': 'write_file',
                  'arguments': {'content': value}}) + '</tool_call>'
            for size in (1, 2, 7, 31, len(raw)):
                with self.subTest(value=value, size=size):
                    result = events(raw, size=size, specs=specs)
                    self.assertEqual([e[0] for e in result], ['tool'])
                    self.assertEqual(json.loads(result[0][2]), {'content': value})

    def test_delimiter_collision_at_every_single_chunk_boundary(self):
        args = {'content': 'Quote " then \\ and </tool_call> plus ```\ntext',
                'nested': {'items': ['</tool_call>', 42]}}
        raw = '<tool_call>' + json.dumps({'name': 'write_file', 'arguments': args}) + '</tool_call>'
        for split in range(len(raw) + 1):
            parser = Parser({'write_file': {}})
            result = parser.feed(raw[:split]) + parser.feed(raw[split:]) + parser.flush()
            self.assertEqual([e[0] for e in result], ['tool'])
            self.assertEqual(json.loads(result[0][2]), args)

    def test_collision_state_resets_between_tool_blocks(self):
        args = {'content': 'Literal </tool_call> and \\"'}
        raw = '<tool_call>' + json.dumps({'name': 'write_file', 'arguments': args}) + '</tool_call>'
        for size in (1, 7, len(raw) * 2):
            result = events(raw * 2, size=size, specs={'write_file': {}})
            self.assertEqual([e[0] for e in result], ['tool', 'tool'])
            self.assertTrue(all(json.loads(e[2]) == args for e in result))

    def test_collision_does_not_repair_incomplete_or_unoffered_write_calls(self):
        args = {'content': 'Private value </tool_call> inside JSON'}
        valid = '<tool_call>' + json.dumps({'name': 'write_file', 'arguments': args}) + '</tool_call>'
        cases = [valid[:-len('</tool_call>')],
                 valid.replace('write_file', 'unoffered_write'),
                 valid.replace('inside JSON"', 'inside JSON'),
                 valid.replace('"arguments":', '"arguments"')]
        for raw in cases:
            for size in (1, 7, len(raw)):
                result = events(raw, size=size, specs={'write_file': {}})
                self.assertEqual([e[0] for e in result], ['invalid_tool'])
                self.assertNotIn('Private value', str(result))

    def test_stream_retries_rejected_only_tool_turn_without_executing_partial_calls(self):
        methods = next(n for n in module.body if isinstance(n, ast.ClassDef) and
                       n.name == 'StrataBackend').body
        stream = next(n for n in methods if isinstance(n, ast.FunctionDef) and n.name == '_stream')
        tool_block = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_tool_system_block')
        budget_fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                         and n.name == '_generation_limit')
        pb = SimpleNamespace(Reply=lambda **kw: SimpleNamespace(**kw),
                             ChatDelta=lambda **kw: SimpleNamespace(**kw),
                             ToolCallDelta=lambda **kw: SimpleNamespace(**kw))
        ns = dict(json=json, re=re, codecs=codecs, time=time, uuid=uuid,
                  pb=pb, ToolCallStreamParser=Parser,
                  IM_START='<|' + 'im_start' + '|>', IM_END='<|' + 'im_end' + '|>')
        exec(compile(ast.Module(body=[tool_block, budget_fn, stream], type_ignores=[]),
                     str(SOURCE), 'exec'), ns)

        class Probe:
            gen_lock = contextlib.nullcontext()
            stop_ids = set()
            think_close_ids = set()
            engine = SimpleNamespace(max_context=1024)
            def __init__(self, responses):
                self.responses = responses
                self.prompts = []
                self.tok = SimpleNamespace(encode=self.encode,
                                           token_bytes=lambda tid: self.response.encode())
            def encode(self, prompt, parse_special):
                self.prompts.append(prompt)
                return [1]
            def _check_identity(self, value):
                pass
            def _ensure_loaded(self):
                pass
            def _sampling_keys(self, opts):
                return ''
            def _generate(self, ids, max_new, keys):
                self.response = self.responses.pop(0)
                yield 2

        Probe._stream = ns['_stream']
        offered = {'type': 'function', 'function': {
            'name': 'terminal', 'parameters': {'type': 'object',
            'properties': {'command': {'type': 'string'}}}}}
        opts = SimpleNamespace(ModelIdentity='', Prompt='User request',
                               Tools=json.dumps([offered]), Tokens=100, StopPrompts=[])
        valid = '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>'
        invalid = '<tool_call>{"name":"invented","arguments":{"command":"DO-NOT-RUN"}}</tool_call>'
        for bad in (invalid, '<tool_call>{"name":"terminal","arguments":{"command":"DO-NOT-RUN"}'):
            with self.subTest(bad=bad[:35]):
                probe = Probe([bad, valid])
                replies = list(ns['_stream'](probe, opts))
                calls = [t for r in replies for d in r.chat_deltas
                         for t in getattr(d, 'tool_calls', [])]
                content = ''.join(getattr(d, 'content', '') for r in replies for d in r.chat_deltas)
                self.assertEqual(len(probe.prompts), 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].name, 'terminal')
                self.assertEqual(json.loads(calls[0].arguments), {'command': 'pwd'})
                self.assertNotIn('Tool call rejected', content)
                self.assertNotIn('DO-NOT-RUN', ''.join(probe.prompts))
                self.assertIn('Retry', probe.prompts[1])

        probe = Probe([invalid, invalid, invalid])
        replies = list(ns['_stream'](probe, opts))
        self.assertEqual(len(probe.prompts), 3, 'never retry without a bound')
        self.assertFalse([t for r in replies for d in r.chat_deltas
                          for t in getattr(d, 'tool_calls', [])])
        self.assertEqual(''.join(getattr(d, 'content', '') for r in replies
                                 for d in r.chat_deltas),
                         'Tool call rejected: invalid format. Retry with an offered tool.')

        probe = Probe([valid + invalid])
        replies = list(ns['_stream'](probe, opts))
        self.assertEqual(len(probe.prompts), 1,
                         'a valid call has already been sent; never replay that turn')
        self.assertEqual(len([t for r in replies for d in r.chat_deltas
                              for t in getattr(d, 'tool_calls', [])]), 1)

    def test_thinking_tool_examples_are_never_executable(self):
        methods = next(n for n in module.body if isinstance(n, ast.ClassDef)
                       and n.name == 'StrataBackend').body
        stream = next(n for n in methods if isinstance(n, ast.FunctionDef) and n.name == '_stream')
        helpers = [next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name)
                   for name in ('_tool_system_block', '_generation_limit')]
        pb = SimpleNamespace(Reply=lambda **kw: SimpleNamespace(**kw),
                             ChatDelta=lambda **kw: SimpleNamespace(**kw),
                             ToolCallDelta=lambda **kw: SimpleNamespace(**kw))
        ns = dict(json=json, re=re, codecs=codecs, time=time, uuid=uuid, pb=pb,
                  ToolCallStreamParser=Parser,
                  IM_START='<|' + 'im_start' + '|>', IM_END='<|' + 'im_end' + '|>')
        exec(compile(ast.Module(body=helpers + [stream], type_ignores=[]), str(SOURCE), 'exec'), ns)
        thought = 'Example: <tool_call>{"name":"terminal","arguments":{"command":"NEVER-RUN"}}</tool_call> and <tool_call>example</tool_call>'
        answer = '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>'
        chunks = [b'<think>'] + [c.encode() for c in thought] + [b'</think>'] + [c.encode() for c in answer]
        class Probe:
            gen_lock = contextlib.nullcontext()
            stop_ids = set()
            think_open_ids = {1000}
            think_close_ids = {1001}
            engine = SimpleNamespace(max_context=2048)
            tok = SimpleNamespace(encode=lambda *a, **kw: [1],
                                  token_bytes=lambda tid: chunks[0] if tid == 1000 else
                                  b'</think>' if tid == 1001 else chunks[tid])
            def _check_identity(self, value): pass
            def _sampling_keys(self, opts): return ''
            def _generate(self, *args):
                yield 1000
                yield from range(1, len(thought) + 1)
                yield 1001
                yield from range(len(thought) + 2, len(chunks))
        tools = [{'type': 'function', 'function': {'name': 'terminal',
                  'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}}}}]
        opts = SimpleNamespace(ModelIdentity='', Prompt='Test', Tools=json.dumps(tools),
                               Tokens=300, StopPrompts=[])
        replies = list(ns['_stream'](Probe(), opts))
        calls = [t for r in replies for d in r.chat_deltas for t in getattr(d, 'tool_calls', [])]
        content = ''.join(getattr(d, 'content', '') for r in replies for d in r.chat_deltas)
        reasoning = ''.join(getattr(d, 'reasoning_content', '') for r in replies for d in r.chat_deltas)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].arguments), {'command': 'pwd'})
        self.assertEqual(content, '')
        self.assertIn(thought, reasoning)

    def test_zoo_function_parameter_variant_becomes_structured_tool_call(self):
        raw = ('Backend-Build ebenfalls erfolgreich. Jetzt der Frontend-Build:\n\n'
               '<tool_call>\n'
               '<function=execute_command>\n'
               '<parameter=command>\nnpm run build --workspace frontend\n</parameter>\n'
               '<parameter=cwd>\nc:/Users/jan/Documents/Visual Studio Code/Asset Management\n</parameter>\n'
               '<parameter=timeout>\n300\n</parameter>\n'
               '</function>\n</tool_call>')
        result = events(raw, specs={'execute_command': {
            'command': {'type': 'string'}, 'cwd': {'type': 'string'},
            'timeout': {'type': 'integer'},
        }})
        text = ''.join(e[1] for e in result if e[0] == 'text')
        calls = [e for e in result if e[0] == 'tool']
        self.assertEqual(text, 'Backend-Build ebenfalls erfolgreich. Jetzt der Frontend-Build:\n\n')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 'execute_command')
        self.assertEqual(json.loads(calls[0][2]), {
            'command': 'npm run build --workspace frontend',
            'cwd': 'c:/Users/jan/Documents/Visual Studio Code/Asset Management',
            'timeout': 300,
        })

    def test_json_call_still_works_and_alternate_requires_offered_schema(self):
        canonical = '<tool_call>{"name":"execute_command","arguments":{"command":"pwd"}}</tool_call>'
        self.assertEqual(events(canonical, size=1,
                                specs={'execute_command': {'command': {'type': 'string'}}}),
                         [('tool', 'execute_command', '{"command": "pwd"}')])
        self.assertEqual(events(canonical, size=1), [('invalid_tool',)])
        self.assertEqual(events(canonical, size=1, specs={'update_todo_list': {
            'todos': {'type': 'string'}}}), [('invalid_tool',)])
        alternate = ('<tool_call><function=execute_command>'
                     '<parameter=command>300</parameter></function></tool_call>')
        specs = {'execute_command': {'command': {'type': 'string'}}}
        for size in (1, 2, 7, 1024):
            with self.subTest(size=size):
                self.assertEqual(events(alternate, size=size, specs=specs),
                                 [('tool', 'execute_command', '{"command": "300"}')])
        self.assertEqual(events(alternate, size=1), [('invalid_tool',)])

    def test_malformed_or_unoffered_alternate_is_not_executed(self):
        specs = {'execute_command': {'command': {'type': 'string'},
                                     'timeout': {'type': 'integer'}}}
        for body in ('<function=delete_everything><parameter=command>pwd</parameter></function>',
                     '<function=execute_command><parameter=unknown>pwd</parameter></function>',
                     '<function=execute_command><parameter=timeout>tomorrow</parameter></function>',
                     '<function=execute_command><parameter=command>pwd</parameter>junk</function>',
                     '<function=execute_command><parameter=command>pwd</parameter>'
                     '<parameter=command>oops</parameter></function>'):
            with self.subTest(body=body):
                raw = '<tool_call>' + body + '</tool_call>'
                self.assertEqual(events(raw, specs=specs), [('invalid_tool',)])

    def test_final_report_recovers_unescaped_quotes_and_newlines_only(self):
        parser = Parser({'attempt_completion': {'result': {'type': 'string'},
                                                'command': {'type': 'string'}},
                         'execute_command': {'command': {'type': 'string'}}})
        malformed = ('<tool_call>{"name":"attempt_completion","arguments":{"result":'
                     '"# Bericht\nDas Feld "result" ist dokumentiert.\\nFertig."}}</tool_call>')
        events = []
        for char in malformed:
            events.extend(parser.feed(char))
        events.extend(parser.flush())
        calls = [event for event in events if event[0] == 'tool']
        self.assertEqual(len(calls), 1, events)
        self.assertEqual(calls[0][1], 'attempt_completion')
        self.assertEqual(json.loads(calls[0][2])['result'],
                         '# Bericht\nDas Feld "result" ist dokumentiert.\nFertig.')
        unsafe = Parser({'execute_command': {'command': {'type': 'string'}}})
        self.assertTrue(any(event[0] == 'invalid_tool' for event in
                            unsafe.feed(malformed) + unsafe.flush()))
        side_effect = Parser({'execute_command': {'command': {'type': 'string'}}})
        other = malformed.replace('attempt_completion', 'execute_command').replace('result', 'command')
        self.assertTrue(any(event[0] == 'invalid_tool' for event in
                            side_effect.feed(other) + side_effect.flush()))

    def test_rejection_diagnostic_exposes_shape_not_argument_values(self):
        secret = 'do-not-log-this-secret'
        parser = Parser({'attempt_completion': {'result': {'type': 'string'}}})
        raw = ('<tool_call><function=attempt_completion>'
               '<parameter=unexpected>' + secret + '</parameter>'
               '</function></tool_call>')
        self.assertEqual(parser.feed(raw), [('invalid_tool',)])
        diagnostic = parser.last_rejection
        self.assertIn('attempt_completion', diagnostic)
        self.assertIn('xml', diagnostic)
        self.assertNotIn(secret, diagnostic)
        self.assertNotIn('unexpected', diagnostic)
        self.assertEqual(parser.flush(), [])
        json_parser = Parser({'terminal': {'command': {'type': 'string'}}})
        malformed = ('<tool_call>{"name":"terminal","arguments":{"command":'
                     '"' + secret + '"oops}}</tool_call>')
        self.assertEqual(json_parser.feed(malformed), [('invalid_tool',)])
        self.assertIn('json_error=', json_parser.last_rejection)
        self.assertNotIn(secret, json_parser.last_rejection)
        self.assertNotIn('oops', json_parser.last_rejection)

    def test_incomplete_tool_block_is_rejected_not_shown(self):
        raw = '<tool_call><function=execute_command><parameter=command>pwd'
        self.assertEqual(events(raw, size=1, specs={'execute_command': {
            'command': {'type': 'string'}}}), [('invalid_tool',)])

    def test_zoo_duplicate_parameter_spellings_with_same_value(self):
        checklist = ('[ ] Read nodemailer usage in the three services\n'
                     '[ ] Research nodemailer 10.x breaking changes\n'
                     '[ ] Commit and push]')
        raw = ('<tool_call>\n<function=update_todo_list>\n'
               '<parameter=todos>\n' + checklist + '\n</parameter>\n'
               '<parameter name="todos">' + checklist + '\n</parameter>\n'
               '</function>\n</tool_call>')
        for size in (1, 3, 19, 999):
            with self.subTest(size=size):
                result = events(raw, size=size,
                                specs={'update_todo_list': {'todos': {'type': 'string'}}})
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0][0:2], ('tool', 'update_todo_list'))
                self.assertEqual(json.loads(result[0][2]), {'todos': checklist})

    def test_zoo_conflicting_duplicate_values_are_rejected(self):
        raw = ('<tool_call><function=execute_command>'
               '<parameter=command>echo safe</parameter>'
               '<parameter name="command">echo unsafe</parameter>'
               '</function></tool_call>')
        specs = {'execute_command': {'command': {'type': 'string'}}}
        self.assertEqual(events(raw, specs=specs), [('invalid_tool',)])

    def test_zoo_stray_closing_tags_after_complete_todo_parameter(self):
        # Seen in the Zoo screenshot: an intact offered argument followed by
        # stray closers from a second, incompatible tool-call dialect.
        checklist = ('[x] Read nodemailer usage in the three services to assess v10 API compatibility\n'
                     '[x] Research nodemailer 10.x breaking changes\n'
                     '[x] Verify Node.js >=20 in backend Dockerfile/engines and bump nodemailer')
        raw = ('<tool_call>\n<function=update_todo_list>\n'
               '<parameter=todos>\n' + checklist + '\n</parameter>\n'
               '</invoke>\n</parameter> </function> </tool_call>')
        specs = {'update_todo_list': {'todos': {'type': 'string'}}}
        for size in (1, 3, 17, 999):
            with self.subTest(size=size):
                self.assertEqual(events(raw, size=size, specs=specs),
                                 [('tool', 'update_todo_list',
                                   json.dumps({'todos': checklist}, ensure_ascii=False))])

    def test_bridge_stream_supplies_offered_schemas_to_fallback_parser(self):
        method = next(n for n in module.body if isinstance(n, ast.ClassDef) and
                      n.name == 'StrataBackend').body
        stream = next(n for n in method if isinstance(n, ast.FunctionDef) and n.name == '_stream')
        pb = SimpleNamespace(
            Reply=lambda **kw: SimpleNamespace(**kw),
            ChatDelta=lambda **kw: SimpleNamespace(**kw),
            ToolCallDelta=lambda **kw: SimpleNamespace(**kw))
        ns = dict(json=json, re=re, codecs=codecs, time=time, uuid=uuid,
                  pb=pb, ToolCallStreamParser=Parser,
                  IM_START='<|' + 'im_start' + '|>', IM_END='<|' + 'im_end' + '|>')
        tool_block = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_tool_system_block')
        budget_fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                         and n.name == '_generation_limit')
        exec(compile(ast.Module(body=[tool_block, budget_fn, stream], type_ignores=[]), str(SOURCE), 'exec'), ns)
        raw = ('Run the build:\n<tool_call><function=execute_command>'
               '<parameter=command>npm run build --workspace frontend</parameter>'
               '<parameter=timeout>300</parameter></function></tool_call>')
        chunks = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        think_bytes = b'</think>'
        class Tokenizer:
            def encode(self, prompt, parse_special):
                return [1]
            def token_bytes(self, tid):
                if tid == 1:
                    return think_bytes
                return chunks[tid - 2].encode()
        class Probe:
            def __init__(self):
                self.tok = Tokenizer()
                self.gen_lock = contextlib.nullcontext()
                self.stop_ids = set()
                self.think_close_ids = {1}
                self.engine = SimpleNamespace(max_context=1024)
            def _check_identity(self, value):
                pass
            def _ensure_loaded(self):
                pass
            def _sampling_keys(self, opts):
                return {}
            def _generate(self, ids, max_new, keys):
                self.seen_max_new = max_new
                yield 1
                for tid in range(2, 2 + len(chunks)):
                    yield tid
        Probe._stream = ns['_stream']
        tools = [{'type': 'function', 'function': {
            'name': 'execute_command', 'parameters': {'type': 'object',
            'properties': {'command': {'type': 'string'},
                           'timeout': {'type': 'integer'}}}}}]
        opts = SimpleNamespace(ModelIdentity='', Prompt='', Tools=json.dumps(tools),
                               Tokens=300, StopPrompts=[])
        replies = list(ns['_stream'](Probe(), opts))
        clean = ''.join(getattr(d, 'content', '') for r in replies for d in r.chat_deltas)
        calls = [t for r in replies for d in r.chat_deltas for t in getattr(d, 'tool_calls', [])]
        self.assertEqual(clean, 'Run the build:\n')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, 'execute_command')
        self.assertEqual(json.loads(calls[0].arguments),
                         {'command': 'npm run build --workspace frontend', 'timeout': 300})
        self.assertFalse(any(getattr(d, 'reasoning_content', '').isspace()
                             for r in replies for d in r.chat_deltas),
                         'whitespace-only reasoning makes an empty Thinking block in Zoo')
        self.assertNotIn(b'<tool_call>', b''.join(r.message for r in replies),
                         'raw tool markup must not reach the Go-side streaming extractor')

        # The Zoo screenshot variant must not become a visible assistant bubble.
        raw = ('<tool_call><function=execute_command>'
               '<parameter=command>npm run build --workspace frontend</parameter>'
               '</invoke></parameter></function></tool_call>')
        chunks = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        replies = list(ns['_stream'](Probe(), opts))
        calls = [t for r in replies for d in r.chat_deltas for t in getattr(d, 'tool_calls', [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].arguments),
                         {'command': 'npm run build --workspace frontend'})
        self.assertNotIn(b'<tool_call>', b''.join(r.message for r in replies))
        self.assertNotIn('<tool_call>', ''.join(getattr(d, 'content', '')
                                               for r in replies for d in r.chat_deltas))

        # Unknown/unparseable blocks must fail closed, not execute a guessed
        # function or reveal tool syntax in either bridge message or deltas.
        raw = ('<tool_call><function=delete_everything>'
               '<parameter=command>pwd</parameter></function></tool_call>')
        chunks = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        replies = list(ns['_stream'](Probe(), opts))
        self.assertFalse([t for r in replies for d in r.chat_deltas
                          for t in getattr(d, 'tool_calls', [])])
        self.assertNotIn(b'<tool_call>', b''.join(r.message for r in replies))
        safe_content = ''.join(getattr(d, 'content', '')
                               for r in replies for d in r.chat_deltas)
        self.assertIn('Tool call rejected', safe_content)
        self.assertNotIn('<tool_call>', safe_content)

        # A malformed final report is emitted as a structured completion in
        # streaming mode, even when Zoo offers an optional command parameter.
        opts.Tools = json.dumps([{'type': 'function', 'function': {
            'name': 'attempt_completion', 'parameters': {'type': 'object',
            'properties': {'result': {'type': 'string'},
                           'command': {'type': 'string'}}}}}])
        report = '# Bericht\nDas Feld "result" ist dokumentiert.'
        raw = ('<tool_call>{"name":"attempt_completion",'
               '"arguments":{"result":"' + report + '"}}</tool_call>')
        chunks = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        replies = list(ns['_stream'](Probe(), opts))
        calls = [t for r in replies for d in r.chat_deltas
                 for t in getattr(d, 'tool_calls', [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, 'attempt_completion')
        self.assertEqual(json.loads(calls[0].arguments), {'result': report})
        self.assertNotIn(b'<tool_call>', b''.join(r.message for r in replies))
        self.assertNotIn('<tool_call>', ''.join(getattr(d, 'content', '')
                                               for r in replies for d in r.chat_deltas))
        opts.Tools = json.dumps(tools)

        # Tool-first responses have no genuine content or reasoning to make
        # LocalAI prefer chat deltas. They must still carry the tool call
        # without sending a synthetic blank Thinking delta or raw markup.
        think_bytes = b''
        raw = ('<tool_call>{"name":"execute_command",'
               '"arguments":{"command":"npm run build --workspace frontend"}}</tool_call>')
        chunks = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        replies = list(ns['_stream'](Probe(), opts))
        self.assertEqual(b''.join(r.message for r in replies), b'')
        self.assertEqual([d.reasoning_content for r in replies for d in r.chat_deltas
                          if getattr(d, 'reasoning_content', '')], [])
        self.assertEqual(len([t for r in replies for d in r.chat_deltas
                              for t in getattr(d, 'tool_calls', [])]), 1)
        predict = next(n for n in method if isinstance(n, ast.FunctionDef) and n.name == 'Predict')
        exec(compile(ast.Module(body=[predict], type_ignores=[]), str(SOURCE), 'exec'), ns)
        Probe._stream = ns['_stream']
        result = ns['Predict'](Probe(), opts, None)
        self.assertEqual(result.message, b'',
                         'non-streaming Predict must not reparse raw tool markup')
        predict_calls = [t for d in result.chat_deltas for t in getattr(d, 'tool_calls', [])]
        self.assertEqual(len(predict_calls), 1)
        self.assertEqual(predict_calls[0].name, 'execute_command')
        opts.Tokens = 0
        probe = Probe()
        list(ns['_stream'](probe, opts))
        self.assertEqual(probe.seen_max_new, probe.engine.max_context - 1 - 8)

    def test_empty_think_frame_does_not_create_reasoning_block(self):
        methods = next(n for n in module.body if isinstance(n, ast.ClassDef) and
                       n.name == 'StrataBackend').body
        stream = next(n for n in methods if isinstance(n, ast.FunctionDef) and n.name == '_stream')
        tool_block = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_tool_system_block')
        pb = SimpleNamespace(Reply=lambda **kw: SimpleNamespace(**kw),
                             ChatDelta=lambda **kw: SimpleNamespace(**kw),
                             ToolCallDelta=lambda **kw: SimpleNamespace(**kw))
        ns = dict(json=json, re=re, codecs=codecs, time=time, uuid=uuid,
                  pb=pb, ToolCallStreamParser=Parser,
                  IM_START='<|' + 'im_start' + '|>', IM_END='<|' + 'im_end' + '|>')
        budget_fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                         and n.name == '_generation_limit')
        exec(compile(ast.Module(body=[tool_block, budget_fn, stream], type_ignores=[]), str(SOURCE), 'exec'), ns)
        tokens = [b'<think>', b'\n', b'\n', b'</think>',
                  b'<tool_call>{"name":"update_todo_list",',
                  b'"arguments":{"todos":"[ ] Check"}}</tool_call>']
        class Probe:
            tok = SimpleNamespace(encode=lambda *a, **kw: [1], token_bytes=lambda tid: tokens[tid])
            gen_lock = contextlib.nullcontext()
            think_close_ids = {3}
            stop_ids = set()
            def _check_identity(self, value):
                pass
            def _sampling_keys(self, opts):
                return {}
            def _generate(self, ids, max_new, keys):
                yield from range(len(tokens))
        tools = [{'type': 'function', 'function': {'name': 'update_todo_list',
                  'parameters': {'type': 'object', 'properties': {'todos': {'type': 'string'}}}}}]
        opts = SimpleNamespace(ModelIdentity='', Prompt='', Tools=json.dumps(tools),
                               Tokens=100, StopPrompts=[])
        replies = list(ns['_stream'](Probe(), opts))
        reasoning = ''.join(getattr(d, 'reasoning_content', '') for r in replies for d in r.chat_deltas)
        self.assertEqual(reasoning, '')
        self.assertNotIn(b'<think>', b''.join(r.message for r in replies))
        self.assertEqual(len([t for r in replies for d in r.chat_deltas
                              for t in getattr(d, 'tool_calls', [])]), 1)

        # A nonempty thought still streams to the client normally.
        tokens = [b'<think>', b'Plan the next step', b'\n', b'</think>',
                  b'<tool_call>{"name":"update_todo_list",',
                  b'"arguments":{"todos":"[ ] Check"}}</tool_call>']
        replies = list(ns['_stream'](Probe(), opts))
        reasoning = ''.join(getattr(d, 'reasoning_content', '') for r in replies for d in r.chat_deltas)
        self.assertIn('Plan the next step', reasoning)


if __name__ == '__main__':
    unittest.main()
