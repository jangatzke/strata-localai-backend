"""Offline protocol tests: never start an engine or load a model."""
import base64
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(os.environ.get('INKLING_TOKENIZER_DIR', Path(__file__).resolve().parent))
TOKENIZER_CONFIG = {'tokenizer': str(HERE / 'tokenizer.json'),
                    'strata_repo_tools': os.environ.get('INKLING_TOOLS_DIR', str(Path(__file__).resolve().parent))}
sys.path.insert(0, str(Path(__file__).resolve().parent))
if os.environ.get('STRATA_BRIDGE_SOURCE_DIR'):
    sys.path.append(os.environ['STRATA_BRIDGE_SOURCE_DIR'])


@unittest.skipUnless(Path(TOKENIZER_CONFIG['tokenizer']).is_file() and
                     (Path(TOKENIZER_CONFIG['strata_repo_tools']) / 'strata_tokenizer.py').is_file(),
                     'Inkling HF tokenizer fixture and Strata tools required')
class InklingBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = importlib.import_module('inkling_grpc_backend')

    def test_main_wires_inkling_backend_not_qwen(self):
        import inspect
        body = inspect.getsource(self.m.main)
        self.assertIn('backend = InklingBackend(cfg, tok', body)

    def test_parallel_command_rejects_qwen_batch_flag(self):
        engine = self.m.Engine({'parallel': 2})
        self.assertEqual(engine.command_args(['engine', '--serve']),
                         ['engine', '--serve', '--parallel', '2'])
        with self.assertRaisesRegex(ValueError, 'batch'):
            engine.command_args(['engine', '--batch', '2'])

    def test_from_hf_o200k_specials_and_stop_ids(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        self.assertEqual(tok.pre, 'gpt-4o')
        self.assertTrue(tok.ignore_merges)
        self.assertEqual(tok.encode('<|content_text|>', parse_special=True), [200004])
        self.assertEqual(tok.encode('<|unused_200054|>', parse_special=True), [200054])
        self.assertEqual(tok.decode(tok.encode('hello é 🚀')), 'hello é 🚀')
        self.assertEqual(self.m.stop_token_ids(tok), {200006, 199999})

    def test_model_channel_stream_yields_content_and_reasoning(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        output = [200001, 200008, *tok.encode('thought'), 200010,
                  200001, 200004, *tok.encode('answer'), 200006]
        b._ensure_loaded = lambda: None
        b._generate = lambda *args, **kwargs: iter(output)
        o = self._request('<|message_user|><|content_text|>hi<|end_message|><|message_model|>')
        replies = list(b._stream(o))
        self.assertEqual(''.join(d.reasoning_content for r in replies for d in r.chat_deltas), 'thought')
        self.assertEqual(''.join(d.content for r in replies for d in r.chat_deltas), 'answer')
        self.assertNotIn('<|', b''.join(r.message for r in replies).decode())

    def test_stop_token_drains_terminal_before_next_request(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        b._ensure_loaded = lambda: None
        b._generate = lambda *args, **kwargs: iter([200004, *tok.encode('OK'), 200006])
        b.engine.alive = lambda: True
        actions = []
        b.engine.stop = lambda: actions.append('stop')
        b.engine.drain = lambda: actions.append('drain')
        o = self._request('<|message_user|><|content_text|>hi<|end_message|><|message_model|>')
        self.assertEqual(''.join(r.message.decode() for r in b._stream(o)), 'OK')
        self.assertEqual(actions, ['drain'])

    def test_stop_token_emits_monitor_completion_after_valid_body(self):
        from contextlib import redirect_stdout
        from io import StringIO
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        b._generate = lambda *args, **kwargs: iter([200004, *tok.encode('OK'), 200006])
        b.engine.drain = lambda: None
        o = self._request('<|message_user|><|content_text|>hi<|end_message|><|message_model|>')
        captured = StringIO()
        with redirect_stdout(captured):
            self.assertEqual(''.join(r.message.decode() for r in b._stream(o)), 'OK')
        self.assertRegex(captured.getvalue(),
                         r'\[strata-test\] stop_ids hit at token 2: tid=200006')

    def test_openai_tools_without_rendered_declaration_are_synthesized(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        seen = {}
        def fake_generate(ids, *args, **kwargs):
            seen['ids'] = list(ids)
            return iter([200001, *tok.encode('lookup'), 200049,
                         *tok.encode('{"name":"lookup","args":{"q":"a"}}'), 200010, 200006])
        b._ensure_loaded = lambda: None
        b._generate = fake_generate
        o = self._request('<|message_user|><|content_text|>go<|end_message|><|message_model|>',
                          tools=json.dumps([{'type': 'function', 'function': {
                              'name': 'lookup', 'description': 'Find things',
                              'parameters': {'type': 'object', 'properties': {'q': {'type': 'string'}},
                                             'required': ['q']}}}]))
        replies = list(b._stream(o))
        calls = [tc for r in replies for d in r.chat_deltas for tc in d.tool_calls]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, 'lookup')
        self.assertEqual(json.loads(calls[0].arguments), {'q': 'a'})
        # The model was trained with bare {"type":"object"} parameter specs; full
        # JSON schemas (properties/required) make it end the turn immediately.
        prefix = tok.encode('<|message_system|>tool_declare<|content_xml|>'
                            '[{"description":"Find things","name":"lookup",'
                            '"parameters":{"type":"object"},"type":"function"}]<|end_message|>',
                            parse_special=True)
        self.assertEqual(seen['ids'][:len(prefix)], prefix)

    def test_synthesized_declaration_accepts_calls_without_schema_args(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        b._ensure_loaded = lambda: None
        b._generate = lambda *args, **kwargs: iter([200001, *tok.encode('lookup'), 200049,
                                                    *tok.encode('{"name":"lookup","args":{}}'),
                                                    200010, 200006])
        b.engine.drain = lambda: None
        o = self._request('<|message_user|><|content_text|>go<|end_message|><|message_model|>',
                          tools=json.dumps([{'type': 'function', 'function': {
                              'name': 'lookup', 'description': 'Find things',
                              'parameters': {'type': 'object', 'properties': {'q': {'type': 'string'}},
                                             'required': ['q']}}}]))
        calls = [tc for r in b._stream(o) for d in r.chat_deltas for tc in d.tool_calls]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, 'lookup')
        self.assertEqual(json.loads(calls[0].arguments), {})

    def test_empty_tools_array_is_treated_as_no_tools(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        b._ensure_loaded = lambda: None
        b._generate = lambda *args, **kwargs: iter([200001, 200004, *tok.encode('OK'), 200006])
        b.engine.drain = lambda: None
        o = self._request('<|message_user|><|content_text|>hi<|end_message|><|message_model|>',
                          tools='[]')
        self.assertEqual(''.join(r.message.decode() for r in b._stream(o)), 'OK')

    def test_invalid_tool_call_does_not_emit_success_marker(self):
        from contextlib import redirect_stdout
        from io import StringIO
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        b._generate = lambda *args, **kwargs: iter([200001, *tok.encode('other'), 200049,
                                                   *tok.encode('{"name":"other","args":{}}'),
                                                   200010, 200006])
        b.engine.drain = lambda: None
        o = self._request('<|message_system|>tool_declare<|content_xml|>[]<|end_message|>'
                          '<|message_user|><|content_text|>hi<|end_message|><|message_model|>',
                          tools=json.dumps([{'function': {'name': 'lookup', 'parameters': {}}}]))
        captured = StringIO()
        with redirect_stdout(captured), self.assertRaises(self.m.ToolFormatError):
            list(b._stream(o))
        self.assertNotIn('stop_ids hit', captured.getvalue())

    def test_json_tool_call_is_structured_and_offered_only(self):
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name': 'inkling', 'parallel': 1}, tok)
        call = '{"name":"lookup","args":{"q":"a</tool_call>"}}'
        output = [200001, *tok.encode('lookup'), 200049,
                  *tok.encode(call), 200010, 200006]
        b._ensure_loaded = lambda: None
        b._generate = lambda *args, **kwargs: iter(output)
        o = self._request('<|message_system|>tool_declare<|content_xml|>[]<|end_message|>'
                          '<|message_user|><|content_text|>go<|end_message|><|message_model|>',
                          tools=json.dumps([{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'q':{'type':'string'}},'required':['q']}}}]))
        replies = list(b._stream(o))
        calls = [tc for r in replies for d in r.chat_deltas for tc in d.tool_calls]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, 'lookup')
        self.assertEqual(json.loads(calls[0].arguments), {'q':'a</tool_call>'})
        self.assertNotIn('args', b''.join(r.message for r in replies).decode())
        b._generate = lambda *args, **kwargs: iter([200001, *tok.encode('unoffered'), 200049,
                                                      *tok.encode('{"name":"unoffered","args":{}}'), 200010, 200006])
        with self.assertRaisesRegex(self.m.ToolFormatError, 'unoffered'):
            list(b._stream(o))

    def test_image_paths_are_percent_encoded_and_cleaned(self):
        from PIL import Image
        from io import BytesIO
        buf = BytesIO()
        Image.new('RGB', (1, 1), 'red').save(buf, 'PNG')
        source = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        tok = self.m.build_tokenizer(TOKENIZER_CONFIG)
        b = self.m.InklingBackend({'model_name':'inkling','parallel':1,'vision':True}, tok)
        paths = b._prepare_images([source], tok.encode('<|content_image|><|unused_200054|>', parse_special=True))
        try:
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())
            self.assertEqual(paths[0].read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(self.m.image_keys(paths), ' image=' + self.m.quote(str(paths[0]), safe=''))
            self.assertEqual(tok.encode('<|unused_200054|>', parse_special=True), [200054])
        finally:
            for p in paths: p.unlink(missing_ok=True)
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            b._prepare_images([source], tok.encode('no image'))
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            b._prepare_images(['http://127.0.0.1/img'], [200054])

    @staticmethod
    def _request(prompt, tools=''):
        from types import SimpleNamespace
        return SimpleNamespace(Prompt=prompt, ModelIdentity='inkling', Images=[], Tools=tools,
                               Tokens=20, Temperature=0, TopP=0, TopK=0, MinP=0, Penalty=0,
                               FrequencyPenalty=0, PresencePenalty=0, Seed=0, StopPrompts=[])


if __name__ == '__main__':
    unittest.main()
