"""Opt-in, private capture of rejected tool envelopes (never journal the payload)."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import unittest

SOURCE = Path(__file__).with_name('strata_grpc_backend.py')
module = ast.parse(SOURCE.read_text())
parser_node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'ToolCallStreamParser')
capture_node = next((n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_capture_rejected_tool'), None)
ns = {'json': json, 'os': os, 're': re, 'stat': stat, 'time': time,
      'uuid': __import__('uuid')}
if capture_node:
    exec(compile(ast.Module(body=[capture_node], type_ignores=[]), str(SOURCE), 'exec'), ns)
exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(SOURCE), 'exec'), ns)
Parser = ns['ToolCallStreamParser']


class RejectedToolCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / 'private'
        self.directory.mkdir(mode=0o700)
        self.cfg = {'rejected_tool_capture': {
            'directory': str(self.directory), 'until_epoch': time.time() + 60}}
        self.specs = {'read_file': {'path': {'type': 'string'},
                                    'offset': {'type': 'integer'}}}
        self.required = {'read_file': {'path'}}
        self.raw = ('<tool_call>\n<function=read_file>\n'
                    '<parameter=path>private & <tag>\n</parameter>\n'
                    '<parameter=unknown>secret-value</parameter>\n'
                    '</function>\n</tool_call>')

    def reject(self, cfg):
        parser = Parser(self.specs, self.required, 'native_xml',
                        strict_markup=True,
                        reject_capture=lambda block, reason, specs, required:
                        ns['_capture_rejected_tool'](cfg, block, reason, specs, required))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            events = []
            for char in self.raw:
                events.extend(parser.feed(char))
            events.extend(parser.flush())
        return events, output.getvalue()

    def test_captures_exact_rejected_block_and_schema_in_private_file(self):
        events, printed = self.reject(self.cfg)
        self.assertEqual(events, [('invalid_tool',)])
        files = list(self.directory.iterdir())
        self.assertEqual(len(files), 1)
        self.assertEqual(stat.S_IMODE(files[0].stat().st_mode), 0o600)
        data = json.loads(files[0].read_text())
        self.assertEqual(data['block'], self.raw[len('<tool_call>'):-len('</tool_call>')])
        self.assertEqual(data['reason'], 'unparseable-block')
        self.assertEqual(data['tool_specs'], self.specs)
        self.assertEqual(data['tool_required'], {'read_file': ['path']})
        self.assertNotIn('secret-value', printed)
        self.assertNotIn('private &', printed)

    def test_disabled_and_expired_capture_create_no_files(self):
        for cfg in ({}, {'rejected_tool_capture': {'directory': str(self.directory),
                                                  'until_epoch': time.time() - 1}}):
            self.assertEqual(self.reject(cfg)[0], [('invalid_tool',)])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_refuses_world_readable_directory_without_leaking_payload(self):
        self.directory.chmod(0o755)
        events, printed = self.reject(self.cfg)
        self.assertEqual(events, [('invalid_tool',)])
        self.assertEqual(list(self.directory.iterdir()), [])
        self.assertNotIn('secret-value', printed)

    def test_orphan_closer_captures_the_actual_delimiter(self):
        parser = Parser(self.specs, self.required, 'native_xml', strict_markup=True,
                        reject_capture=lambda block, reason, specs, required:
                        ns['_capture_rejected_tool'](self.cfg, block, reason, specs, required))
        self.assertIn(('invalid_tool',), parser.feed('</tool_call>'))
        captured = json.loads(next(self.directory.iterdir()).read_text())
        self.assertEqual(captured['block'], '</tool_call>')
        self.assertEqual(captured['reason'], 'orphan-tool-delimiter')

    def test_valid_call_creates_no_capture(self):
        parser = Parser(self.specs, self.required, 'native_xml', strict_markup=True,
                        reject_capture=lambda block, reason, specs, required:
                        ns['_capture_rejected_tool'](self.cfg, block, reason, specs, required))
        raw = ('<tool_call><function=read_file><parameter=path>README.md</parameter>'
               '</function></tool_call>')
        self.assertEqual([event[0] for event in parser.feed(raw) + parser.flush()], ['tool'])
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
