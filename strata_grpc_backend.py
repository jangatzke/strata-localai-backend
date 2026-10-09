#!/usr/bin/env python3
"""Strata LocalAI backend: a gRPC bridge between LocalAI and the Strata engine.

Implements LocalAI's backend.Backend proto. Model loading / unloading:

  LoadModel -> spawns the Strata engine (`strata --serve ...`) with the args
              from the JSON config; waits for the engine's READY line.
  Predict / PredictStream -> tokenizes the prompt with the pack's tokenizer;
              uses legacy GEN/T/DONE by default or two-slot BGEN/BT/BDONE when
              the config explicitly sets parallel=2.
  Free     -> sends QUIT to the engine and waits for it to exit, releasing
              all GPU/RAM so LocalAI can load other models.

The tokenizer comes from the Strata repo's tools/strata_tokenizer.py and the
pack's vocab.json / merges.txt / token_type.json.
"""

import argparse
import base64
import codecs
import hashlib
import io
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from concurrent import futures
from pathlib import Path

import grpc

import backend_pb2 as pb
import backend_pb2_grpc as pb_grpc

# ChatML special tokens, used as stop markers (assembled so the literals
# never appear verbatim in logs of this file's source).
IM_END = "<|" + "im_end" + "|>"
IM_START = "<|" + "im_start" + "|>"
VISION_START = "<|" + "vision_start" + "|>"
IMAGE_PAD = "<|" + "image_pad" + "|>"

READY_TIMEOUT_S = 1200   # engine load takes minutes (experts + PLE)
GEN_TIMEOUT_S = 3600     # per-request hard ceiling
PROGRESS_STALL_S = 180   # no PP/token progress after generation starts


class ToolFormatError(RuntimeError):
    """A tool-enabled turn failed closed before any calls were published."""


class EngineLogFollower:
    """Tails the engine's log file and mirrors interesting lines to stdout.

    LocalAI cannot capture an external backend's stderr (there is no process
    it spawned), so the backend forwards the engine's own log lines to its
    stdout, which lands in the journal of the systemd service.
    """

    def __init__(self, path: str, pattern: str | None):
        self.path = path
        self.rx = re.compile(pattern) if pattern else None
        if path and Path(path).exists():
            self.pos = Path(path).stat().st_size
        else:
            self.pos = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            try:
                if not self.path or not Path(self.path).exists():
                    time.sleep(5)
                    continue
                size = Path(self.path).stat().st_size
                if size < self.pos:          # log rotated/truncated
                    self.pos = 0
                if size == self.pos:
                    time.sleep(2)
                    continue
                with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(self.pos)
                    data = f.read()
                self.pos = f.tell()
                for line in data.splitlines():
                    if not line.strip():
                        continue
                    if self.rx is None or self.rx.search(line):
                        print(f"[strata-engine] {line}", flush=True)
            except Exception:
                time.sleep(5)


DEFAULT_LOG_PATTERN = r"INFO |ERR |READY |error|Error|watchdog|serve:"


class EngineError(RuntimeError):
    pass


class Engine:
    """The resident `strata --serve` process: stdin/stdout token protocol."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.parallel = int(cfg.get("parallel", 1))
        if self.parallel not in (1, 2):
            raise ValueError("parallel must be 1 or 2")
        self.log_path = cfg.get("log")
        self.proc = None
        self.lines = queue.Queue()
        self.pump = None
        self.ended = True
        self.max_context = 0
        self.can_stop = False
        self.info = {}
        self.log = None
        self.send_lock = threading.Lock()
        self.admission_lock = threading.Lock()
        self.admission_broken = False
        self.slot_lines = {slot: queue.Queue() for slot in range(self.parallel)}
        self.free_slots = queue.Queue()
        for slot in range(self.parallel):
            self.free_slots.put(slot)

    def command_args(self, base: list[str]) -> list[str]:
        """Add the engine batch switch only for explicit parallel mode."""
        args = list(base)
        has_upstream_slots = any(arg in ("--batch", "--slots") or
                                 arg.startswith(("--batch=", "--slots=")) for arg in args)
        if self.parallel == 2 and not has_upstream_slots:
            args += ["--batch", "2"]
        return args

    def available_slots(self) -> int:
        return self.free_slots.qsize()

    def validate_batch_slots(self):
        if self.parallel != 2:
            return
        reported = self.info.get("batch_slots")
        if not isinstance(reported, int) or reported < 2:
            raise EngineError(f"parallel=2 requires engine INFO batch_slots>=2; got batch_slots={reported}")

    def _env(self) -> dict:
        env = dict(os.environ)
        for k, v in (self.cfg.get("env") or {}).items():
            env[str(k)] = str(v)
        dirs = [d for d in self.cfg.get("lib_dirs") or [] if Path(d).is_dir()]
        if dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        if self.cfg.get("backend") == "hip" and self.cfg.get("gpu") is not None:
            env["HIP_VISIBLE_DEVICES"] = str(self.cfg["gpu"])
        return env

    def alive(self) -> bool:
        return self.proc is not None and not self.ended and self.proc.poll() is None

    def start(self):
        """Spawn the engine and block (bounded) until READY."""
        if self.alive():
            return
        self.ended = True
        self.max_context = 0
        self.can_stop = False
        self.info = {}
        self.log = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        args = self.command_args([self.cfg["exe"], "--serve", *self.cfg["args"]])
        self.proc = subprocess.Popen(
            args, cwd=self.cfg.get("cwd") or ".", stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8",
            bufsize=1, env=self._env())
        self.lines = queue.Queue()
        self.admission_broken = False
        self.slot_lines = {slot: queue.Queue() for slot in range(self.parallel)}
        self.free_slots = queue.Queue()
        for slot in range(self.parallel):
            self.free_slots.put(slot)
        self.pump = threading.Thread(target=self._pump, daemon=True)
        self.pump.start()
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=5.0)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise EngineError("the engine exited before READY (see the log)")
                continue
            if line is None:
                raise EngineError("the engine exited before READY (see the log)")
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            elif line.startswith("READY"):
                parts = line.split()
                self.max_context = int(parts[1]) if len(parts) > 1 else 0
                self.can_stop = "stop" in parts[2:]
                self.ended = False
                if self.max_context <= 0:
                    raise EngineError("the engine reported no context")
                self.validate_batch_slots()
                return
        raise EngineError(f"the engine did not become READY within {READY_TIMEOUT_S}s")

    def _pump(self):
        proc, q = self.proc, self.lines
        for line in proc.stdout:
            self._route_line(line)
        if self.proc is proc:
            self.ended = True
        q.put(None)
        if self.parallel == 2:
            for target in self.slot_lines.values():
                target.put(None)

    def _route_line(self, line: str):
        """Dispatch tagged batch output without letting requests steal tokens."""
        if self.parallel == 2 and line.startswith(("BT ", "BDONE ")):
            parts = line.split()
            try:
                slot = int(parts[1])
                target = self.slot_lines[slot]
            except (ValueError, IndexError, KeyError):
                self.lines.put("ERR malformed batch output: " + line.strip())
                return
            target.put(line)
            return
        if line.startswith("INFO "):
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                self.info[k] = int(v) if v.lstrip("-").isdigit() else v
        self.lines.put(line)

    def _send(self, line: str):
        if not self.alive():
            raise EngineError("the engine is not running")
        with self.send_lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()

    def stop(self, slot=None):
        """Ask the engine to abort the current request mid-generation."""
        if self.alive() and self.can_stop:
            try:
                if self.parallel == 2:
                    if slot is not None:
                        self._send(f"BSTOP {slot}")
                else:
                    self._send("STOP")
            except Exception:
                pass

    def drain(self, timeout_s: float = 120.0, slot=None):
        """Discard leftovers of an aborted generation.

        After STOP the engine still emits its remaining `T` lines and the
        final `DONE`. Leaving them in the queue would make the NEXT request
        read the previous conversation's tokens (lost step), so every abort
        path drains until DONE (or a bounded timeout) before returning.
        """
        if self.parallel == 2:
            if slot is None:
                return
            source = self.slot_lines[slot]
            terminal = "BDONE"
        else:
            source = self.lines
            terminal = "DONE"
        if source is None:
            return
        deadline = time.time() + timeout_s
        n = 0
        drained_terminal = False
        while time.time() < deadline:
            if not self.alive():
                break
            try:
                line = source.get(timeout=min(1.0, max(0.001, deadline - time.time())))
            except queue.Empty:
                # Silence during prefill is not an abort acknowledgement.
                # Retain ownership until terminal output or the hard deadline.
                continue
            if line is None:
                break
            if line.startswith(terminal) or (self.parallel == 2 and line.startswith("ERR")):
                n += 1
                drained_terminal = True
                break
            n += 1
        if n:
            scope = f" for slot {slot}" if slot is not None else ""
            print(f"[strata-backend] drain: discarded {n} leftover engine lines{scope}", flush=True)
        if self.parallel == 2 and self.alive() and not drained_terminal:
            raise EngineError(f"batch slot {slot} did not drain safely without BDONE")
        if self.parallel == 1 and self.alive() and not drained_terminal:
            # Delayed untagged output must never be handed to a new GEN owner.
            # start() clears this quarantine only after spawning a fresh engine.
            self.admission_broken = True
            raise EngineError("serial generation did not drain safely without DONE; reload the engine")

    def batch_generate(self, max_new, keys, ids, embeddings=None, check_active=None):
        """Admit and serve one request on an isolated Strata batch slot."""
        if self.parallel != 2:
            raise EngineError("batch generation requires parallel=2")
        while True:
            if self.admission_broken:
                raise EngineError("batch admission is unreconciled; reload the engine")
            if check_active is not None:
                check_active()
            try:
                slot = self.free_slots.get(timeout=0.1)
                break
            except queue.Empty:
                continue
        admitted = completed = False
        pending = stop_sent = False
        started = time.time()
        try:
            command = "BGENI" if embeddings is not None else "BGEN"
            image_arg = f" {embeddings}" if embeddings is not None else ""
            line = (f"{command} {slot} {max_new}{keys}{image_arg} "
                    f"{','.join(map(str, ids))}")
            # BGEN admission is serialized by Strata. Its untagged T/DONE
            # compatibility output must be consumed here, never by active slots.
            with self.admission_lock:
                if self.admission_broken:
                    raise EngineError("batch admission is unreconciled; reload the engine")
                if check_active is not None:
                    check_active()
                self._send(line)
                pending = True
                deadline = time.time() + GEN_TIMEOUT_S
                admission_tokens = []
                admission_done = False
                serial_fallback = None
                failure = None
                try:
                    while True:
                        if failure is None:
                            try:
                                if check_active is not None:
                                    check_active()
                                if time.time() >= deadline:
                                    raise EngineError(f"batch admission timed out for slot {slot}")
                            except Exception as exc:
                                # BGEN is already on stdin. Retain the untagged
                                # owner until BADM resolves it. BSTOP targets
                                # only this slot, never an active peer.
                                failure = exc
                                self.stop(slot)
                                stop_sent = True
                                deadline = time.time() + 120.0
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise EngineError(f"batch admission unreconciled for slot {slot}") from failure
                        try:
                            response = self.lines.get(timeout=min(0.1, remaining))
                        except queue.Empty:
                            if not self.alive():
                                raise EngineError("the engine died during batch admission")
                            continue
                        if response is None:
                            raise EngineError("the engine died during batch admission")
                        if response.startswith("BADM "):
                            parts = response.split()
                            if len(parts) < 3 or int(parts[1]) != slot:
                                raise EngineError("batch admission returned the wrong slot")
                            if parts[2] == "1":
                                admitted = True
                            elif parts[2] == "0" and admission_done:
                                serial_fallback = admission_tokens
                            else:
                                raise EngineError(f"batch admission rejected slot {slot}")
                            pending = False
                            if failure is not None:
                                raise failure
                            if check_active is not None:
                                check_active()
                            break
                        if response.startswith("ERR"):
                            raise EngineError(response.strip())
                        if response.startswith("T "):
                            admission_tokens.append(int(response.split()[1]))
                        elif response.startswith("DONE"):
                            admission_done = True
                        # PP and INFO also belong to serialized admission.
                finally:
                    if pending:
                        # Fail closed: no new owner may consume delayed BADM.
                        # Already admitted tagged peers remain independent.
                        self.admission_broken = True

            if serial_fallback is not None:
                for token in serial_fallback:
                    yield token
                completed = True
                return {"tokens": len(serial_fallback), "elapsed": time.time() - started,
                        "reason": "serial-fallback"}

            n = 0
            # BGEN reads the prompt as a one-token legacy GEN admission. That
            # first T token is part of the response even when BADM continues
            # decoding in the selected batch slot.
            for token in admission_tokens:
                n += 1
                yield token
            while True:
                if check_active is not None:
                    check_active()
                if time.time() - started > GEN_TIMEOUT_S:
                    raise EngineError(f"generation timed out for slot {slot}")
                try:
                    response = self.slot_lines[slot].get(timeout=1.0)
                except queue.Empty:
                    if not self.alive():
                        raise EngineError("the engine died mid-generation")
                    continue
                if response is None:
                    raise EngineError("the engine died mid-generation")
                if response.startswith("BT "):
                    n += 1
                    yield int(response.split()[2])
                elif response.startswith("BDONE "):
                    parts = response.split(maxsplit=5)
                    completed = True
                    return {"tokens": n,
                            "generated": int(parts[2]) if len(parts) > 2 else n,
                            "reason": parts[3] if len(parts) > 3 else "",
                            "engine_ms": float(parts[4]) if len(parts) > 4 else 0.0,
                            "elapsed": time.time() - started}
                elif response.startswith("ERR"):
                    raise EngineError(response.strip())
        finally:
            if admitted and not completed:
                if not stop_sent:
                    self.stop(slot)
                self.drain(slot=slot)
            # A rejected admission cannot have active output, but remove any
            # malformed/stale tagged lines before this numeric slot is reused.
            while True:
                try:
                    self.slot_lines[slot].get_nowait()
                except queue.Empty:
                    break
            if not pending or not self.alive():
                self.free_slots.put(slot)

    def free(self):
        """QUIT the engine and release GPU/RAM."""
        if self.proc is None:
            return
        try:
            if self.alive():
                self._send("QUIT")
                try:
                    self.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
        finally:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except Exception:
                pass
            self.proc = None
            self.lines = None
            self.ended = True
            self.can_stop = False


def _tool_system_block(tools_json: str, dialect='json') -> str:
    """Describe offered tools in the configured generation dialect.

    LocalAI does not inject definitions for this custom chat template.
    native_xml matches the active Strata pack; json retains compatibility
    with deployments using the older bridge-specific contract.
    """
    try:
        tools = json.loads(tools_json)
    except Exception:
        return ""
    if not isinstance(tools, list) or not tools:
        return ""
    import re as _re
    _FN_EX = _re.compile(r"<function=[^>]*>.*?</function>", _re.DOTALL)
    _FN_OPEN = _re.compile(r"<function=[^>]*>")
    _PARAM = _re.compile(r"</?parameter=[^>]*>")
    def _sanitize(desc):
        if not isinstance(desc, str):
            return desc
        desc = _FN_EX.sub("", desc)
        desc = _FN_OPEN.sub("", desc)
        desc = _PARAM.sub("", desc)
        return desc
    specs = []
    for t in tools:
        if isinstance(t, dict):
            try:
                fn = t.get("function") or t
                if isinstance(fn.get("description"), str):
                    fn["description"] = _sanitize(fn["description"])
                params = (fn.get("parameters") or {}).get("properties") or {}
                for v in params.values():
                    if isinstance(v, dict) and isinstance(v.get("description"), str):
                        v["description"] = _sanitize(v["description"])
            except Exception:
                pass
            specs.append(json.dumps(t, ensure_ascii=False))
    if not specs:
        return ""
    if dialect == 'native_xml':
        return ('# Tools\n\nYou have access to the following functions:\n\n<tools>\n' +
                '\n'.join(specs) + '\n</tools>\n\n'
                'If you choose to call a function ONLY reply in the following format with NO suffix:\n\n'
                '<tool_call>\n<function=example_function_name>\n'
                '<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
                '<parameter=example_parameter_2>\nThis is the value for the second parameter\n'
                'that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n'
                '<IMPORTANT>\nReminder:\n'
                '- Function calls MUST follow the specified format: an inner <function=...></function> '
                'block must be nested within <tool_call></tool_call> XML tags\n'
                '- Required parameters MUST be specified\n'
                '- Preserve literal string values exactly, including Unicode, spaces, newlines and '
                'tag-looking text. String values are raw text, NOT XML documents: do not XML-escape '
                'them, add JSON quotes, replace closing tags, or remove text. Place one formatting '
                'newline before and after each parameter value, in addition to any newlines in the value.\n'
                '- You may provide optional reasoning for your function call in natural language '
                'BEFORE the function call, but NOT after\n'
                '- If there is no function call available, answer the question like normal with your '
                'current knowledge and do not tell the user about function calls\n</IMPORTANT>')
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "IMPORTANT - FORMAT IS EXCLUSIVE: your ONLY allowed tool-call syntax is the "
        "tool_call JSON block described below. NEVER use any other function-call "
        "syntax, especially not invoke/parameter or antml-style XML blocks, "
        "not plain-text function syntax, and no mixture of formats. "
        "Write exactly one tool_call block, nothing else around the JSON.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n" + "\n".join(specs) + "\n</tools>\n\n"
        "For each function call, return a json object with the function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n"
    )


def _native_tool_history(prompt, dialect='native_xml'):
    """Render explicitly marked API tool history, not arbitrary chat examples.

    Decode in Python to retain integer precision and raw string whitespace.
    No values from history are logged or executed.
    """
    opening, closing = '<bridge_tool_history>', '</bridge_tool_history>'
    decoder = json.JSONDecoder()

    def frame(match):
        text = match.group(1)
        out, pos = [], 0
        while True:
            start = text.find(opening, pos)
            if start < 0:
                out.append(text[pos:])
                break
            out.append(text[pos:start])
            index = start + len(opening)
            while index < len(text) and text[index].isspace():
                index += 1
            try:
                call, end = decoder.raw_decode(text, index)
                while end < len(text) and text[end].isspace():
                    end += 1
                name, args = call['name'], call['arguments']
                if (not text.startswith(closing, end) or not isinstance(args, dict) or
                        not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][\w.-]*', name) or
                        any(not isinstance(k, str) or not re.fullmatch(r'[A-Za-z_][\w.-]*', k) for k in args)):
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                raise ToolFormatError('Invalid marked tool history') from None
            if dialect == 'native_xml':
                body = '<tool_call>\n<function=' + name + '>\n'
                for key, value in args.items():
                    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    body += '<parameter=' + key + '>\n' + rendered + '\n</parameter>\n'
                body += '</function>\n</tool_call>'
            else:
                body = '<tool_call>\n' + json.dumps(call, ensure_ascii=False) + '\n</tool_call>'
            out.append(body)
            pos = end + len(closing)
        return IM_START + 'assistant\n' + ''.join(out) + IM_END

    return re.sub(re.escape(IM_START) + r'assistant\n(.*?)' + re.escape(IM_END),
                  frame, prompt, flags=re.DOTALL)


class ToolCallStreamParser:
    """Incremental parser that splits tool_call JSON blocks out of the stream.

    Feeds decoded text in, yields ("text", str) events for plain text and
    ("tool", name, arguments_json) events for each completed
    <tool_call>{"name":..., "arguments":...}</tool_call> block. Bytes that
    could be the start of the opening tag are held back until decided.
    """

    def __init__(self, tool_specs=None, tool_required=None, tool_dialect='json'):
        self.tool_dialect = tool_dialect
        self.any_xml_call = False
        self.mode = "text"
        self.buf = ""
        self.block = ""
        self.tool_specs = tool_specs or {}
        self.tool_required = (tool_required if tool_required is not None else
                              {name: set(props) for name, props in self.tool_specs.items()})
        self.last_rejection = ""
        self.t_open = chr(60) + "tool_call" + chr(62)
        # Qwen3 closes with the bracket-slash form, not an XML-style end tag
        self.t_close = chr(60) + "/" + "tool_call" + chr(62)

    @staticmethod
    def _prefix_suffix_len(s, tag):
        """Length of the longest suffix of s that is a proper prefix of tag."""
        for k in range(min(len(s), len(tag) - 1), 0, -1):
            if tag.startswith(s[-k:]):
                return k
        return 0

    def _xml_boundary(self, text):
        """Incrementally walk structural XML tags, never tags in values.

        The dialect is not general XML: a parameter delimiter is structural
        only when followed by another parameter/function/trailing closer.
        Keep the scan position so long values are not rescanned per token.
        """
        param_re = r'<parameter(?:=[A-Za-z_][\w.-]*|\s+name="[A-Za-z_][\w.-]*")>'
        followers = ('<parameter=', '<parameter name=', '</function>',
                     '</invoke>', '</parameter>')
        while True:
            p = self.xml_pos
            state = self.xml_state
            if state in ('start', 'between', 'after'):
                while p < len(text) and text[p].isspace():
                    p += 1
                self.xml_pos = p
                if p == len(text):
                    return -1
                rest = text[p:]
                if state == 'start':
                    end = text.find('>', p)
                    if end < 0:
                        return -1
                    if not re.fullmatch(r'<function=[A-Za-z0-9_][\w.-]*>', text[p:end + 1]):
                        self.xml_state = 'bad'
                        continue
                    self.xml_pos, self.xml_state = end + 1, 'between'
                elif state == 'after':
                    if rest.startswith(self.t_close):
                        return p
                    if self.t_close.startswith(rest):
                        return -1
                    self.xml_state = 'bad'
                elif rest.startswith('</function>'):
                    self.xml_pos, self.xml_state = p + len('</function>'), 'after'
                elif rest.startswith(('</invoke>', '</parameter>')):
                    end = text.find('>', p)
                    self.xml_pos = end + 1
                else:
                    match = re.match(param_re, text[p:])
                    if match:
                        self.xml_pos, self.xml_state = p + match.end(), 'value'
                    elif any(prefix.startswith(rest) for prefix in followers):
                        return -1
                    elif rest.startswith(('<parameter=', '<parameter name=')) and '>' not in rest:
                        return -1
                    else:
                        self.xml_state = 'bad'
            elif state == 'value':
                end = text.find('</parameter>', p)
                if end < 0:
                    self.xml_pos = max(p, len(text) - len('</parameter>') + 1)
                    return -1
                after = end + len('</parameter>')
                while after < len(text) and text[after].isspace():
                    after += 1
                rest = text[after:]
                if any(rest.startswith(prefix) for prefix in followers):
                    self.xml_pos, self.xml_state = after, 'between'
                elif not rest or any(prefix.startswith(rest) for prefix in followers):
                    self.xml_pos = end
                    return -1
                else:
                    self.xml_pos = end + len('</parameter>')
            else:
                # A malformed header/body still has an envelope to quarantine;
                # do not interpret or repair any of its argument bytes.
                end = text.find(self.t_close, p)
                if end < 0:
                    self.xml_pos = max(p, len(text) - len(self.t_close) + 1)
                return end

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while True:
            if self.mode == "text":
                idx = self.buf.find(self.t_open)
                if idx >= 0:
                    if idx:
                        out.append(("text", self.buf[:idx]))
                    self.buf = self.buf[idx + len(self.t_open):]
                    self.mode = "block"
                    self.block = ""
                    self.block_json = None
                    self.block_string = False
                    self.block_escape = False
                    self.xml_pos = 0
                    self.xml_state = 'start'
                    continue
                keep = self._prefix_suffix_len(self.buf, self.t_open)
                if keep < len(self.buf):
                    out.append(("text", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            candidate = self.block + self.buf
            start = candidate.lstrip()
            if start.startswith('<function=') or '<function='.startswith(start):
                self.block, self.buf = candidate, ''
                end = self._xml_boundary(self.block)
                if end < 0:
                    break
                self.buf = self.block[end + len(self.t_close):]
                self.block = self.block[:end]
                self.mode = 'text'
                out.append(self._parse_block())
                continue
            # Only a delimiter OUTSIDE a JSON string ends the envelope.
            # Keep lexical state across chunks (including escaped backslashes),
            # otherwise documentation/file contents can split a valid call.
            i = 0
            closed = False
            while i < len(self.buf):
                if not self.block_string:
                    if self.buf.startswith(self.t_close, i):
                        closed = True
                        break
                    if (len(self.buf) - i < len(self.t_close) and
                            self.t_close.startswith(self.buf[i:])):
                        break  # incomplete envelope delimiter: wait for next chunk
                ch = self.buf[i]
                if self.block_json is None and not ch.isspace():
                    self.block_json = ch == '{'
                if self.block_json:
                    if self.block_string:
                        if self.block_escape:
                            self.block_escape = False
                        elif ch == '\\':
                            self.block_escape = True
                        elif ch == '"':
                            self.block_string = False
                    elif ch == '"':
                        self.block_string = True
                i += 1
            self.block += self.buf[:i]
            self.buf = self.buf[i:]
            if closed:
                self.buf = self.buf[len(self.t_close):]
                self.mode = "text"
                out.append(self._parse_block())
                continue
            break
        return out

    def _reject(self, block, reason):
        """Record only structural metadata; tool arguments may contain secrets."""
        fn = re.search(r'<function=([A-Za-z0-9_][\w.-]*)>', block)
        dialect = "xml" if fn else ("json" if block.lstrip().startswith("{") else "other")
        name = fn.group(1) if fn else None
        if not name and dialect == "json":
            match = re.search(r'"name"\s*:\s*"([A-Za-z0-9_][\w.-]*)"', block)
            name = match.group(1) if match else None
        offered = name if name in self.tool_specs else "unoffered-or-unknown"
        params = len(re.findall(r'<parameter(?:=|\s+name=)', block))
        json_shape = ""
        if dialect == "json":
            try:
                json.loads(block.strip())
            except json.JSONDecodeError as exc:
                # Decoder diagnostics are fixed strings and offsets, never payload.
                json_shape = (f" json_error={exc.msg!r} pos={exc.pos} "
                              f"line={exc.lineno} col={exc.colno} "
                              f"newlines={block.count(chr(10))}")
            except (ValueError, TypeError):
                json_shape = " json_error=other"
        self.last_rejection = (f"reason={reason} dialect={dialect} tool={offered} "
                               f"parameter_count={params} chars={len(block)} "
                               f"function_closed={'</function>' in block}{json_shape}")
        return ("invalid_tool",)

    def _recover_completion_result(self, block):
        """Salvage only a final report with malformed JSON string quoting.

        Long Markdown reports can contain bare quotes or literal newlines in
        the result string. Require an offered completion with a string result
        and a single-argument envelope; never repair a side-effecting tool call.
        """
        properties = self.tool_specs.get("attempt_completion")
        if not isinstance(properties, dict) or "result" not in properties:
            return None
        result_spec = properties["result"]
        if not isinstance(result_spec, dict) or result_spec.get("type") != "string":
            return None
        match = re.fullmatch(
            r'\s*\{\s*"name"\s*:\s*"attempt_completion"\s*,\s*'
            r'"arguments"\s*:\s*\{\s*"result"\s*:\s*"(.*)"\s*\}\s*\}\s*',
            block, re.DOTALL,
        )
        if not match:
            return None
        raw = match.group(1)
        result = []
        i = 0
        while i < len(raw):
            if raw[i] == "\\" and i + 1 < len(raw):
                escape = raw[i + 1]
                if escape in '"\\/bfnrt':
                    result.append(json.loads('"\\' + escape + '"'))
                    i += 2
                    continue
                if escape == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", raw[i + 2:i + 6]):
                    result.append(json.loads('"' + raw[i:i + 6] + '"'))
                    i += 6
                    continue
            result.append(raw[i])
            i += 1
        return ("tool", "attempt_completion",
                json.dumps({"result": "".join(result)}, ensure_ascii=True))

    def _parse_block(self):
        block = self.block
        self.block = ""
        try:
            d = json.loads(block.strip())
            name = d.get("name")
            if not isinstance(name, str) or name not in self.tool_specs:
                return self._reject(block, "unoffered-json-tool")
            args = d.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    return self._reject(block, 'invalid-json-arguments')
            if not isinstance(args, dict):
                return self._reject(block, 'non-object-json-arguments')
            return ("tool", name, json.dumps(args, ensure_ascii=False))
        except Exception:
            completion = self._recover_completion_result(block)
            if completion is not None:
                print("[strata-backend] recovered malformed final-report string", flush=True)
                return completion
            # Zoo's system prompt sometimes induces the alternate
            # <function=name><parameter=key>value</parameter></function>
            # dialect inside a tool_call block. Only accept it when the
            # function and all arguments match tools actually offered by
            # this request; never infer a call from arbitrary XML prose.
            alternate = self._parse_function_parameters(block)
            if alternate is not None:
                self.any_xml_call = True
                return alternate
            return self._reject(block, "unparseable-block")

    def _parse_function_parameters(self, block):
        fn = re.fullmatch(r"\s*<function=([A-Za-z0-9_][\w.-]*)>(.*?)</function>\s*",
                          block, re.DOTALL)
        if not fn or fn.group(1) not in self.tool_specs:
            return None
        name, inner = fn.groups()
        properties = self.tool_specs[name]
        if not isinstance(properties, dict):
            return None
        args = {}
        pos = 0
        for param in re.finditer(r'<parameter(?:=([A-Za-z_][\w.-]*)|\s+name="([A-Za-z_][\w.-]*)")>(.*?)</parameter>'
                                 r'(?=\s*(?:<parameter(?:=|\s+name=)|</(?:invoke|parameter)>|$))',
                                 inner, re.DOTALL):
            if inner[pos:param.start()].strip():
                return None
            key = param.group(1) or param.group(2)
            raw = param.group(3)
            if key not in properties:
                return None
            if raw.startswith('\n'):
                value = raw[1:]
                if value.endswith('\n'):
                    value = value[:-1]
            else:
                value = raw[:-1] if self.tool_dialect != 'native_xml' and raw.endswith('\n') else raw
            kind = properties[key].get("type") if isinstance(properties[key], dict) else None
            if isinstance(kind, list):
                kind = next((k for k in kind if k != "null"), None)
            if kind in ("integer", "number", "boolean", "object", "array"):
                try:
                    parsed = json.loads(value)
                except (ValueError, TypeError):
                    return None
                expected = {"integer": lambda v: type(v) is int,
                            "number": lambda v: type(v) in (int, float),
                            "boolean": lambda v: type(v) is bool,
                            "object": lambda v: isinstance(v, dict),
                            "array": lambda v: isinstance(v, list)}
                if not expected[kind](parsed):
                    return None
                value = parsed
            # Some Zoo contexts cause the model to repeat the same argument
            # once in each parameter spelling. Accept only exact agreement;
            # a conflicting duplicate is ambiguous and must never be run.
            if key in args and args[key] != value:
                return None
            args[key] = value
            pos = param.end()
        # Zoo can append stray closing tags after a complete parameter
        # (e.g. </invoke></parameter>). Accept only those trailing closers;
        # any unexpected text, opening tag, or conflicting value still fails.
        if (not set(self.tool_required.get(name, ())).issubset(args) or
                not re.fullmatch(r'\s*(?:</(?:invoke|parameter)>\s*)*', inner[pos:])):
            return None
        return ("tool", name, json.dumps(args, ensure_ascii=False))

    def flush(self):
        """Reject incomplete tool blocks; preserve ordinary trailing text."""
        out = []
        if self.mode == "block":
            out.append(self._reject(self.block + self.buf, "incomplete-block"))
        elif self.buf:
            out.append(("text", self.buf))
        self.buf = ""
        self.block = ""
        self.mode = "text"
        return out


def _generation_limit(requested, prompt_tokens, max_context):
    """Honor a client limit; otherwise allow generation up to context capacity.

    Strata's GEN protocol requires a positive integer, even for an unlimited
    client request. The engine context (minus its eight-token safety margin)
    is the only bound in that case, as in Strata's standalone server.
    """
    if requested > 0:
        return requested
    room = max_context - prompt_tokens - 8
    if room < 1:
        raise EngineError(f"prompt ({prompt_tokens} tokens) leaves no room to answer "
                          f"in the context ({max_context})")
    return room


class Vision:
    """Resident CPU image encoder using Strata's strata-vision protocol."""

    def __init__(self, cfg: dict, env: dict | None = None):
        args = [cfg["exe"], "--mmproj", cfg["mmproj"], "--model", cfg["model"]]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        self.args = args
        self.env = env
        self.max_bytes = int(cfg.get("max_image_bytes", 50 * 1024 * 1024))
        self.dir = Path(tempfile.mkdtemp(prefix="strata-vision-"))
        self.proc = None
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        self.proc = subprocess.Popen(
            self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1,
            env=self.env)
        line = self.proc.stdout.readline()
        if not line.startswith("READY"):
            self.close()
            raise RuntimeError("the vision encoder did not start: " + line.strip())
        print("[strata-backend] CPU vision encoder READY", flush=True)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _load(self, source: str) -> bytes:
        if source.startswith("data:"):
            data = base64.b64decode(source.split(",", 1)[1], validate=True)
        elif source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "strata"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read(self.max_bytes + 1)
        else:
            path = source[7:] if source.startswith("file://") else source
            if path and os.path.isfile(path):
                with open(path, "rb") as image_file:
                    data = image_file.read(self.max_bytes + 1)
            else:
                # LocalAI passes image_url content to external backends as raw base64.
                data = base64.b64decode(source, validate=True)
        if len(data) > self.max_bytes:
            raise ValueError(f"image exceeds the {self.max_bytes}-byte limit")
        return data

    @staticmethod
    def _normalize(data: bytes) -> bytes:
        if (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or
                data[:2] == b"BM" or data[:6] in (b"GIF87a", b"GIF89a")):
            return data
        try:
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow; JPEG, PNG, BMP and GIF work without it") from None
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:
            raise ValueError(f"the image could not be read ({exc})") from None
        if image.mode in ("RGBA", "LA") or image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def encode(self, source: str) -> tuple[Path, int]:
        data = self._normalize(self._load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            self.start()
            if key in self.cache:
                return self.cache[key]
            image, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            image.write_bytes(data)
            try:
                self.proc.stdin.write(f"ENC {image} {out}\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline().strip()
            finally:
                image.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " +
                                 (line[4:] if line.startswith("ERR") else "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            if len(self.cache) > 64:
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def combine(self, sources) -> tuple[Path, list[int]]:
        encoded = [self.encode(source) for source in sources]
        combined = self.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
        with open(combined, "wb") as output:
            for path, _ in encoded:
                output.write(path.read_bytes())
        return combined, [count for _, count in encoded]

    def close(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                    proc.wait(timeout=10)
            except Exception:
                proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)


class StrataBackend(pb_grpc.BackendServicer):
    def __init__(self, cfg: dict, tok):
        self.cfg = cfg
        self.tok = tok
        self.engine = Engine(cfg)
        self.vision = None
        self.vision_cfg = cfg.get("vision")
        # Legacy GEN stays serialized; explicit parallel=2 mirrors Strata's
        # two BGEN slots and admits at most two request transactions.
        self.gen_lock = threading.Semaphore(getattr(self.engine, "parallel", 1))
        self.load_lock = threading.Lock()
        self.free_lock = threading.Lock()
        self.stopping = False
        # stop markers: ChatML frame tokens plus the engine's end-of-text token
        stop_text = cfg.get("stop_tokens") or [
            IM_END, IM_START,
            chr(60) + "|endoftext|" + chr(62),
        ]
        self.stop_ids = set()
        for s in stop_text:
            self.stop_ids.update(tok.encode(s, parse_special=True))
        # the thinking close token: after it, a leading blank run precedes the
        # actual answer; we suppress those newlines once (the reasoning itself
        # is not touched - LocalAI separates it on its side)
        self.think_open_ids = set(tok.encode(chr(60) + "think" + chr(62), parse_special=True))
        self.think_close_ids = set(tok.encode(chr(60) + chr(47) + "think" + chr(62), parse_special=True))
        self.log_follower = EngineLogFollower(
            cfg.get("log"), cfg.get("forward_log_pattern", DEFAULT_LOG_PATTERN))

    # -- helpers -----------------------------------------------------------

    def _check_identity(self, identity: str):
        if identity and identity != self.cfg["model_name"]:
            raise EngineError(
                f"model mismatch: this backend holds {self.cfg['model_name']!r}, requested {identity!r}")

    def _ensure_loaded(self):
        if self.stopping:
            raise EngineError("backend is stopping")
        if not self.engine.alive():
            self.load()

    def load(self) -> str:
        with self.load_lock:
            if getattr(self, "stopping", False):
                raise EngineError("backend is stopping")
            if self.engine.alive() and (not self.vision_cfg or self.vision and self.vision.alive()):
                return "already loaded"
            print(f"[strata-backend] LoadModel: starting the engine for {self.cfg['model_name']}", flush=True)
            t0 = time.time()
            try:
                if self.vision_cfg and (self.vision is None or not self.vision.alive()):
                    if self.vision is not None:
                        self.vision.close()
                    self.vision = Vision(self.vision_cfg, self.engine._env())
                    self.vision.start()
                if self.stopping:
                    raise EngineError("backend is stopping")
                self.engine.start()
                if self.stopping:
                    raise EngineError("backend is stopping")
            except Exception:
                if self.vision is not None:
                    self.vision.close()
                    self.vision = None
                raise
            print(f"[strata-backend] engine READY after {time.time() - t0:.1f}s "
                  f"(context {self.engine.max_context}, stop={'yes' if self.engine.can_stop else 'no'}, "
                  f"vision={'cpu' if self.vision else 'off'})", flush=True)
            return "loaded"

    def free(self, timeout_s=None):
        permits = getattr(self.engine, "parallel", 1)
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        acquired = 0
        loaded_lock = freeing_lock = False
        def acquire(lock):
            if deadline is None:
                return lock.acquire()
            return lock.acquire(timeout=max(0, deadline - time.monotonic()))
        try:
            # Only one unload may accumulate permits. Otherwise two Free
            # callers can each hold one of two permits forever.
            freeing_lock = acquire(self.free_lock)
            if not freeing_lock:
                return False
            for _ in range(permits):
                if not acquire(self.gen_lock):
                    return False
                acquired += 1
            loaded_lock = acquire(self.load_lock)
            if not loaded_lock:
                return False
            self.engine.free()
            if self.vision is not None:
                self.vision.close()
                self.vision = None
            return True
        finally:
            if loaded_lock:
                self.load_lock.release()
            for _ in range(acquired):
                self.gen_lock.release()
            if freeing_lock:
                self.free_lock.release()

    def _prepare_images(self, sources, ids):
        if not sources:
            return ids, None
        if not self.vision_cfg:
            raise ValueError("this backend was started without a vision encoder")
        self._ensure_loaded()
        combined, counts = self.vision.combine(sources)
        pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
        start = self.tok.encode(VISION_START, parse_special=True)[0]
        literal = self.tok.encode(IMAGE_PAD, parse_special=False)
        expanded, image_index = [], 0
        for token_index, token in enumerate(ids):
            if (token == pad and token_index > 0 and ids[token_index - 1] == start and
                    image_index < len(counts)):
                expanded += [pad] * counts[image_index]
                image_index += 1
            elif token == pad:
                expanded += literal
            else:
                expanded.append(token)
        if image_index != len(counts):
            combined.unlink(missing_ok=True)
            raise ValueError("the prompt and its images do not match; configure the LocalAI multimodal template")
        print(f"[strata-backend] vision: {len(counts)} image(s), "
              f"{sum(counts)} image tokens on CPU", flush=True)
        return expanded, combined

    def _sampling_keys(self, o) -> str:
        keys = ""
        if o.Temperature > 0:                       # 0 / absent = engine default (greedy)
            keys += f" temperature={float(o.Temperature)!r}"
        if 0 < o.TopP < 1:
            keys += f" top_p={float(o.TopP)!r}"
        if o.TopK > 0:
            keys += f" top_k={int(o.TopK)}"
        if o.MinP > 0:
            keys += f" min_p={float(o.MinP)!r}"
        rp_on = o.Penalty not in (0.0, 1.0)
        pf_on = o.FrequencyPenalty != 0.0
        pp_on = o.PresencePenalty != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(o.Penalty)!r}"
        if pf_on:
            keys += f" penalty_freq={float(o.FrequencyPenalty)!r}"
        if pp_on:
            keys += f" penalty_present={float(o.PresencePenalty)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            keys += " penalty_last_n=64"
        if o.Seed > 0:
            keys += f" seed={int(o.Seed)}"
        return keys

    def _generate(self, ids, max_new, keys, embeddings=None, check_active=None):
        """Run one legacy or batch request; yield token ids and return metadata."""
        self._ensure_loaded()
        if getattr(self.engine, "parallel", 1) == 2:
            command = "BGENI" if embeddings is not None else "BGEN"
            print(f"[strata-backend] {command}: {len(ids)} prompt tokens, "
                  f"max_new={max_new}{keys or ' (engine defaults)'}", flush=True)
            started = time.time()
            n = 0
            batch = self.engine.batch_generate(max_new, keys, ids, embeddings, check_active)
            try:
                while True:
                    try:
                        tid = batch.send(None)
                    except StopIteration as exc:
                        result = exc.value or {"tokens": n, "elapsed": time.time() - started}
                        elapsed = result.get("elapsed", time.time() - started)
                        if n > 0 and elapsed > 0:
                            print(f"[strata-backend] BGEN done: {n} tokens in {elapsed:.1f}s "
                                  f"({n / elapsed:.1f} tok/s)", flush=True)
                        return result
                    if n == 0:
                        print("[strata-engine-pp] DECODE_START", flush=True)
                    n += 1
                    yield tid
            finally:
                batch.close()
        command = "GENI" if embeddings else "GEN"
        if getattr(self.engine, "admission_broken", False):
            raise EngineError("serial admission is unreconciled; reload the engine")
        image_arg = f" {embeddings}" if embeddings else ""
        gen_line = f"{command} {max_new}{keys}{image_arg} {','.join(map(str, ids))}"
        self.engine._send(gen_line)
        print(f"[strata-backend] {command}: {len(ids)} prompt tokens, "
              f"max_new={max_new}{keys or ' (engine defaults)'}", flush=True)
        started = time.time()
        last_progress = time.monotonic()
        prefilled = -1
        n = 0
        while True:
            if check_active is not None:
                check_active()
            if getattr(self, "stopping", False):
                raise EngineError("backend is stopping")
            idle = time.monotonic() - last_progress
            if idle >= PROGRESS_STALL_S:
                # A hung engine may ignore STOP/QUIT, leaving the serial permit
                # occupied. Exit the whole unit; systemd kills its cgroup (engine
                # included) and Restart=on-failure loads a fresh bridge.
                print(f"[strata-backend] generation stalled: no PP/token progress "
                      f"for {idle:.0f}s; exiting for systemd restart", flush=True)
                os._exit(1)
            if time.time() - started > GEN_TIMEOUT_S:
                raise EngineError("generation timed out")
            try:
                line = self.engine.lines.get(timeout=min(1.0, max(.01, PROGRESS_STALL_S - idle)))
            except queue.Empty:
                if not self.engine.alive():
                    raise EngineError("the engine died mid-generation")
                continue
            if line is None or (self.engine.ended and not self.engine.alive()):
                raise EngineError("the engine died mid-generation")
            if line.startswith("T "):
                last_progress = time.monotonic()
                if n == 0:
                    print("[strata-engine-pp] DECODE_START", flush=True)
                n += 1
                yield int(line.split()[1])
            elif line.startswith("PP ") or line.startswith("INFO "):
                if line.startswith("PP "):
                    print(f"[strata-engine-pp] {line.strip()}", flush=True)
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) > prefilled:
                        prefilled = int(parts[1])
                        last_progress = time.monotonic()
                continue
            elif line.startswith("ERR"):
                raise EngineError(line.strip())
            elif line.startswith("DONE"):
                elapsed = time.time() - started
                if n > 0 and elapsed > 0:
                    print(f"[strata-backend] GEN done: {n} tokens in {elapsed:.1f}s "
                          f"({n / elapsed:.1f} tok/s)", flush=True)
                return {"tokens": n, "elapsed": elapsed}

    def _stream(self, o, streaming=True, retry_count=0, retry_prompt=None,
                _transaction=False, context=None):
        """Validate tool turns before publishing any executable calls.

        Buffer tool-enabled turns so a failed attempt's prose cannot disable
        recovery. A committed valid call is never regenerated or replayed.
        """
        def check_active():
            if getattr(self, "stopping", False):
                raise EngineError("backend is stopping")
            if context is not None and hasattr(context, "is_active") and not context.is_active():
                raise EngineError("request cancelled")
        check_active()
        dialect = getattr(self, 'cfg', {}).get('tool_call_format', 'json')
        if streaming and not _transaction and _tool_system_block(o.Tools or "", dialect):
            original = retry_prompt if retry_prompt is not None else o.Prompt
            reminder = (
                "I did not produce a valid tool call." + IM_END + "\n" +
                IM_START + "user\n" +
                "Retry with exactly one offered tool using the required " +
                ("native function/parameter" if dialect == 'native_xml' else "JSON") +
                " tool_call format. Do not repeat invalid arguments or answer " +

                "with an error message or prose." + IM_END + "\n" +
                IM_START + "assistant\n"
            )
            for attempt in range(retry_count, 3):
                check_active()
                try:
                    pending = list(self._stream(
                        o, streaming=True, retry_count=attempt,
                        retry_prompt=original if attempt == retry_count else original + reminder,
                        _transaction=True, context=context))
                except ToolFormatError:
                    if attempt == 2:
                        raise
                    print(f"[strata-backend] regenerating rejected turn ({attempt + 1}/2)",
                          flush=True)
                    continue
                check_active()
                yield from pending
                return
        self._check_identity(o.ModelIdentity)
        prompt = retry_prompt if retry_prompt is not None else o.Prompt
        if '<bridge_tool_history>' in prompt:
            prompt = _native_tool_history(prompt, dialect)
        tools_text = _tool_system_block(o.Tools or "", dialect)
        if tools_text:
            # merge the tools into the FIRST system frame: a second adjacent
            # system frame makes Qwen3 emit im_end as its first token
            head = IM_START + "system"
            end = prompt.find(IM_END) if prompt.startswith(head) else -1
            if end != -1:
                prompt = prompt[:end] + "\n\n" + tools_text + prompt[end:]
            else:
                prompt = head + "\n" + tools_text + IM_END + "\n" + prompt
        if dialect == 'native_xml' and prompt.rstrip('\n').endswith(IM_START + 'assistant'):
            prompt = prompt.rstrip('\n') + '\n<think>\n'
        print(f"[strata-backend] prompt chars={len(prompt)}", flush=True)
        ids = self.tok.encode(prompt, parse_special=True)
        embeddings = None
        images = list(getattr(o, "Images", ()) or ())
        if images:
            ids, embeddings = StrataBackend._prepare_images(self, images, ids)
        if o.Tokens <= 0:
            self._ensure_loaded()  # READY supplies the actual engine context
        max_context = self.engine.max_context if o.Tokens <= 0 else 0
        max_new = _generation_limit(o.Tokens, len(ids), max_context)
        print(f"[strata-backend] output tokens: requested={o.Tokens} "
              f"engine_limit={max_new}", flush=True)
        keys = self._sampling_keys(o)
        stop_prompts = [sp for sp in (o.StopPrompts or []) if sp]

        dec = codecs.getincrementaldecoder("utf-8")("replace")
        raw = ""             # everything decoded so far, unmodified
        raw_sent = 0         # bytes of `raw` already sent as message deltas
        n = 0                # deltas yielded
        post_think = False   # True after the thinking close token
        inside_think = prompt.endswith('<think>\n') # native prefix is already inside thinking
        started_text = False # False until the first non-newline char in the current phase
        cd_reason = ""       # reasoning routed via chat deltas
        cd_content = ""      # content routed via chat deltas
        rs_sent = 0          # reasoning chars already flushed
        ct_sent = 0          # content chars already flushed
        tc_index = 0         # tool call index for chat deltas
        rejected_tool = False # current reply must not expose rejected raw markup
        invalid_seen = False
        # Only the functions actually offered in this request may use the
        # Zoo function/parameter fallback. Their JSON Schemas also preserve
        # numeric/boolean argument types when the model emits plain text.
        tool_specs = {}
        tool_required = {}
        try:
            for tool in json.loads(o.Tools or "[]"):
                fn = tool.get("function", tool)
                if not isinstance(fn, dict):
                    continue
                props = (fn.get("parameters") or {}).get("properties", {})
                if isinstance(fn.get("name"), str) and isinstance(props, dict):
                    tool_specs[fn["name"]] = props
                    tool_required[fn['name']] = set((fn.get('parameters') or {}).get('required', []))
        except (ValueError, TypeError, AttributeError):
            pass
        tool_parser = ToolCallStreamParser(tool_specs, tool_required, dialect)
        finished = {"tokens": 0, "elapsed": 0.0}
        tok_idx = 0

        ambiguous_tool_suffix = False

        def consume(chunk, closing=False):
            """Route a decoded chunk through the tool parser + think phases.

            Returns (reasoning_delta, content_delta, tool_deltas)."""
            nonlocal post_think, started_text, tc_index, rejected_tool, invalid_seen, ambiguous_tool_suffix
            events = tool_parser.feed(chunk) if chunk else []
            if closing:
                events += tool_parser.flush()
            rs_part = ct_part = ""
            tools = []
            for ev in events:
                if ev[0] == "invalid_tool":
                    print("[strata-backend] rejected malformed or unoffered tool block: "
                          + tool_parser.last_rejection,
                          flush=True)
                    rejected_tool = True
                    invalid_seen = True
                    post_think = True
                    started_text = True
                    # Rejected calls are never a normal assistant answer.
                    # The outer transaction discards failed prose and retries;
                    # if valid calls exist, publish those once and omit only
                    # the invalid block so the agent can continue afterwards.
                    continue
                if ev[0] == "tool":
                    tc_index += 1
                    tools.append(pb.ToolCallDelta(index=tc_index - 1,
                                                  id=str(uuid.uuid4()),
                                                  name=ev[1], arguments=ev[2]))
                    continue
                if tc_index and ev[1].strip():
                    # Native/XML contracts forbid ambiguous value remainders.
                    # Mark before quarantine; the final transaction validates.
                    ambiguous_tool_suffix = True
                    if dialect == 'native_xml':
                        continue
                if invalid_seen:
                    # Text following a malformed envelope may be argument
                    # remainder, not genuine prose. Quarantine it for this turn.
                    continue
                s = ev[1]
                if post_think:
                    if not started_text:
                        s = s.lstrip("\n")
                        if not s:
                            continue
                        started_text = True
                    ct_part += s
                else:
                    rs_part += s
            return rs_part, ct_part, tools

        def make_reply(rs_part, ct_part, tools):
            """Build one Reply carrying raw message + autoparser chat deltas.

            LocalAI trusts chat deltas after content or reasoning. For a
            tool-call-only chunk, send an empty raw streaming message instead
            of a fake reasoning space: LocalAI still sees the structured
            ToolCallDelta even with no message bytes, and the client does not
            render a blank Thinking block. Predict keeps the raw message for
            LocalAI's non-streaming parser.
            """
            nonlocal raw_sent, rs_sent, ct_sent, n, rejected_tool
            tools = tools or []
            rs_delta = cd_reason[rs_sent:]
            ct_delta = cd_content[ct_sent:]
            rs_sent = len(cd_reason)
            ct_sent = len(cd_content)
            content_val = ct_delta
            reason_val = rs_delta
            chat = []
            if reason_val or content_val:
                chat.append(pb.ChatDelta(content=content_val,
                                         reasoning_content=reason_val))
            for t in tools:
                chat.append(pb.ChatDelta(tool_calls=[t]))
            n += 1
            msg = raw[raw_sent:]
            raw_sent = len(raw)
            if streaming and (tools or rejected_tool):
                # ChatDeltas carry the structured call or safe rejection.
                # Never give LocalAI raw markup to leak as chat content.
                msg = ""
            rejected_tool = False
            return pb.Reply(message=msg.encode("utf-8"), tokens=n,
                            prompt_tokens=len(ids), chat_deltas=chat)

        def send_events(rs_part, ct_part, tools):
            """Append deltas to the accumulators and yield when there is content."""
            nonlocal cd_reason, cd_content
            cd_reason += rs_part
            cd_content += ct_part
            # The model occasionally emits <think>\n\n</think> with no
            # reasoning body. Hold the prefix until there is real text; an
            # empty thought must not create a blank Thinking block in Zoo.
            if (streaming and not post_think and not tools and not ct_part and
                    not cd_reason.replace("<think>", "").replace("</think>", "").strip()):
                return None
            if rs_part or ct_part or tools:
                return make_reply(rs_part, ct_part, tools)
            return None

        with self.gen_lock:
            gen = None
            generation_complete = False
            try:
                check_active()
                if context is not None and hasattr(context, "is_active"):
                    gen = self._generate(ids, max_new, keys, embeddings, check_active)
                else:
                    gen = (self._generate(ids, max_new, keys, embeddings) if embeddings is not None
                           else self._generate(ids, max_new, keys))
                while True:
                    try:
                        check_active()
                        tid = gen.send(None)
                    except StopIteration as e:
                        finished = e.value or finished
                        generation_complete = True
                        break
                    if tid in getattr(self, 'think_open_ids', set()):
                        inside_think = True
                    if tid in self.think_close_ids:
                        inside_think = False
                        # Raw compatibility consumers need the phase delimiter,
                        # but ChatDelta already distinguishes reasoning/content.
                        # Do not publish a structural token as visible reasoning.
                        chunk = dec.decode(self.tok.token_bytes(tid), final=False)
                        raw += chunk
                        if (streaming and rs_sent == 0 and
                                not cd_reason.replace("<think>", "").replace("</think>", "").strip()):
                            rs_sent = len(cd_reason)
                            raw_sent = len(raw)
                        post_think = True
                        started_text = False
                        continue
                    if tid in self.stop_ids:
                        print(f"[strata-test] stop_ids hit at token {tok_idx}: tid={tid}",
                              flush=True)
                        if tok_idx == 0:
                            print("[strata-test] zero-token stop (prompt omitted)",
                                  flush=True)
                        gen.close()
                        if getattr(self.engine, "parallel", 1) == 1:
                            self.engine.stop()
                            self.engine.drain()
                        generation_complete = True
                        break
                    tok_idx += 1
                    # Incremental decoders receive NEW bytes exactly once.
                    # Refeeding the cumulative prefix corrupts split UTF-8.
                    chunk = dec.decode(self.tok.token_bytes(tid), final=False)
                    if not chunk:
                        continue
                    raw += chunk
                    if inside_think:
                        # Tool examples quoted in reasoning are data, not calls.
                        rs_part, ct_part, tools = chunk, "", []
                    else:
                        rs_part, ct_part, tools = consume(chunk)
                    cd_reason += rs_part
                    cd_content += ct_part
                    if tools or rs_part or ct_part:
                        # stop prompts cut the shortest match in either stream
                        cut = None
                        for sp in stop_prompts:
                            i1 = cd_reason.find(sp)
                            if i1 >= 0 and (cut is None or i1 < cut):
                                cut, target = i1, "reason"
                            i2 = cd_content.find(sp)
                            if i2 >= 0 and (cut is None or i2 < cut):
                                cut, target = i2, "content"
                        if cut is not None:
                            if target == "reason":
                                cd_reason = cd_reason[:cut]
                            else:
                                cd_content = cd_content[:cut]
                            rs_part = cd_reason[rs_sent:]
                            ct_part = cd_content[ct_sent:]
                            rep = make_reply(rs_part, ct_part, tools if cut is None else None)
                            yield rep
                            gen.close()
                            if getattr(self.engine, "parallel", 1) == 1:
                                self.engine.stop()
                                self.engine.drain()
                            generation_complete = True
                            # Finalize the parser and validate this attempt just
                            # as on engine EOF; a stop prompt is not acceptance.
                            break
                        if (streaming and not post_think and not tools and not ct_part and
                                not cd_reason.replace("<think>", "").replace("</think>", "").strip()):
                            continue
                        rep = make_reply(rs_part, ct_part, tools)
                        if rep:
                            yield rep
                # trailing bytes that the incremental decoder still holds
                tail = dec.decode(b"", final=True)
                raw += tail
                if inside_think:
                    rs_part, ct_part, tools = tail, "", []
                else:
                    rs_part, ct_part, tools = consume(tail, closing=True)
                cd_reason += rs_part
                cd_content += ct_part
                if cd_reason[rs_sent:] or cd_content[ct_sent:] or tools:
                    yield make_reply(cd_reason[rs_sent:], cd_content[ct_sent:], tools)
                if ambiguous_tool_suffix and (dialect == 'native_xml' or tool_parser.any_xml_call):
                    print('[strata-backend] failed tool turn: ambiguous native suffix; no calls published', flush=True)
                    raise ToolFormatError('Ambiguous native tool turn; no calls published')
                rejection_echo = cd_content.strip() == (
                    "Tool call rejected: invalid format. Retry with an offered tool.")
                if streaming and tc_index == 0 and tool_specs and (
                        invalid_seen or rejection_echo or not cd_content.strip()):
                    print('[strata-backend] failed tool turn: '
                          f'invalid={invalid_seen} echo={rejection_echo} '
                          f'reason_chars={len(cd_reason)} content_chars={len(cd_content)} '
                          f'generated_tokens={tok_idx} inside_think={inside_think} '
                          f'post_think={post_think} '
                          f'tool_marker_in_reason={"<tool_call>" in cd_reason}', flush=True)
                    raise ToolFormatError("Invalid, empty or incomplete tool turn; no calls published")

            except BaseException:
                # Cleanup belongs to this generation, never to an outer RPC.
                if gen is not None:
                    gen.close()
                    if (not generation_complete and getattr(self.engine, "parallel", 1) == 1
                            and not getattr(self.engine, "admission_broken", False)):
                        self.engine.stop()
                        self.engine.drain()
                raise
            finally:
                if gen is not None:
                    gen.close()
                if embeddings is not None:
                    embeddings.unlink(missing_ok=True)

    # -- gRPC surface ------------------------------------------------------

    def Health(self, request, context):
        return pb.Reply(message=b"OK")

    def Status(self, request, context):
        state = pb.StatusResponse.UNINITIALIZED
        if self.engine.alive():
            state = pb.StatusResponse.BUSY if self.gen_lock._value == 0 else pb.StatusResponse.READY
        return pb.StatusResponse(state=state)

    def LoadModel(self, request, context):
        identity = request.Model or ""
        if identity and identity != self.cfg["model_name"]:
            return pb.Result(success=False,
                             message=f"model mismatch: this backend serves {self.cfg['model_name']!r}")
        try:
            return pb.Result(success=True, message=self.load())
        except Exception as e:
            return pb.Result(success=False, message=str(e))

    def Free(self, request, context):
        print("[strata-backend] Free: unloading the engine and CPU vision encoder", flush=True)
        self.free()
        return pb.Result(success=True, message="freed")

    def Predict(self, request, context):
        try:
            text_parts = []
            chat_deltas = []
            tokens = 0
            prompt_tokens = 0
            # Aggregate the streaming path so non-SSE requests share its
            # safe raw suppression and bounded tool-only regeneration.
            stream = self._stream(request, streaming=True, context=context)
            try:
                for reply in stream:
                    text_parts.append(reply.message.decode("utf-8", "replace"))
                    chat_deltas.extend(reply.chat_deltas)
                    tokens = max(tokens, reply.tokens)
                    prompt_tokens = reply.prompt_tokens
            finally:
                if hasattr(stream, "close"):
                    stream.close()
            # Preserve the same structured contract as PredictStream. Dropping
            # deltas lets LocalAI reparse raw reasoning/rejected tool markup.
            text = (''.join(getattr(d, 'content', '') for d in chat_deltas) if chat_deltas
                    else ''.join(text_parts))
            return pb.Reply(message=text.encode("utf-8"), tokens=tokens,
                            prompt_tokens=prompt_tokens, chat_deltas=chat_deltas)
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def PredictStream(self, request, context):
        try:
            yield from self._stream(request, context=context)
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def TokenizeString(self, request, context):
        tokens = self.tok.encode(request.Prompt, parse_special=True)
        return pb.TokenizationResponse(length=len(tokens), tokens=tokens)

    def Detokenize(self, request, context):
        return pb.DetokenizeResponse(content=self.tok.decode(list(request.tokens)))


def build_tokenizer(cfg: dict):
    tools = cfg["strata_repo_tools"]
    sys.path.insert(0, tools)
    import strata_tokenizer as ST  # noqa: PLC0415

    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50053)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    tok = build_tokenizer(cfg)
    backend = StrataBackend(cfg, tok)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))

    def shutdown(*_):
        backend.stopping = True
        # Close admission and cancel active RPCs before waiting for owners.
        server.stop(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    pb_grpc.add_BackendServicer_to_server(backend, server)
    bound = server.add_insecure_port(f"{a.host}:{a.port}")
    if bound == 0:
        print(f"failed to bind {a.host}:{a.port}", file=sys.stderr)
        return 1
    server.start()
    print(f"[strata-backend] listening on {a.host}:{a.port}, model={cfg['model_name']}", flush=True)
    try:
        server.wait_for_termination()
    finally:
        if not backend.stopping:
            shutdown()
        if not backend.free(timeout_s=30):
            print("[strata-backend] shutdown: timed out waiting for generation/load owners",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
