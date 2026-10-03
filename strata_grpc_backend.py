#!/usr/bin/env python3
"""Strata LocalAI backend: a gRPC bridge between LocalAI and the Strata engine.

Implements LocalAI's backend.Backend proto. Model loading / unloading:

  LoadModel -> spawns the Strata engine (`strata --serve ...`) with the args
              from the JSON config; waits for the engine's READY line.
  Predict / PredictStream -> tokenizes the prompt with the pack's tokenizer,
              writes a `GEN <max_new> <sampling keys> <ids>` line to the
              engine's stdin, streams `T <id>` lines back out as Reply chunks.
  Free     -> sends QUIT to the engine and waits for it to exit, releasing
              all GPU/RAM so LocalAI can load other models.

The tokenizer comes from the Strata repo's tools/strata_tokenizer.py and the
pack's vocab.json / merges.txt / token_type.json.
"""

import argparse
import codecs
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
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

READY_TIMEOUT_S = 1200   # engine load takes minutes (experts + PLE)
GEN_TIMEOUT_S = 3600     # per-request hard ceiling


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
        self.log_path = cfg.get("log")
        self.proc = None
        self.lines = None
        self.pump = None
        self.ended = True
        self.max_context = 0
        self.can_stop = False
        self.info = {}
        self.log = None

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
        self.log = open(self.log_path, "a", encoding="utf-8") if self.log_path else subprocess.DEVNULL
        args = [self.cfg["exe"], "--serve", *self.cfg["args"]]
        self.proc = subprocess.Popen(
            args, cwd=self.cfg.get("cwd") or ".", stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8",
            bufsize=1, env=self._env())
        self.lines = queue.Queue()
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
                return
        raise EngineError(f"the engine did not become READY within {READY_TIMEOUT_S}s")

    def _pump(self):
        proc, q = self.proc, self.lines
        for line in proc.stdout:
            q.put(line)
        if self.proc is proc:
            self.ended = True
        q.put(None)

    def _send(self, line: str):
        if not self.alive():
            raise EngineError("the engine is not running")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def stop(self):
        """Ask the engine to abort the current request mid-generation."""
        if self.alive() and self.can_stop:
            try:
                self._send("STOP")
            except Exception:
                pass

    def drain(self, timeout_s: float = 120.0):
        """Discard leftovers of an aborted generation.

        After STOP the engine still emits its remaining `T` lines and the
        final `DONE`. Leaving them in the queue would make the NEXT request
        read the previous conversation's tokens (lost step), so every abort
        path drains until DONE (or a bounded timeout) before returning.
        """
        if self.lines is None:
            return
        deadline = time.time() + timeout_s
        n = 0
        idle = 0
        while time.time() < deadline:
            if not self.alive():
                break
            try:
                line = self.lines.get(timeout=1.0)
                idle = 0
            except queue.Empty:
                # nothing pending for a moment; a few idle seconds mean the
                # engine has nothing queued anymore (a missing DONE is fine)
                idle += 1
                if idle >= 3:
                    break
                continue
            if line is None:
                break
            if line.startswith("DONE") or line.startswith("ERR"):
                n += 1
                break
            n += 1
        if n:
            print(f"[strata-backend] drain: discarded {n} leftover engine lines", flush=True)

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


def _tool_system_block(tools_json: str) -> str:
    """Build the Qwen3-native system block describing available tools.

    LocalAI does not inject tool definitions for custom chat templates, so the
    bridge prepends them itself. The model is trained to answer with
    <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks.
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


class ToolCallStreamParser:
    """Incremental parser that splits tool_call JSON blocks out of the stream.

    Feeds decoded text in, yields ("text", str) events for plain text and
    ("tool", name, arguments_json) events for each completed
    <tool_call>{"name":..., "arguments":...}</tool_call> block. Bytes that
    could be the start of the opening tag are held back until decided.
    """

    def __init__(self, tool_specs=None):
        self.mode = "text"
        self.buf = ""
        self.block = ""
        self.tool_specs = tool_specs or {}
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
                    continue
                keep = self._prefix_suffix_len(self.buf, self.t_open)
                if keep < len(self.buf):
                    out.append(("text", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            end = self.buf.find(self.t_close)
            if end >= 0:
                self.block += self.buf[:end]
                self.buf = self.buf[end + len(self.t_close):]
                self.mode = "text"
                out.append(self._parse_block())
                continue
            keep = self._prefix_suffix_len(self.buf, self.t_close)
            if keep < len(self.buf):
                self.block += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
            break
        return out

    def _reject(self, block, reason):
        """Record only structural metadata; tool arguments may contain secrets."""
        fn = re.search(r'<function=([A-Za-z_][\w.-]*)>', block)
        dialect = "xml" if fn else ("json" if block.lstrip().startswith("{") else "other")
        name = fn.group(1) if fn else None
        if not name and dialect == "json":
            match = re.search(r'"name"\s*:\s*"([A-Za-z_][\w.-]*)"', block)
            name = match.group(1) if match else None
        offered = name if name in self.tool_specs else "unoffered-or-unknown"
        params = len(re.findall(r'<parameter(?:=|\s+name=)', block))
        self.last_rejection = (f"reason={reason} dialect={dialect} tool={offered} "
                               f"parameter_count={params} chars={len(block)} "
                               f"function_closed={'</function>' in block}")
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
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            return ("tool", name, args)
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
                return alternate
            return self._reject(block, "unparseable-block")

    def _parse_function_parameters(self, block):
        fn = re.fullmatch(r"\s*<function=([A-Za-z_][\w.-]*)>(.*?)</function>\s*",
                          block, re.DOTALL)
        if not fn or fn.group(1) not in self.tool_specs:
            return None
        name, inner = fn.groups()
        properties = self.tool_specs[name]
        if not isinstance(properties, dict):
            return None
        args = {}
        pos = 0
        for param in re.finditer(r'<parameter(?:=([A-Za-z_][\w.-]*)|\s+name="([A-Za-z_][\w.-]*)")>(.*?)</parameter>',
                                 inner, re.DOTALL):
            if inner[pos:param.start()].strip():
                return None
            key = param.group(1) or param.group(2)
            raw = param.group(3)
            if key not in properties:
                return None
            value = raw.strip()
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
        if not args or not re.fullmatch(r'\s*(?:</(?:invoke|parameter)>\s*)*', inner[pos:]):
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


class StrataBackend(pb_grpc.BackendServicer):
    def __init__(self, cfg: dict, tok):
        self.cfg = cfg
        self.tok = tok
        self.engine = Engine(cfg)
        self.gen_lock = threading.Semaphore(1)   # one generation at a time
        self.load_lock = threading.Lock()
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
        self.think_close_ids = set(tok.encode(chr(60) + chr(47) + "think" + chr(62), parse_special=True))
        self.log_follower = EngineLogFollower(
            cfg.get("log"), cfg.get("forward_log_pattern", DEFAULT_LOG_PATTERN))

    # -- helpers -----------------------------------------------------------

    def _check_identity(self, identity: str):
        if identity and identity != self.cfg["model_name"]:
            raise EngineError(
                f"model mismatch: this backend holds {self.cfg['model_name']!r}, requested {identity!r}")

    def _ensure_loaded(self):
        if not self.engine.alive():
            self.load()

    def load(self) -> str:
        with self.load_lock:
            if self.engine.alive():
                return "already loaded"
            print(f"[strata-backend] LoadModel: starting the engine for {self.cfg['model_name']}", flush=True)
            t0 = time.time()
            self.engine.start()
            print(f"[strata-backend] engine READY after {time.time() - t0:.1f}s "
                  f"(context {self.engine.max_context}, stop={'yes' if self.engine.can_stop else 'no'})",
                  flush=True)
            return "loaded"

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

    def _generate(self, ids, max_new, keys):
        """Run one GEN request; yields token ids; returns metadata at the end."""
        self._ensure_loaded()
        gen_line = f"GEN {max_new}{keys} {' '.join(map(str, ids))}"
        self.engine._send(gen_line)
        print(f"[strata-backend] GEN: {len(ids)} prompt tokens, max_new={max_new}{keys or ' (engine defaults)'}",
              flush=True)
        started = time.time()
        n = 0
        while True:
            if time.time() - started > GEN_TIMEOUT_S:
                self.engine.stop()
                self.engine.drain()
                raise EngineError("generation timed out")
            try:
                line = self.engine.lines.get(timeout=1.0)
            except queue.Empty:
                if not self.engine.alive():
                    raise EngineError("the engine died mid-generation")
                continue
            if line is None or (self.engine.ended and not self.engine.alive()):
                raise EngineError("the engine died mid-generation")
            if line.startswith("T "):
                if n == 0:
                    print("[strata-engine-pp] DECODE_START", flush=True)
                n += 1
                yield int(line.split()[1])
            elif line.startswith("PP ") or line.startswith("INFO "):
                if line.startswith("PP "):
                    print(f"[strata-engine-pp] {line.strip()}", flush=True)
                continue
            elif line.startswith("ERR"):
                raise EngineError(line.strip())
            elif line.startswith("DONE"):
                elapsed = time.time() - started
                if n > 0 and elapsed > 0:
                    print(f"[strata-backend] GEN done: {n} tokens in {elapsed:.1f}s "
                          f"({n / elapsed:.1f} tok/s)", flush=True)
                return {"tokens": n, "elapsed": elapsed}

    def _stream(self, o, streaming=True):
        """Common Predict/PredictStream logic: yields pb.Reply per delta."""
        self._check_identity(o.ModelIdentity)
        prompt = o.Prompt
        tools_text = _tool_system_block(o.Tools or "")
        if tools_text:
            # merge the tools into the FIRST system frame: a second adjacent
            # system frame makes Qwen3 emit im_end as its first token
            head = IM_START + "system"
            end = prompt.find(IM_END) if prompt.startswith(head) else -1
            if end != -1:
                prompt = prompt[:end] + "\n\n" + tools_text + prompt[end:]
            else:
                prompt = head + "\n" + tools_text + IM_END + "\n" + prompt
        print(f"[strata-backend] prompt({len(prompt)} chars): {prompt[:600]!r} ... TAIL: {prompt[-400:]!r}", flush=True)
        ids = self.tok.encode(prompt, parse_special=True)
        max_new = o.Tokens if o.Tokens > 0 else 4096
        keys = self._sampling_keys(o)
        stop_prompts = [sp for sp in (o.StopPrompts or []) if sp]

        dec = codecs.getincrementaldecoder("utf-8")("replace")
        buf = bytearray()
        raw = ""             # everything decoded so far, unmodified
        raw_sent = 0         # bytes of `raw` already sent as message deltas
        n = 0                # deltas yielded
        post_think = False   # True after the thinking close token
        started_text = False # False until the first non-newline char in the current phase
        cd_reason = ""       # reasoning routed via chat deltas
        cd_content = ""      # content routed via chat deltas
        rs_sent = 0          # reasoning chars already flushed
        ct_sent = 0          # content chars already flushed
        tc_index = 0         # tool call index for chat deltas
        rejected_tool = False # current reply must not expose rejected raw markup
        # Only the functions actually offered in this request may use the
        # Zoo function/parameter fallback. Their JSON Schemas also preserve
        # numeric/boolean argument types when the model emits plain text.
        tool_specs = {}
        try:
            for tool in json.loads(o.Tools or "[]"):
                fn = tool.get("function", tool)
                if not isinstance(fn, dict):
                    continue
                props = (fn.get("parameters") or {}).get("properties", {})
                if isinstance(fn.get("name"), str) and isinstance(props, dict):
                    tool_specs[fn["name"]] = props
        except (ValueError, TypeError, AttributeError):
            pass
        tool_parser = ToolCallStreamParser(tool_specs)
        finished = {"tokens": 0, "elapsed": 0.0}
        tok_idx = 0

        def consume(chunk: str, closing=False):
            """Route a decoded chunk through the tool parser + think phases.

            Returns (reasoning_delta, content_delta, tool_deltas)."""
            nonlocal post_think, started_text, tc_index, rejected_tool
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
                    post_think = True
                    started_text = True
                    ct_part += "Tool call rejected: invalid format. Retry with an offered tool."
                    continue
                if ev[0] == "tool":
                    tc_index += 1
                    tools.append(pb.ToolCallDelta(index=tc_index - 1,
                                                  id=str(uuid.uuid4()),
                                                  name=ev[1], arguments=ev[2]))
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
            gen = self._generate(ids, max_new, keys)
            try:
                while True:
                    try:
                        tid = gen.send(None)
                    except StopIteration as e:
                        finished = e.value or finished
                        break
                    if tid in self.think_close_ids:
                        # the thinking close marker routes to the reasoning
                        # stream; the blank run before the answer is suppressed
                        buf += self.tok.token_bytes(tid)
                        full = dec.decode(bytes(buf), final=False)
                        chunk = full[len(raw):]
                        raw = full
                        rep = send_events(chunk, "", None)
                        if rep:
                            yield rep
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
                        if tok_idx == 0 and prompt:
                            try:
                                ts = time.strftime("%Y%m%d-%H%M%S")
                                with open(f"/tmp/strata-badprompt-{ts}.txt", "w") as fh:
                                    fh.write(prompt)
                                print(f"[strata-test] dumped zero-token prompt to "
                                      f"/tmp/strata-badprompt-{ts}.txt", flush=True)
                            except OSError as e:
                                print(f"[strata-test] prompt dump failed: {e}", flush=True)
                        self.engine.stop()
                        self.engine.drain()
                        break
                    tok_idx += 1
                    buf += self.tok.token_bytes(tid)
                    full = dec.decode(bytes(buf), final=False)
                    if len(full) <= len(raw):
                        continue          # the decoder is still buffering multibyte input
                    chunk = full[len(raw):]
                    raw = full
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
                            self.engine.stop()
                            self.engine.drain()
                            return
                        if (streaming and not post_think and not tools and not ct_part and
                                not cd_reason.replace("<think>", "").replace("</think>", "").strip()):
                            continue
                        rep = make_reply(rs_part, ct_part, tools)
                        if rep:
                            yield rep
            except GeneratorExit:
                self.engine.stop()
                self.engine.drain()
                raise
        # trailing bytes that the incremental decoder still holds
        tail = dec.decode(bytes(buf), final=True)
        if len(tail) > len(raw):
            rs_part, ct_part, tools = consume(tail[len(raw):], closing=True)
        else:
            rs_part, ct_part, tools = consume("", closing=True)
        cd_reason += rs_part
        cd_content += ct_part
        if cd_reason[rs_sent:] or cd_content[ct_sent:] or tools:
            yield make_reply(cd_reason[rs_sent:], cd_content[ct_sent:], tools)

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
        print("[strata-backend] Free: unloading the engine (releasing GPU/RAM)", flush=True)
        self.engine.free()
        return pb.Result(success=True, message="freed")

    def Predict(self, request, context):
        try:
            text_parts = []
            tokens = 0
            prompt_tokens = 0
            for reply in self._stream(request, streaming=False):
                text_parts.append(reply.message.decode("utf-8", "replace"))
                tokens = max(tokens, reply.tokens)
                prompt_tokens = reply.prompt_tokens
            return pb.Reply(message="".join(text_parts).encode("utf-8"), tokens=tokens,
                            prompt_tokens=prompt_tokens)
        except Exception as e:
            if self.engine.alive():
                self.engine.drain()
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    def PredictStream(self, request, context):
        try:
            yield from self._stream(request)
        except Exception as e:
            # an aborted/crashed generation leaves engine lines in the queue;
            # drain them so the next request cannot read the old conversation
            if self.engine.alive():
                self.engine.drain()
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

    def shutdown(*_):
        backend.engine.free()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_BackendServicer_to_server(backend, server)
    bound = server.add_insecure_port(f"{a.host}:{a.port}")
    if bound == 0:
        print(f"failed to bind {a.host}:{a.port}", file=sys.stderr)
        return 1
    server.start()
    print(f"[strata-backend] listening on {a.host}:{a.port}, model={cfg['model_name']}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
