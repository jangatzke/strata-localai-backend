# Strata ↔ LocalAI gRPC Bridge

A Python external backend for [LocalAI](https://github.com/mudler/LocalAI) that runs the [Strata](https://github.com/Niko1221/Strata) inference engine and exposes it as an OpenAI-compatible chat model through LocalAI. The engine, model weights and tokenizer pack are **not** included.

This repository captures the working bridge deployed for `qwen3.8-flash-next-strata`. It is tied to Strata's `--serve` stdin/stdout protocol (`READY`, `GEN`, `T`, `DONE`, `STOP`, `QUIT`) and LocalAI's gRPC `backend.Backend` protocol; it is not a general-purpose LocalAI backend for arbitrary engines.

## Architecture

```text
OpenAI-compatible client (e.g. Zoo Code)
    → LocalAI /v1/chat/completions
    → external gRPC backend on port 50053
    → Strata engine --serve + model pack/tokenizer
```

`LoadModel` starts Strata and waits for `READY`; `Predict` and `PredictStream` tokenize ChatML prompts and process generated token IDs; `Free` stops the engine and releases GPU/RAM. Streaming tool calls are converted into structured `ChatDelta.tool_calls`. The parser accepts the native JSON tool format and a schema-bound XML-like fallback seen in Zoo Code. Malformed tool-call *blocks* fail closed: no guessed tool execution or raw block in streaming content. This does not guarantee that a model can never produce some other unrecognized syntax.

## Files

| File | Purpose |
| --- | --- |
| `strata_grpc_backend.py` | Bridge implementation and tool-stream parser |
| `backend.proto` | Matching LocalAI backend protocol; generate Python stubs from it |
| `qwen3.8-flash-next-strata.yaml` | LocalAI model config and ChatML templates |
| `strata-backend.example.json` | Example Strata paths and engine arguments for the deployed IQ3_S setup; edit for your host |
| `run.sh`, `strata-localai-backend.service` | Sample launcher and systemd unit (paths/user are host-specific) |
| `test_parser.py` | Deterministic parser and bridge streaming regressions |
| `test_live_stream.py` | Live LocalAI SSE smoke test |

## Requirements

- A built Strata checkout with the matching `engine/strata --serve` protocol, a model pack, tokenizer files, and GPU runtime for that build. The example uses ROCm/HIP and an IQ3_S Qwen3.8 Flash Next pack.
- Python 3.10+ (the deployed service uses Python 3.14), `grpcio`, `grpcio-tools`, and `regex` (used by Strata's tokenizer).
- LocalAI with support for external gRPC backends and structured `Reply.chat_deltas`, using the `backend.proto` version in this repository. The deployed LocalAI source revision was `3d63736`; other versions may need a matching proto and streaming behavior check.

## Install on the inference host

For a **fresh** installation, clone this repository into a directory owned by the intended service user (the example unit expects `/srv/strata-localai-backend`). Do not clone over an existing deployment. Adjust paths, service user, model ID, and interface to suit your host. Do not run a second standalone Strata server against the same model at the same time.

```bash
# Ensure /srv permits the service user to create the target directory first.
git clone https://github.com/jangatzke/strata-localai-backend.git /srv/strata-localai-backend
cd /srv/strata-localai-backend
python3.14 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. backend.proto
cp strata-backend.example.json strata-backend.json
# Edit strata-backend.json: exe, args, cwd, tokenizer, log, lib_dirs, gpu and env.
chmod +x run.sh
# Adapt and install strata-localai-backend.service under /etc/systemd/system/.
# Then systemctl daemon-reload, enable and start the service.
```

The example JSON contains paths from one deployment, **not portable defaults**. `run.sh` currently binds gRPC to `0.0.0.0:50053`; restrict that port with a firewall or bind it to an appropriate reachable interface. The example systemd unit runs as `jan` and assumes `/srv/strata-localai-backend`, so adapt its user and paths before installing it.

Configure LocalAI's `external_backends.json` with an address reachable from its container, for example:

```json
{"strata": "YOUR_INFERENCE_HOST:50053"}
```

Place `qwen3.8-flash-next-strata.yaml` in LocalAI's model directory and ensure its `backend: strata` matches that mapping. Start the bridge service. LocalAI hot-reloads the external backend mapping, but newly added or changed model YAML templates may require a LocalAI restart. Do not restart unrelated containers.

## Verify

Unit regressions do **not** require a running engine:

```bash
python3 -m unittest -q test_parser test_recovery
python3 -m py_compile strata_grpc_backend.py
```

With both the bridge and LocalAI running:

```bash
LOCALAI_BASE_URL=http://YOUR_LOCALAI_HOST:8081/v1 python3 test_live_stream.py
LOCALAI_BASE_URL=http://YOUR_LOCALAI_HOST:8081/v1 python3 test_live_history.py
```

The smoke test checks for `finish_reason: tool_calls`, an offered `update_todo_list` call, and no raw tool markup in SSE content. It does not force the model to reproduce every malformed dialect. Also verify a non-streaming completion and a follow-up turn containing the tool result for your client. Bridge status: `systemctl status strata-localai-backend`; logs: `journalctl -u strata-localai-backend -f`. The engine may take tens of seconds to load on the first request.

The history probe checks five consecutive non-streaming completions, feeding each actual assistant toolcall back into the next request with an explicit harness message stating that it was **not executed**. It verifies exact Unicode arguments and offered names without performing any file writes. `LOCALAI_MODEL` can select the model; optional `LIVE_HISTORY_RESULTS` stores only verification metadata, not arguments or session content. These fixtures do not replace a long real Telegram/Zoo session.

## Operational notes

- Output length: a positive client `max_tokens` is passed through unchanged. When omitted or non-positive, the bridge does not impose a fixed output budget; because Strata's `GEN` protocol requires a positive integer, it uses the remaining engine context minus its eight-token safety margin. Reasoning and tool-call text both count toward any explicit client limit.
- Tool-enabled turns are buffered until validation completes. Rejected turns are regenerated up to twice even when they contain a prose preamble. Failed prose/arguments are discarded, not replayed or logged. An exact standalone legacy rejection-text echo also triggers recovery. If any valid calls exist, they are published once; invalid neighbors are omitted and that turn is never regenerated. If all three attempts fail, gRPC returns an actual error, not a normal assistant rejection answer. Rejections include only fixed shape diagnostics (dialect, offered name, decoder error/offset), never argument values. Buffering delays tool-enabled output until the turn is complete; no-tools text still streams normally.
- Tool envelope boundaries are JSON-string-aware: a closing tool-call marker inside a JSON string (including escaped quotes/backslashes, Markdown fences and arbitrary stream chunk boundaries) remains argument data. Only a marker outside a string closes the envelope; incomplete JSON still fails closed. String contents are never repaired by this boundary scanner.
- Explicit thinking phases bypass the tool parser. Tool examples in reasoning are never executable and cannot switch the reply into content/rejection mode.
- Non-streaming `Predict` aggregates the same safe path as `PredictStream`, returns structured chat deltas, and suppresses raw tool/argument markup instead of asking LocalAI to reparse it. Both paths share bounded transactional recovery; neither regenerates a turn containing a valid call.
- Native XML boundaries walk function/parameter structure incrementally, preserving literal closing tags inside string values. Strings lose only the template's one outer newline at each end, not significant whitespace. Offered functions accept empty arguments when their schema has no required parameters. Missing required parameters, conflicting duplicates, unknown tools/parameters, malformed structure and truncated calls remain fail-closed.
- `tool_call_format: native_xml` uses the active Strata pack's function/parameter dialect in both initial and recovery prompts; absent configuration retains the legacy `json` contract. The example enables `native_xml`. The LocalAI template marks API tool-history entries with `<bridge_tool_history>`; Python renders those entries in the configured dialect, preserving integer precision and raw string whitespace. User examples are not rewritten. Native assistant prefill ends in `<think>\n`, and stream parsing starts inside that thinking phase. Keep the bridge configuration and model template consistent.
- Feed each new token byte block to the incremental UTF-8 decoder exactly once and finalize with empty input. Re-feeding cumulative bytes corrupts Unicode split across token boundaries.
- Native/XML calls with non-whitespace suffix text invalidate the whole buffered turn, including provisionally parsed calls. Such text may be the remainder of a string whose literal closing-tag sequence was mistaken for structure. Recovery regenerates only this unpublished turn; exhaustion returns an RPC error without publishing a truncated writing call. This applies to the native contract and the XML fallback, while ordinary JSON-call neighbor handling remains unchanged. Raw native strings remain an inherently ambiguous encoding; do not claim every literal delimiter sequence can be safely represented.
- Live validation is not a blanket guarantee: three API-only probes produced exact arguments after the native-contract change. An additional adversarial tool-history fixture produced a structured call but a non-exact file-content argument; that discrepancy and a long real Telegram continuation remain unverified. No generated probe tool was executed. A bridge rejection alone is not evidence of a model or context-length defect.
- Diagnostics log prompt length and fixed parser-state/shape metadata only, not prompt excerpts, argument values or full failed conversations. Preserve this privacy boundary when adding probes.
- A bridge-only Python change needs a bridge restart; a model YAML template change may need a LocalAI restart. A Zoo Code UI run is necessary before claiming its display is fixed.
- Strata can leave tokens queued after a cancelled generation. The bridge drains to `DONE` (or a bounded idle timeout) before accepting the next request.
- The service expects generated `backend_pb2.py` and `backend_pb2_grpc.py` beside the bridge. They are intentionally not committed.

## Third-party protocol notice

`backend.proto` is copied without modification from [LocalAI `backend/backend.proto` at revision `3d63736`](https://github.com/mudler/LocalAI/blob/3d63736/backend/backend.proto), licensed under MIT. See `LICENSE.LocalAI` for the upstream license. This repository does not include Strata source code or model files.
