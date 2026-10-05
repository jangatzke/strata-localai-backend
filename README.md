# Strata ↔ LocalAI gRPC Bridge

A Python external backend for [LocalAI](https://github.com/mudler/LocalAI) that runs the [Strata](https://github.com/Niko1221/Strata) inference engine and exposes it as an OpenAI-compatible chat model through LocalAI. The engine, model weights, vision projector and tokenizer pack are **not** included.

This bridge targets Strata's `--serve` stdin/stdout protocol and LocalAI's gRPC `backend.Backend` protocol. It is not a general-purpose backend for arbitrary engines. The examples use the dedicated, non-login service account **`stratal`**, not an interactive administrator. Existing installations may use another account; changing these examples does not migrate an existing service.

## Architecture

```text
OpenAI-compatible client
    → LocalAI /v1/chat/completions
    → external gRPC backend on port 50053
    → Strata text engine --serve + model pack/tokenizer
      + optional CPU strata-vision image encoder
```

`LoadModel` starts the configured engine and waits for `READY`. `Predict` and `PredictStream` tokenize ChatML prompts and convert token IDs into structured content, reasoning and tool-call deltas. `Free` unloads the text/vision processes and releases resources; a later request can load them again. The separate Strata `server.py` is not required and must not hold the same model's resources while the bridge is serving it.

The example defaults to **`parallel: 1`**, the serial `GEN`/`GENI` path. Set top-level `"parallel": 2` for exactly two request slots using `BGEN`/`BGENI`, `BADM`, `BT`, `BDONE` and slot-scoped `BSTOP`. The bridge adds `--batch 2` unless engine arguments already contain `--batch`/`--slots` syntax. Those arguments are preserved; the engine must report at least two slots in `INFO batch_slots=N`. Other root-level `parallel` values are rejected. Two slots do not guarantee two active decoders during every phase: admission/prefill is serialized, and the engine may finish an admission via its legacy compatibility path.

## Files

| File | Purpose |
| --- | --- |
| `strata_grpc_backend.py` | Engine/vision lifecycle, gRPC surface and tool-stream parser |
| `backend.proto` | LocalAI protocol used to generate Python stubs |
| `qwen3.8-flash-next-strata.yaml` | LocalAI model configuration and chat/history/image templates |
| `strata-backend.example.json` | Host-specific HIP/IQ3_S example, including CPU vision |
| `run.sh`, `strata-localai-backend.service` | Launcher and hardened systemd example using `stratal` |
| `test_parser.py`, `test_recovery.py` | Deterministic parser, streaming and bounded-recovery regressions |
| `test_batch.py`, `test_vision.py`, `test_lifecycle.py` | Batch routing, image protocol, cleanup/cancellation and shutdown regressions |
| `test_recovery_grpc.py` | CPU-only real-gRPC recovery/exhaustion fixtures |
| `test_live_stream.py`, `test_live_history.py` | Live LocalAI probes; generated calls are never executed |

## Requirements

- A matching built Strata checkout, model pack, tokenizer files and runtime libraries. The example uses ROCm/HIP, a Qwen3.8 Flash Next IQ3_S pack and Python 3.14 ROCm library paths; these are **not portable defaults**.
- Python 3.10+ with `venv`/pip support and the dependencies in `requirements.txt`. The reference deployment uses Python 3.14. Match `lib_dirs` to the Python version and GPU architecture of the **Strata build's** environment, which may differ from the bridge venv.
- LocalAI supporting external gRPC backends and structured `Reply.chat_deltas`. This repository's proto comes from LocalAI revision `3d63736`; other versions need a compatibility check.
- For the example image setup: a built `strata-vision`, a compatible `mmproj` and model shard. Keep the `vision` block **and** text-engine `--vision` flag together. For text-only use, remove both. The model YAML contains an image template but does not install a vision encoder. JPEG/PNG/BMP/GIF signatures pass through without Pillow; converting other formats requires optional Pillow (`./venv/bin/python -m pip install Pillow`). Encoder support for the supplied format is still required.
- On the HIP example host, systemd grants the service the `render` supplementary group. Check `/dev/kfd` and render-node permissions on your distribution and adapt GPU groups as necessary. A plain `sudo -u stratal` invocation does not automatically reproduce a unit-only supplementary group.

## Fresh installation on the inference host

Do not run this recipe over an existing deployment. Provision Strata and its model/pack files separately under `/srv/stratal/Strata` and `/srv/stratal/Strata-data`, with service-user read/execute access and appropriate write access for Strata's caches. Do not grant broad write access to `/srv`.

```bash
# Fresh Linux host only; adapt the nologin path and GPU groups as needed.
sudo useradd --system --user-group --create-home \
  --home-dir /var/lib/stratal --shell /usr/sbin/nologin stratal
sudo install -d -o stratal -g stratal -m 0750 \
  /srv/strata-localai-backend /srv/stratal /var/log/stratal
sudo -u stratal env HOME=/var/lib/stratal git clone \
  https://github.com/jangatzke/strata-localai-backend.git /srv/strata-localai-backend
sudo -u stratal sh -c 'cd /srv/strata-localai-backend && \
  python3.14 -m venv venv && \
  ./venv/bin/python -m pip install -r requirements.txt && \
  ./venv/bin/python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. backend.proto && \
  cp strata-backend.example.json strata-backend.json && \
  chmod +x run.sh'
# Edit strata-backend.json and the sample unit for your host BEFORE starting.
sudo install -o root -g root -m 0644 /srv/strata-localai-backend/strata-localai-backend.service \
  /etc/systemd/system/strata-localai-backend.service
sudo systemctl daemon-reload
sudo systemctl enable --now strata-localai-backend
```

Substitute another installed Python 3.10+ for `python3.14` if appropriate; this does not change the ROCm paths in the JSON automatically. The sample unit hides `/home`, so its model and library paths intentionally live under `/srv/stratal`, with state/cache/log directories managed by systemd under `/var/lib/stratal`, `/var/cache/stratal` and `/var/log/stratal`.

Review every JSON path, especially `exe`, `cwd`, `strata_repo_tools`, tokenizer, pack/shards, MTP data, expert profile, control vector, vision projector, `lib_dirs` and tuning file. The supplied performance flags and experimental control vector are hardware/model-specific choices, not universal recommendations. Remove optional arguments together with all their values when their assets are unavailable. `gpu: 0` refers to the runtime device index, not a stable PCI identity.

`run.sh` binds gRPC to `0.0.0.0:50053` by default (`PORT` overrides the port; `HOST` overrides the bind address). This gRPC listener has **no authentication or TLS**. Restrict access to LocalAI/trusted hosts with a firewall or private interface; never expose it publicly. `127.0.0.1` is reachable only from the same network namespace, so a container on another namespace needs a reachable host address.

Configure LocalAI's `external_backends.json`, using the actual reachable address:

```json
{"strata": "YOUR_INFERENCE_HOST:50053"}
```

Place `qwen3.8-flash-next-strata.yaml` in LocalAI's model directory; its `backend: strata` must match the mapping and its model identity must match `model_name` in the bridge JSON. LocalAI hot-reloads the backend mapping; adding/changing model templates can require restarting **LocalAI**. A Python-only bridge update requires restarting **only the bridge**. Neither requires a host reboot or restarting unrelated containers.

## Verify

Generate the protobuf stubs and install requirements first. Run CPU regressions with the bridge venv from the checkout; they use fake engines and do not load a GPU model:

```bash
./venv/bin/python -m unittest -q \
  test_parser test_recovery test_batch test_vision test_lifecycle
./venv/bin/python test_recovery_grpc.py
./venv/bin/python -m py_compile strata_grpc_backend.py
```

Run `test_recovery_grpc.py` as a separate process: it mutates fixture state and executes assertions directly. If using pytest, select the five CPU test files explicitly; unrestricted collection imports the live SSE probe and can contact LocalAI. pytest is optional and is not installed by `requirements.txt`.

With LocalAI and the bridge running, set the API URL and, for authenticated LocalAI, `LOCALAI_API_KEY` in your environment (do not put real keys in source or shell history):

```bash
export LOCALAI_BASE_URL=http://YOUR_LOCALAI_HOST:8081/v1
export LOCALAI_MODEL=qwen3.8-flash-next-strata
./venv/bin/python test_live_stream.py
./venv/bin/python test_live_history.py
```

The SSE probe checks offered `update_todo_list` tool calls, `finish_reason: tool_calls` and absence of raw tool markup. The history probe checks five consecutive non-streaming turns with exact Unicode arguments, feeding actual tool calls back with an explicit **not executed** harness result. Neither probe executes returned calls. Optional `LIVE_HISTORY_RESULTS` stores verification metadata, not arguments/session text. These probes do not establish that every malformed dialect or long client session works.

Also check two distinct exact-marker text prompts back-to-back, an image with deterministic colors/positions, and a text request after the image. Verify each answer belongs to its own prompt, rather than relying on HTTP 200 or the model list. For `parallel: 2`, warm the engine and test distinguishable concurrent requests plus request-local cancellation; client wall-time overlap alone does not prove overlapping decode.

Service status: `systemctl status strata-localai-backend`. Logs: `journalctl -u strata-localai-backend -f` and the JSON's engine `log` path. The first request loads the model and can take tens of seconds or minutes. The code allows up to 1200 seconds for engine READY and a 3600-second generation ceiling; client/proxy timeouts can be shorter.

## Generation and tool contract

- A positive client `max_tokens` is used as the generation budget. When omitted/non-positive, the bridge uses remaining engine context minus an eight-token safety margin instead of imposing a fixed 4096-token default. For an omitted/non-positive limit, insufficient context is rejected; positive client limits are passed through without a bridge-side context-capacity check. Reasoning and tool text count toward the budget.
- `tool_call_format: native_xml` uses the Strata pack's function/parameter dialect in both initial and corrective prompts. An absent setting retains the legacy `json` contract with a schema-bound XML-like fallback. The YAML marks API tool history with `bridge_tool_history`; Python renders it in the configured dialect, preserving numeric precision and string whitespace. User examples are not rewritten.
- Tool-enabled streaming turns are buffered until validation completes. Rejected/empty turns may regenerate up to twice, discarding failed prose/arguments. A valid published call is never regenerated or replayed. Exhaustion raises an actual RPC error. This delays tool-enabled output; no-tools text still streams normally.
- Accepted offered calls become structured `ChatDelta.tool_calls`; malformed/truncated envelopes and unknown tools fail closed, subject to the narrow final-report exception below. JSON envelopes check an offered function name and object arguments (including a JSON string that decodes to an object), but do **not** validate required properties, argument names or property types. XML checks offered parameter names, required keys, supported top-level types and duplicate-value consistency, not full nested JSON Schema or enums. Clients/tool executors must perform their own complete validation and authorization. JSON neighbor handling can omit invalid calls while preserving valid ones; ambiguous native/XML suffix text rejects the **whole unpublished turn**, including provisionally parsed calls.
- One legacy compatibility heuristic can recover malformed JSON quotes/newlines in an offered `attempt_completion` with a string `result` property. Its greedy envelope matcher rebuilds a single `result` string, but does **not** prove the malformed input contained only one argument: apparent additional fields can be swallowed into the recovered report text. Do not treat it as strict envelope validation, a general repair mechanism or a safe repair path for side-effecting tools.
- JSON envelope scanning is string/escape-aware. Closing markers inside a valid JSON string remain argument data across arbitrary stream boundaries. Native raw strings are inherently ambiguous for some literal delimiter sequences; do not claim arbitrary payloads are always representable. Native string values are not XML-unescaped or silently repaired; template formatting newlines and the legacy fallback have separate conventions.
- Explicit thinking phases bypass the tool parser. Recognized thinking control tokens are not appended to structured reasoning. Each new byte block enters the incremental UTF-8 decoder once, and the decoder is finalized with empty input.
- Non-streaming `Predict` aggregates the same safe path and retains structured chat deltas rather than asking LocalAI to reparse rejected/raw tool text. Diagnostics report lengths and fixed parser-state/shape metadata, not prompt excerpts or argument values.

## Cleanup, cancellation and shutdown

- Generation cleanup and parser finalization stay under the owning request's semaphore permit. Outer RPC error handlers do **not** drain the shared queue after releasing ownership. Completed/already-drained generations are not stopped or drained a second time.
- Vision input accepts HTTP(S), local paths/`file://`, data URLs and raw base64. Remote URLs are fetched **by the bridge**, and local files are read with the service account’s permissions; there is no host/path allowlist. The default 50 MiB `max_image_bytes` limit is not an SSRF or file-access policy. Restrict untrusted callers and apply network/filesystem isolation before exposing vision to them.
- gRPC cancellation is checked during output waits and buffered tool retries. Cancelled queued callers may still wait to acquire a semaphore/admission lock; they are checked before sending new work once admitted.
- Serial cancellation stops/drains that generation before releasing ownership. Batch cancellation uses only `BSTOP <slot>` and drains that slot through `BDONE` before reuse, leaving admitted peers independent.
- Cancellation during pending `BGEN` admission retains ownership until delayed `BADM` is reconciled, within a separate 120-second cleanup window. If it cannot be reconciled, the slot stays quarantined and new admissions fail closed until engine reload; already-admitted peers can finish. Image embedding files are request-local and removed on completion/cancellation.
- Concurrent `Free` calls are serialized before accumulating generation permits. Normal unload permits subsequent reload. On SIGTERM/SIGINT, the bridge first marks itself stopping and closes gRPC admission; queued generation and new model loads are rejected before unload.
- Shutdown bounds **lock acquisition** to 30 seconds, not total teardown. Engine/vision process waits and pending batch reconciliation can take longer. The sample unit has a 90-second stop timeout; systemd can force termination when that expires. Choose a larger timeout if your operational policy requires the full reconciliation window.

## Safe updates and scope of verification

Back up deployed code/configuration, stage the replacement separately, run CPU regressions under the service user's environment, install the tested files and restart only the required service. Read back the installed checksum and service state, then verify text, structured tools and vision through LocalAI. Do not replace an active configuration with the example JSON or rename an existing service account as a side effect of a documentation update.

If maintenance requires a three-minute idle window, check both backend status and unchanged request/engine activity continuously for at least 180 seconds **before the first production write and restart**. New activity or missing evidence resets the window; an old log timestamp alone is not sufficient.

Passing API probes establishes those specific paths, not every client's UI behavior, arbitrary native delimiter payload or long Telegram/Zoo continuation. Verify the affected real client separately before declaring its issue resolved.

## Third-party protocol notice

`backend.proto` is copied from [LocalAI `backend/backend.proto` at revision `3d63736`](https://github.com/mudler/LocalAI/blob/3d63736/backend/backend.proto), licensed under MIT. See `LICENSE.LocalAI`. This repository does not include Strata source code or model files.
