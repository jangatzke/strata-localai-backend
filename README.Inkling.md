# Inkling 2 Mini Strata bridge

`inkling_grpc_backend.py` is a separate LocalAI gRPC entrypoint for the `strata-inkling` shape engine; it does not replace the Qwen bridge. It loads the HF o200k `tokenizer.json` through the MIT-licensed Strata upstream `tools/strata_tokenizer.py` from the dedicated `strata_repo_tools` directory in its JSON configuration. The upstream tokenizer and model pack are deliberately not vendored here.

The production instance lives in `/srv/strata-localai-backend-inkling` and runs as `strata-localai-backend-inkling.service`. Its normal sampling stop emits `[strata-test] stop_ids hit at token N: tid=ID` **only after** a valid channel body and the engine's terminal `DONE` have been drained. The llama-bottleneck-monitor parses that event as a completed request. Other aborts/errors must not be recorded as successful completions.

Offline regression tests need the pack tokenizer and upstream tool module. For example:

```sh
INKLING_TOKENIZER_DIR=/path/to/Inkling-2-Mini-Strata \
INKLING_TOOLS_DIR=/path/to/Strata/tools \
python -m unittest test_inkling_grpc_backend -v
```

Without those assets, this separate test class skips; existing bridge tests remain independent. The bridge does not start an engine during these tests. Deploying a code change requires restarting **only** the Inkling bridge after checking that no requests are active; that restart unloads the currently resident Inkling model.
