# DynamicLLM Runtime — Quick Start

A Rust gateway that routes inference requests across OpenAI-compatible
backends (OpenAI API, `llama-server`) with retries, timeouts, concurrency
limits, memory-aware admission, and telemetry.

## Prerequisites

- Rust stable (edition 2021) with cargo.
- Network access to crates.io on first build (dependencies are vendored
  into the local cargo cache after that).
- Optional: an OpenAI API key, and/or a local
  [`llama-server`](https://github.com/ggerganov/llama.cpp) for local models.

## Configure

Copy-free defaults live in `config/`:

- `config/runtime.json` — host/port, timeouts, retries, concurrency,
  routing strategy, eviction policy, backend URLs.
- `config/models.json` — registered models and their capabilities.

Secrets and overrides via environment (or a `.env` file, which is
git-ignored):

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key for the `openai` backend |
| `OPENAI_BASE_URL` | Override (default `https://api.openai.com/v1`) |
| `LLAMACPP_BASE_URL` | Override (default `http://127.0.0.1:8081`) |
| `LLAMACPP_BINARY` | `llama-server` executable for managed mode (default `llama-server`) |
| `LLAMACPP_CTX_SIZE` | `--ctx-size` for managed servers (default `4096`) |
| `LLAMACPP_N_GPU_LAYERS` | GPU layers offloaded by llama.cpp (default `0` = CPU-only) |
| `DLLM_HOST` / `DLLM_PORT` | Bind address (default `127.0.0.1:8080`) |
| `DLLM_CONFIG_DIR` | Config directory (default `config`) |
| `DLLM_LOG` | `trace`/`debug`/`info`/`warn`/`error` |

## Run

```sh
cargo run
```

With a local model (external server):

```sh
# Terminal 1: start llama.cpp's OpenAI-compatible server
llama-server -m model.gguf --port 8081

# Terminal 2: start the runtime (uses config/llamacpp_base_url)
cargo run
```

With a local model (managed server, SSD-backed, min RAM):

Set `gguf_path` on the model in `config/models.json` and the runtime
spawns and stops `llama-server` itself — no Terminal 1 needed:

```json
{
  "id": "big-local",
  "backend": "llama.cpp",
  "gguf_path": "E:/models/large.gguf",
  "local_port": 8081,
  "memory_required_mb": 4096,
  "quantization": "GGUF"
}
```

Managed servers start with `--mmap` and without `--mlock`, so weights
stay memory-mapped from SSD instead of pinned in RAM (a model larger
than RAM can run, with slower tokens when pages fault in). Tune with
`llamacpp_binary`, `llamacpp_ctx_size`, and `llamacpp_extra_args` in
`config/runtime.json`. Each managed model needs its own `local_port`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | Registered models |
| `GET` | `/v1/status` | Loaded models, memory, scheduler, telemetry |
| `POST` | `/v1/infer` | Non-streaming inference (`InferenceRequest` JSON) |
| `POST` | `/v1/infer/stream` | SSE streaming inference (`text/event-stream`) |
| `POST` | `/v1/agent/run` | Agentic task run (`AgentRequest` JSON, tool loop + audit) |
| `GET` | `/v1/agent/tools` | Tool definitions the agent executor can run |
| `GET` | `/dashboard` | Static ops dashboard (models, memory, scheduler, telemetry) |

Example:

```sh
curl -X POST http://127.0.0.1:8080/v1/infer \
  -H 'content-type: application/json' \
  -d '{"prompt": "Say hi", "model": "gpt-3.5-turbo", "max_tokens": 32}'
```

Agent example (`task` aliases `prompt`; `allowed_tools` is deny-by-default):

```sh
curl -X POST http://127.0.0.1:8080/v1/agent/run \
  -H 'content-type: application/json' \
  -d '{"task": "Compute 2+3", "allowed_tools": ["calc"], "max_steps": 4, "timeout_secs": 30}'
```

Request hints: set `"model"` to pin a model, `"stream": true` if you need a
streaming-capable route, `"tools": [...]` to require tool support, and
`"metadata": {"task": "<name>"}` to route by task affinity.

## Dashboard

With the server running, open `http://127.0.0.1:8080/dashboard` (or your
configured `host:port`). It is one dependency-free page that polls
`/v1/status` and `/v1/models` and shows registered models, loaded/resident
models with lifecycle state, host memory plus per-model residency,
scheduler/agent/prefetch gates, telemetry (totals, average latency,
success/fail, per-model request/error/latency/success-rate tables,
prefetch hit rate), and configured remote workers. Auto-refresh defaults
to 5s and can be changed or turned off in the header.

## Rust SDK

Embed the runtime in-process instead of calling HTTP (same operations,
no socket). See `examples/sdk_basic.rs` (`cargo run --example sdk_basic`):

```rust
use dynamic_llm_runtime::sdk::{AgentTaskBuilder, Client, InferBuilder};

let client = Client::from_config_dir("config").await?;
let response = client
    .infer(InferBuilder::new("Say hi").max_tokens(32).build())
    .await?;
let agent = client
    .run_agent(
        AgentTaskBuilder::new("Compute 2+3")
            .allow_tool("calc")
            .build(),
    )
    .await?;
let status = client.status().await; // same payload as GET /v1/status
```

`Client::builder()` overrides config fields (`routing_strategy`,
`request_timeout_secs`, `max_concurrent_requests`, ...) and the model
list; `client.runtime()` exposes the underlying `Runtime` for advanced
use (sessions, prefetch). Builder defaults mirror the HTTP API
(`max_steps = 8`, `timeout_secs = 60`, deny-by-default tool allowlist).

## Newer config fields (`config/runtime.json`)

Added by recent stages (all optional, serde defaults keep old files valid):

| Field | Default | Purpose |
| --- | --- | --- |
| `routing_latency_weight` | `1.0` | Latency term weight for `adaptive`/`quality_aware` routing |
| `routing_success_weight` | `1.0` | Success-rate term weight for `adaptive`/`quality_aware` routing |
| `routing_affinity_bonus` | `0.5` | Bonus when a candidate matches the request task affinity |
| `prefetch_concurrency` | `2` | Max concurrent background prefetches |
| `prefetch_warm_models` | `[]` | Model ids prefetched in the background at startup |
| `predictive_prefetch_enabled` | `false` | Prefetch the predictor's next-model guess after each request |
| `predictor_history_limit` | `100` | Recent model ids the next-model predictor retains |
| `workers` | `[]` | Remote workers (`{id, addr, models, capacity}`); empty = single-node |
| `llamacpp_binary` | `"llama-server"` | Executable for managed local models |
| `llamacpp_ctx_size` | `4096` | `--ctx-size` for managed servers |
| `llamacpp_extra_args` | `[]` | Extra argv appended to managed `llama-server` commands |
| `n_gpu_layers` | `0` | GPU layers for managed servers and the in-process engine (`0` = CPU-only; needs a CUDA-capable build when raised) |

Routing strategies: `fixed`, `rule_based`, `capability_based`,
`resource_aware`, `fallback`, `adaptive`, `quality_aware`.
Eviction policies: `lru`, `lfu`, `cost_aware`, `priority_aware`.

## Verify

```sh
cargo test     # unit + mock-backend-backed integration tests
cargo clippy --all-targets
cargo fmt --check
cargo bench    # manual-timing harness (registry, serde, SSE, router)
```

## Notes

- Config files are JSON (`serde_json`). The design doc mentions TOML; the
  `toml` crate was unavailable in the offline build environment, so JSON
  was used — the loader is one module (`src/config.rs`) if you want to
  switch back.
- `ModelManager` admission is budget accounting (`local_model_budget_mb`
  plus a host-RAM sanity check), not physical weight residency: evicting
  a plain HTTP-adapter handle does not free a remote server's RAM.
  Managed adapters (models with `gguf_path`) are the exception — unload
  stops their `llama-server` process, which does release its RAM.
- Streaming failover covers stream *startup* only (local or remote
  workers): failing over mid-stream would duplicate tokens. SSE deltas
  arrive as unnamed `data:` events, failures as one named `error` event
  (same text as the JSON `{"error"}` shape), and every stream ends with
  `data: [DONE]`.
