# DynamicLLM Runtime — 4-Minute Video Script (~540 words, ~130 wpm)

Target: 4:00 total. Practice once at natural pace; pauses for demos are built in.

---

## SCENE 1 — Hook + Problem [0:00–0:25]

**On screen:** Title slide + pain points.

> Every LLM app hits the same wall. A different API for every provider. A fixed model for every task — whether it's a one-line summary or complex code. And on limited RAM and GPU memory, loading one more model can crash everything. No visibility into what's slow, what's loaded, or what failed. Today I'll show you how DynamicLLM Runtime fixes that.

*(~62 words)*

## SCENE 2 — What is DLLM? [0:25–0:55]

**On screen:** Architecture diagram.

> DynamicLLM Runtime, or DLLM, is a Rust-based execution layer between your app and your models. Instead of wiring each model API yourself, you talk to one runtime. It validates your request, picks the right model, prepares resources, runs inference, and returns a normalized response — with metrics for every step. One interface for cloud APIs and local engines.

*(~60 words)*

## SCENE 3 — Unified backends + Smart routing [0:55–1:30]

**On screen:** Backends + 7 routing strategies.

> Under the hood, inference adapters isolate every backend — OpenAI APIs, llama.cpp servers, local HTTP engines, and an in-process GGUF engine. A central model registry tracks capabilities, context limits, and memory needs. Then the router chooses. Seven strategies: fixed, rule-based, capability-based, resource-aware, fallback, plus adaptive and quality-aware that score real latency and success rate with a task-affinity bonus. A short summary goes to a small fast model; hard code goes to the strong one.

*(~82 words)*

## SCENE 4 — Lifecycle + Memory + Scheduling [1:30–2:05]

**On screen:** Lifecycle states + eviction policies.

> Models have a lifecycle — registered, loading, ready, busy, evicting, failed — with safe transitions so concurrent requests never corrupt state. The memory manager tracks SSD, RAM, and GPU VRAM. It does budget admission control, retains hot models, and evicts with LRU, LFU, cost-aware, or priority-aware policies. Managed local models run memory-mapped from SSD, so even a model larger than RAM can run. Add background prefetch and predictive scheduling, and cold starts drop sharply.

*(~78 words)*

## SCENE 5 — Agents + API + Dashboard + SDK [2:05–2:35]

**On screen:** API table + dashboard screenshot.

> For agentic AI, DLLM runs multi-step tool loops with a task controller, tool registry, session state, step limits, timeouts, and a full audit log — with deny-by-default tool permissions. You get a clean HTTP API: health, models, status, non-streaming infer, SSE streaming infer, agent run, and agent tools. A dependency-free dashboard shows models, lifecycle, memory, scheduler, and per-model telemetry live. Or skip HTTP entirely with the in-process Rust SDK — same operations, no socket.

*(~83 words)*

## SCENE 6 — LIVE DEMO [2:35–3:20]

**On screen:** Terminal + browser. Run commands, pause to let output show.

> Let me show you. I start the server with cargo run — it loads config and binds to localhost 8080. Now I list models — one cloud, one local. I send an inference request — the router picks, the adapter runs it, and I get a normalized response. Next, the agent endpoint computes two plus three with only the calc tool allowed. And here's the dashboard — loaded models, memory, scheduler gates, and live telemetry. That's observability out of the box.

*(~87 words — speak slowly, let typing/output fill time)*

Demo commands (see DEMO_GUIDE.md):
```
cargo run
curl http://127.0.0.1:8080/v1/models
curl -X POST http://127.0.0.1:8080/v1/infer -H "content-type: application/json" -d "{\"prompt\":\"Say hi\",\"max_tokens\":32}"
curl -X POST http://127.0.0.1:8080/v1/agent/run -H "content-type: application/json" -d "{\"task\":\"Compute 2+3\",\"allowed_tools\":[\"calc\"],\"max_steps\":4}"
# open http://127.0.0.1:8080/dashboard
```

## SCENE 7 — Stack + Reliability [3:20–3:50]

**On screen:** Tech stack + reliability checklist.

> It's built on Rust and Tokio for safe concurrency, Axum and Serde for the API, with tracing, sysinfo monitoring, and Criterion benchmarks. Reliability is baked in: bounded retries with backoff, timeouts, graceful shutdown, concurrency limits, backpressure, provider health checks, and structured errors. Every decision is measured — time to first token, total latency, throughput, VRAM, and cache hit rate — against a direct-inference baseline.

*(~62 words)*

## SCENE 8 — Closing [3:50–4:00]

**On screen:** Closing slide + repo + future work.

> DynamicLLM Runtime: one runtime for every model, routing that's smart, memory that's respected, and agents you can trust. All roadmap phases are done and tested. Thanks for watching.

*(~28 words)*

---

**Total: ~542 words = ~4:00 at 130–135 wpm + demo pauses.**
If you run long, trim Scene 4 to one sentence on eviction. If short, add one dashboard metric callout.
