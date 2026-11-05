# Storyboard — DynamicLLM Runtime 4-Min Video

8 scenes. Record slides 1–7 + live demo, then closing. Use OBS / Zoom / PowerPoint screen record at 1080p.

| Time | Scene | Visual (show) | Narration | On-screen text |
|------|-------|---------------|-----------|----------------|
| 0:00–0:25 | 1. Hook | Slide 1: Title + 4 pain icons | VIDEO_SCRIPT Scene 1 | "One API. Every provider." / 4 pains |
| 0:25–0:55 | 2. What is DLLM | Slide 2: Architecture flow (App → API → Orchestrator → Router → Adapters → Telemetry) | Scene 2 | Architecture labels |
| 0:55–1:30 | 3. Backends + Routing | Slide 3: Adapter cards + 7 strategies list | Scene 3 | `OpenAI / llama.cpp / local / in-process` + strategy names |
| 1:30–2:05 | 4. Lifecycle + Memory | Slide 4: State diagram + memory table + eviction policies | Scene 4 | `Registered→Loading→Ready→Busy` / `LRU LFU cost priority` |
| 2:05–2:35 | 5. Agents + API | Slide 5: Endpoint table + dashboard thumb + SDK snippet | Scene 5 | `POST /v1/infer`, `/infer/stream`, `/agent/run`, `/dashboard` |
| 2:35–3:20 | 6. DEMO | Terminal + browser (no slide). Full-screen terminal, font 18+. | Scene 6, slow | Live commands + output |
| 3:20–3:50 | 7. Stack | Slide 6: Stack + reliability + metrics | Scene 7 | `Rust Tokio Axum` / `retries timeouts backpressure` |
| 3:50–4:00 | 8. Close | Slide 7: Closing + repo + future work | Scene 8 | "Thanks for watching" |

## Recording tips

- Demo order: `cargo run` → `/v1/models` → `/v1/infer` → `/v1/agent/run` → `/dashboard`. Pre-build with `cargo build` so `cargo run` is instant.
- If no API key / no local model, that's fine: show `/v1/models`, `/v1/status`, `/dashboard`, and the agent `calc` tool (works without an LLM for the tool demo path — otherwise narrate over the request JSON). Never show secrets on screen.
- Zoom browser to 125% for readability. Clear terminal (`cls`) before each command.
- Record narration separately if needed, then overlay. Export 1080p MP4, < 200 MB.
- Background: quiet room, mic 15 cm away. Clap once at start for sync.

## B-roll / cutaways (optional)

- `config/models.json` open in editor (5s) during Scene 3.
- Dashboard auto-refresh ticking (5s) during Scene 5.
- `cargo test` passing (3s) during Scene 7.
