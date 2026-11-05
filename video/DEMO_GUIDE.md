# Demo Guide — copy-paste commands for Scene 6 (45s live demo)

Pre-record checklist (do once before recording):

```powershell
cargo build
# optional: set key in .env, never on screen
# OPENAI_API_KEY=... 
cargo run
# in second terminal, verify:
curl.exe http://127.0.0.1:8080/health
```

Then stop server (`Ctrl+C`), clear terminals (`cls`), start recording.

---

## Take 1 — Start server (10s)

Terminal 1 (font 18+, full screen):

```powershell
cargo run
```

Wait for `Listening on http://127.0.0.1:8080`. Say: "Server up, config loaded."

## Take 2 — List models (5s)

Terminal 2:

```powershell
curl.exe http://127.0.0.1:8080/v1/models
```

Say: "Two registered models — one cloud, one local."

## Take 3 — Inference (10s)

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/infer -H "content-type: application/json" -d "{\"prompt\": \"Say hi in one sentence\", \"max_tokens\": 32}"
```

Say: "Router picks, adapter runs, normalized response back."
If no backend reachable, say: "Without a key this returns a structured error — same shape, with retries and telemetry recorded." Show `/v1/status` instead:

```powershell
curl.exe http://127.0.0.1:8080/v1/status
```

## Take 4 — Agent (10s)

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/agent/run -H "content-type: application/json" -d "{\"task\": \"Compute 2+3\", \"allowed_tools\": [\"calc\"], \"max_steps\": 4, \"timeout_secs\": 30}"
```

Say: "Agent loop with only calc allowed — deny by default, audited."

Check tools:

```powershell
curl.exe http://127.0.0.1:8080/v1/agent/tools
```

## Take 5 — Dashboard (10s)

Open in browser (zoom 125%):

```
http://127.0.0.1:8080/dashboard
```

Scroll slowly: Status → Models → Lifecycle → Memory → Scheduler → Telemetry.
Say: "Live ops view — no dependencies, polls status and models."

---

## Fallback (offline, no models)

If inference fails offline, still show:

1. `/health` → `{"status":"ok"}`
2. `/v1/models` → registry JSON
3. `/v1/status` → lifecycle + telemetry JSON
4. `/dashboard` → renders with zeros
5. `cargo test` → passing tests

Narrate: "Routing, lifecycle, and telemetry all work offline; backends plug in when available."

## After recording

- `Ctrl+C` server.
- Trim dead air between commands to hit 45s.
- Blur any API keys if visible (shouldn't be — use .env).
