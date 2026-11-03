//! Dependency-free runtime dashboard: one static HTML page (inline CSS/JS,
//! no external assets) served at `GET /dashboard`. It polls `/v1/status`
//! and `/v1/models` and renders models, residency, memory, scheduler,
//! agent, and telemetry state.

use axum::response::Html;

/// Serve the dashboard page.
pub async fn dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

/// Static dashboard markup. Kept as a plain string so the page needs no
/// template engine, no static-file directory, and no new dependencies.
pub const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DynamicLLM Runtime &mdash; Dashboard</title>
<style>
  :root {
    --text: #1a1a1a;
    --muted: #555;
    --border: #d8d8d8;
    --accent: #1e40af;
    --ok: #166534;
    --bad: #b91c1c;
    --panel: #fff;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--text);
    background: #fff;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 14px;
    line-height: 1.45;
  }
  header {
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    flex-wrap: wrap;
  }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header .controls { font-size: 13px; color: var(--muted); }
  header select, header button { font-size: 13px; }
  main { max-width: 1000px; padding: 16px 20px 40px; }
  section.panel {
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--panel);
    padding: 12px 14px;
    margin: 0 0 14px;
  }
  section.panel h2 { font-size: 15px; margin: 0 0 8px; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 4px 8px; border-top: 1px solid #eee; }
  th { border-top: 0; color: var(--muted); font-weight: 600; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
  .dot.ok { background: var(--ok); }
  .dot.bad { background: var(--bad); }
  #error {
    display: none;
    border: 1px solid var(--bad);
    border-radius: 4px;
    color: var(--bad);
    padding: 8px 12px;
    margin: 0 0 14px;
  }
  footer { color: var(--muted); font-size: 12px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>DynamicLLM Runtime &mdash; Dashboard</h1>
  <div class="controls">
    <label>Refresh <select id="interval">
      <option value="0">off</option>
      <option value="2000">2s</option>
      <option value="5000" selected>5s</option>
      <option value="15000">15s</option>
    </select></label>
    <button id="refresh" type="button">Refresh now</button>
    <span id="updated"></span>
  </div>
</header>
<main>
  <div id="error" role="alert"></div>

  <section class="panel">
    <h2>Status</h2>
    <div id="statusline">Loading&hellip;</div>
  </section>

  <section class="panel">
    <h2>Models</h2>
    <table id="models"><thead><tr>
      <th>Id</th><th>Backend</th><th>Context</th>
      <th class="num">Memory MB</th><th>Streaming</th><th>Tools</th><th>Enabled</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section class="panel">
    <h2>Loaded models &amp; lifecycle</h2>
    <div id="loaded"></div>
    <table id="lifecycle"><thead><tr><th>Model</th><th>State</th></tr></thead>
    <tbody></tbody></table>
  </section>

  <section class="panel">
    <h2>Memory</h2>
    <div id="memory"></div>
    <table id="residency"><thead><tr>
      <th>Model</th><th class="num">Assumed MB</th><th class="num">Measured MB</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section class="panel">
    <h2>Scheduler &amp; agents</h2>
    <table id="scheduler"><thead><tr>
      <th>Gate</th><th class="num">Available</th><th class="num">Max</th><th>Extra</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section class="panel">
    <h2>Telemetry</h2>
    <div id="telemetry"></div>
    <table id="permodel"><thead><tr>
      <th>Model</th><th class="num">Requests</th><th class="num">Errors</th>
      <th class="num">Avg latency ms</th><th class="num">Success rate</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <section class="panel" id="workers-panel" hidden>
    <h2>Workers</h2>
    <table id="workers"><thead><tr>
      <th>Id</th><th>Address</th><th>Models</th>
      <th class="num">Capacity</th><th class="num">In flight</th>
    </tr></thead><tbody></tbody></table>
  </section>

  <footer>
    Raw data: <a href="/v1/status">/v1/status</a> &middot;
    <a href="/v1/models">/v1/models</a>
  </footer>
</main>
<script>
(function () {
  "use strict";
  function text(value) {
    if (value === null || value === undefined) return "\u2014";
    if (typeof value === "number" && !Number.isInteger(value)) return value.toFixed(1);
    return String(value);
  }
  function setBody(tableId, rows) {
    var tbody = document.querySelector("#" + tableId + " tbody");
    tbody.textContent = "";
    rows.forEach(function (cells) {
      var tr = document.createElement("tr");
      cells.forEach(function (cell) {
        var td = document.createElement("td");
        if (cell.num) td.className = "num";
        td.textContent = cell.text;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    if (rows.length === 0) {
      var tr = document.createElement("tr");
      var td = document.createElement("td");
      td.textContent = "none";
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
  }
  function num(value) { return { text: text(value), num: true }; }
  function str(value) { return { text: text(value), num: false }; }

  function render(status, models) {
    var ok = status.status === "running";
    var line = document.getElementById("statusline");
    line.textContent = "";
    var dot = document.createElement("span");
    dot.className = "dot " + (ok ? "ok" : "bad");
    line.appendChild(dot);
    line.appendChild(document.createTextNode(
      (ok ? "running" : text(status.status)) +
      " \u00b7 " + text(status.models) + " registered model(s)" +
      " \u00b7 " + (status.loaded_models || []).length + " loaded"
    ));

    setBody("models", (models || []).map(function (m) {
      var caps = m.capabilities || {};
      return [
        str(m.id), str(m.backend), num(m.context_limit),
        num(m.memory_required_mb),
        str(caps.supports_streaming), str(caps.supports_tools),
        str(m.enabled)
      ];
    }));

    var loaded = status.loaded_models || [];
    document.getElementById("loaded").textContent =
      loaded.length ? "Resident: " + loaded.join(", ") : "No models resident.";
    var lifecycle = status.lifecycle || {};
    setBody("lifecycle", Object.keys(lifecycle).sort().map(function (id) {
      return [str(id), str(lifecycle[id])];
    }));

    var mem = status.memory || {};
    var parts = ["Host memory (MB):"];
    parts.push("total " + text(mem.total_mb));
    parts.push("used " + text(mem.used_mb));
    parts.push("available " + text(mem.available_mb));
    document.getElementById("memory").textContent = parts.join(" \u00b7 ");
    var residency = status.residency || [];
    setBody("residency", residency.map(function (r) {
      return [str(r.model_id || r.model), num(r.estimated_mb), num(r.measured_mb)];
    }));

    var sched = status.scheduler || {};
    var agent = status.agent || {};
    var prefetch = status.prefetch || {};
    setBody("scheduler", [
      [str("inference"), num(sched.available_permits), num(sched.max_concurrent), str("")],
      [str("agents"), num(agent.available_permits), num(agent.max_concurrent),
        str(text(agent.tools) + " tools \u00b7 " + text(agent.sessions) + " sessions")],
      [str("prefetch"), num(prefetch.available_permits), num(prefetch.concurrency),
        str("predictor next: " + text((status.predictor || {}).next))]
    ]);

    var t = status.telemetry || {};
    var summary = [
      "total " + text(t.total_requests),
      "completed " + text(t.completed_requests),
      "failed " + text(t.failed_requests),
      "in flight " + text(t.in_flight_requests),
      "avg latency " + text(t.avg_latency_ms) + " ms",
      "prefetch hit rate " + text(t.prefetch_hit_rate) +
        " (" + text(t.prefetch_hits) + "/" + text(t.prefetch_total) + ")"
    ];
    document.getElementById("telemetry").textContent = summary.join(" \u00b7 ");
    var reqs = t.requests_per_model || {};
    var errs = t.errors_per_model || {};
    var lats = t.avg_latency_ms_per_model || {};
    var rates = t.success_rate_per_model || {};
    var ids = {};
    [reqs, errs, lats, rates].forEach(function (m) {
      Object.keys(m).forEach(function (k) { ids[k] = true; });
    });
    setBody("permodel", Object.keys(ids).sort().map(function (id) {
      var rate = rates[id];
      return [str(id), num(reqs[id] || 0), num(errs[id] || 0), num(lats[id]),
        str(rate === undefined || rate === null ? "\u2014" : (rate * 100).toFixed(1) + "%")];
    }));

    var dist = status.distributed || {};
    var panel = document.getElementById("workers-panel");
    if (dist.enabled) {
      panel.hidden = false;
      setBody("workers", (dist.workers || []).map(function (w) {
        return [str(w.id), str(w.addr), str((w.models || []).join(", ")),
          num(w.capacity), num(w.in_flight)];
      }));
    } else {
      panel.hidden = true;
    }
  }

  var timer = null;
  function schedule() {
    if (timer) { clearTimeout(timer); timer = null; }
    var ms = parseInt(document.getElementById("interval").value, 10);
    if (ms > 0) { timer = setTimeout(load, ms); }
  }
  function load() {
    Promise.all([
      fetch("/v1/status").then(function (r) {
        if (!r.ok) throw new Error("GET /v1/status -> " + r.status);
        return r.json();
      }),
      fetch("/v1/models").then(function (r) {
        if (!r.ok) throw new Error("GET /v1/models -> " + r.status);
        return r.json();
      })
    ]).then(function (pair) {
      document.getElementById("error").style.display = "none";
      render(pair[0], pair[1]);
      document.getElementById("updated").textContent =
        "updated " + new Date().toLocaleTimeString();
    }).catch(function (err) {
      var box = document.getElementById("error");
      box.textContent = "Dashboard refresh failed: " + err.message;
      box.style.display = "block";
    }).finally(schedule);
  }
  document.getElementById("refresh").addEventListener("click", load);
  document.getElementById("interval").addEventListener("change", schedule);
  load();
})();
</script>
</body>
</html>
"##;
