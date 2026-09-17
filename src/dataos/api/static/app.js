// DataOS Analyst Workbench - vanilla JS single-page client.
// Talks only to same-origin /api/* endpoints; no build step, no framework.

const state = {
  datasets: [],
  selectedDatasetIds: new Set(),
  contractId: null,
  contract: null,
  runId: null,
  resultOffset: 0,
  resultLimit: 50,
  resultRowCount: 0,
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const body = isJson ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? (body.detail || body.reason || JSON.stringify(body)) : body;
    throw new Error(`${response.status}: ${detail}`);
  }
  return body;
}

function el(id) { return document.getElementById(id); }

function showError(err) {
  console.error(err);
  alert(err.message || String(err));
}

// ---- health badge ----

async function checkHealth() {
  try {
    const health = await api("/api/health");
    const badge = el("backend-badge");
    badge.textContent = `backend: ${health.llm_backend}`;
    badge.className = "badge " + (health.llm_backend === "anthropic" ? "ok" : "warn");
  } catch (err) {
    el("backend-badge").textContent = "backend unreachable";
  }
}

// ---- datasets ----

async function refreshDatasets() {
  state.datasets = await api("/api/datasets");
  const tbody = document.querySelector("#dataset-table tbody");
  tbody.innerHTML = "";
  for (const d of state.datasets) {
    const tr = document.createElement("tr");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selectedDatasetIds.has(d.dataset_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selectedDatasetIds.add(d.dataset_id);
      else state.selectedDatasetIds.delete(d.dataset_id);
    });
    const tdCheck = document.createElement("td");
    tdCheck.appendChild(checkbox);
    tr.appendChild(tdCheck);
    for (const value of [d.name, d.row_count, d.column_count]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

el("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = el("file-input").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const name = el("dataset-name").value.trim();
  if (name) form.append("name", name);

  try {
    const result = await api("/api/datasets", { method: "POST", body: form });
    const report = result.parse_report;
    el("upload-feedback").textContent = report.rows_rejected > 0
      ? `Uploaded with ${report.rows_rejected} rejected row(s) out of ${report.rows_seen} - see server logs for the sample.`
      : `Uploaded ${result.dataset.profile.row_count} row(s), ${result.dataset.profile.column_count} column(s).`;
    await refreshDatasets();
  } catch (err) {
    showError(err);
  }
});

// ---- requirement ----

el("create-requirement-btn").addEventListener("click", async () => {
  const objective = el("objective-input").value.trim();
  if (!objective) return showError(new Error("describe what you want first"));
  if (state.selectedDatasetIds.size === 0) return showError(new Error("select at least one dataset"));

  try {
    const record = await api("/api/requirements", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ objective, dataset_ids: [...state.selectedDatasetIds] }),
    });
    state.contractId = record.contract_id;
    state.contract = record.contract;
    renderContract();
  } catch (err) {
    showError(err);
  }
});

function renderContract() {
  const c = state.contract;
  el("contract-panel").classList.remove("hidden");
  el("contract-summary").innerHTML = `
    <div><strong>Objective:</strong> ${escapeHtml(c.objective)}</div>
    <div><strong>Status:</strong> ${c.status}</div>
    <div><strong>Metrics:</strong> ${c.metrics.map((m) => `${escapeHtml(m.name)} (${m.definition_status})`).join(", ") || "none"}</div>
  `;

  const clarPanel = el("clarifications-panel");
  const approveBtn = el("approve-btn");
  if (c.clarifications && c.clarifications.length > 0) {
    clarPanel.classList.remove("hidden");
    approveBtn.classList.add("hidden");
    const list = el("clarifications-list");
    list.innerHTML = "";
    for (const clar of c.clarifications) {
      const li = document.createElement("li");
      li.textContent = `${clar.question} (${clar.why_material})`;
      list.appendChild(li);
    }
  } else {
    clarPanel.classList.add("hidden");
    approveBtn.classList.remove("hidden");
  }

  document.querySelector("#section-plan").classList.toggle("hidden", c.status !== "APPROVED");
}

el("revise-btn").addEventListener("click", async () => {
  const additionalContext = el("clarification-answer").value.trim();
  if (!additionalContext) return showError(new Error("answer the clarification first"));
  try {
    const record = await api(`/api/requirements/${state.contractId}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ additional_context: additionalContext }),
    });
    state.contract = record.contract;
    renderContract();
  } catch (err) {
    showError(err);
  }
});

el("approve-btn").addEventListener("click", async () => {
  try {
    const record = await api(`/api/requirements/${state.contractId}/approve`, { method: "POST" });
    state.contract = record.contract;
    renderContract();
  } catch (err) {
    showError(err);
  }
});

// ---- plan ----

el("plan-btn").addEventListener("click", async () => {
  try {
    const plan = await api(`/api/requirements/${state.contractId}/plan`, { method: "POST" });
    renderPlan(plan);
  } catch (err) {
    showError(err);
  }
});

function renderPlan(plan) {
  const panel = el("plan-panel");
  const mappingIssues = plan.schema_mapping.blocking_items
    .map((b) => `<div class="issue-row"><span class="pill blocking">MAPPING</span>${escapeHtml(b.reason)}</div>`)
    .join("");
  const qualityIssues = plan.data_quality.issues
    .map((i) => `<div class="issue-row"><span class="pill ${i.severity === "BLOCKING" ? "blocking" : "warn"}">${i.severity}</span>${escapeHtml(i.rule)}: ${escapeHtml(i.observed)}</div>`)
    .join("");

  panel.innerHTML = `
    <div><strong>Steps:</strong> ${plan.steps.map((s) => `${s.operation_id}`).join(" &rarr; ") || "none"}</div>
    <div><strong>Data quality:</strong> ${plan.data_quality.fitness}</div>
    <div><strong>Analytics strategy:</strong> ${plan.analytics_strategy.analysis_type} (${plan.analytics_strategy.status})</div>
    ${mappingIssues || qualityIssues ? `<h4>Issues</h4>${mappingIssues}${qualityIssues}` : ""}
  `;

  document.querySelector("#section-run").classList.toggle("hidden", plan.blocking);
  el("start-btn").classList.toggle("hidden", plan.blocking);
  if (plan.blocking) {
    panel.innerHTML += `<p class="pill blocking">Blocked - resolve the issue(s) above before running.</p>`;
  }
}

el("start-btn").addEventListener("click", async () => {
  try {
    const started = await api(`/api/requirements/${state.contractId}/start`, { method: "POST" });
    state.runId = started.run_id;
    document.querySelector("#section-run").classList.remove("hidden");
    el("run-status").textContent = `Run ${state.runId}: ${started.state}`;
    el("release-btn").classList.toggle("hidden", started.state !== "VERIFYING");
  } catch (err) {
    showError(err);
  }
});

// ---- release / explain / visualize / result ----

el("release-btn").addEventListener("click", async () => {
  try {
    const released = await api(`/api/runs/${state.runId}/release`, { method: "POST" });
    el("run-status").textContent = `Run ${state.runId}: ${released.run.state}`;
    if (released.run.state === "RELEASED") {
      document.querySelector("#section-result").classList.remove("hidden");
      await loadExplanation();
      await loadVisualization();
      await loadResult(0);
    } else {
      const incident = await api(`/api/runs/${state.runId}/incident`);
      el("run-status").innerHTML += `<div class="pill blocking">${incident.failure_class}: ${incident.safe_action}</div>`;
    }
  } catch (err) {
    showError(err);
  }
});

async function loadExplanation() {
  const output = await api(`/api/runs/${state.runId}/explain`);
  const findings = output.explanation.findings
    .map((f) => `<div class="finding"><span class="type">${f.type}</span> ${escapeHtml(f.statement)}</div>`)
    .join("");
  const limitations = (output.explanation.limitations || [])
    .map((l) => `<div class="issue-row"><span class="pill warn">LIMITATION</span>${escapeHtml(l)}</div>`)
    .join("");
  el("explanation-panel").innerHTML = `
    <h3>Explanation</h3>
    <p>${escapeHtml(output.explanation.summary)}</p>
    ${findings}
    ${limitations}
  `;
}

let chartInstances = [];

async function loadVisualization() {
  const viz = await api(`/api/runs/${state.runId}/visualize`);
  const result = await api(`/api/runs/${state.runId}/result?limit=1000`);
  const rows = result.rows;

  for (const chart of chartInstances) chart.destroy();
  chartInstances = [];

  const panel = el("charts-panel");
  panel.innerHTML = "";
  if (viz.warnings.length > 0) {
    panel.innerHTML += viz.warnings
      .map((w) => `<div class="issue-row"><span class="pill warn">CHART WARNING</span>${escapeHtml(w)}</div>`)
      .join("");
  }

  for (const visual of viz.visuals) {
    if (visual.chart_type === "stat") {
      const value = rows.length > 0 ? rows[0][visual.y[0]] : "n/a";
      const tile = document.createElement("div");
      tile.className = "stat-tile";
      tile.innerHTML = `<div class="value">${escapeHtml(String(value))}</div><div class="label">${escapeHtml(visual.title)}</div>`;
      panel.appendChild(tile);
      continue;
    }

    const canvas = document.createElement("canvas");
    panel.appendChild(canvas);
    const labels = rows.map((r) => r[visual.x]);
    const datasets = visual.y.map((column) => ({ label: column, data: rows.map((r) => r[column]) }));
    const chart = new Chart(canvas, {
      type: visual.chart_type,
      data: { labels, datasets },
      options: { plugins: { title: { display: true, text: visual.title } } },
    });
    chartInstances.push(chart);
  }
}

async function loadResult(offset) {
  const result = await api(`/api/runs/${state.runId}/result?limit=${state.resultLimit}&offset=${offset}`);
  state.resultOffset = offset;
  state.resultRowCount = result.row_count;

  const table = el("result-table");
  const columns = result.rows.length > 0 ? Object.keys(result.rows[0]) : [];
  table.querySelector("thead").innerHTML = `<tr>${columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
  table.querySelector("tbody").innerHTML = result.rows
    .map((row) => `<tr>${columns.map((c) => `<td>${escapeHtml(String(row[c]))}</td>`).join("")}</tr>`)
    .join("");

  el("result-pager").innerHTML = `
    <span>${offset + 1}-${Math.min(offset + state.resultLimit, result.row_count)} of ${result.row_count}</span>
  `;
  const prevBtn = document.createElement("button");
  prevBtn.textContent = "Prev";
  prevBtn.className = "secondary";
  prevBtn.disabled = offset === 0;
  prevBtn.onclick = () => loadResult(Math.max(0, offset - state.resultLimit));
  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Next";
  nextBtn.className = "secondary";
  nextBtn.disabled = offset + state.resultLimit >= result.row_count;
  nextBtn.onclick = () => loadResult(offset + state.resultLimit);
  el("result-pager").append(prevBtn, nextBtn);
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

checkHealth();
refreshDatasets();
