/* ============================================================
   Metrics Viewer - Phase1 リファクタリング（UIと状態の分離）
   ※ 挙動は変更せず、責務のみ整理
   ============================================================ */

const API_BASE_URL = "/api";
const AUTO_RELOAD_INTERVAL_MS = 10000;
	
const Mode = Object.freeze({
	UNINITIALIZED: "uninitialized",
	META_LOADING: "metaLoading",
	DATA_LOADING: "dataLoading",
	NORMAL: "normal",
	SCREENSHOT: "screenshot",
	ERROR: "error"
});

const RUN_COLORS_FALLBACK = [
	"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
	"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
];

function getPlotlyColors() {
	try {
		if (window.Plotly?.colors?.qualitative?.D3) return Plotly.colors.qualitative.D3;
		if (window.Plotly?.colors?.qualitative?.Plotly) return Plotly.colors.qualitative.Plotly;
	} catch (_) {}
	return RUN_COLORS_FALLBACK;
}

class Toast {
	static show(msg, ms = 2500) {
		const el = document.createElement("div");
		el.textContent = msg;
		Object.assign(el.style, {
			position: "fixed", top: "10px", left: "50%", transform: "translateX(-50%)",
			background: "rgba(255,64,64,0.92)", color: "#fff", padding: "8px 16px",
			borderRadius: "6px", fontSize: "13px", zIndex: 200, boxShadow: "0 2px 6px rgba(0,0,0,0.3)"
		});
		document.body.appendChild(el);
		setTimeout(() => el.remove(), ms);
	}
}

/* ---------------- データ取得 ---------------- */
class DataFetcher {
	constructor() { this.controllers = []; }
	_ctrl() { const c = new AbortController(); this.controllers.push(c); return c; }

	async fetchRuns() {
		const c = this._ctrl();
		const res = await fetch(`${API_BASE_URL}/runs.json`, { signal: c.signal });
		if (!res.ok) throw new Error(`Failed runs.json: ${res.status}`);
		return res.json();
	}

	async fetchMetricsAll(runIds = [], tagKeys = []) {
	    const c = this._ctrl();
	    const body = new URLSearchParams();

	    for (const r of runIds) body.append("runIds", r);
	    for (const t of tagKeys) body.append("tagKeys", t);
	    const res = await fetch(`${API_BASE_URL}/metrics.json`, {
	        method: "POST",
	        headers: { "Content-Type": "application/x-www-form-urlencoded" },
	        body,
	        signal: c.signal
	    });

	    if (!res.ok) throw new Error(`Failed metrics.json: ${res.status}`);
	    return res.json();
	}

	async fetchMetricsDiff(runIds = [], tagKeys = [], since = null) {
	    // 差分APIをまだ実装していないにで今はfetchMetricsAllを呼ぶ
	    return this.fetchMetricsAll(runIds, tagKeys);
	}

	abortAll() {
		for (const c of this.controllers) { try { c.abort(); } catch (_) {} }
		this.controllers = [];
	}
}

/* ---------------- キャッシュ ---------------- */
class DataCache {
	constructor() { this.data = {}; }
	clear() { this.data = {}; }

	mergeAll(payload) {
		this.data = {};
		if (!payload || !Array.isArray(payload.data)) return;
		for (const m of payload.data) {
			if (!this.data[m.runId]) this.data[m.runId] = {};
			this.data[m.runId][m.tagKey] = { steps: m.steps, values: m.values };
		}
	}

	mergeDiff(payload) {
		if (!payload || !Array.isArray(payload.data)) return;
		for (const m of payload.data) {
			if (!this.data[m.runId]) this.data[m.runId] = {};
			const cur = this.data[m.runId][m.tagKey];
			if (!cur) {
				this.data[m.runId][m.tagKey] = { steps: m.steps, values: m.values };
				continue;
			}
			const last = cur.steps.length ? cur.steps[cur.steps.length - 1] : -Infinity;
			for (let i = 0; i < m.steps.length; i++) {
				if (m.steps[i] > last) {
					cur.steps.push(m.steps[i]);
					cur.values.push(m.values[i]);
				}
			}
		}
	}

	get(runId, tagKey) { return this.data[runId]?.[tagKey] || null; }
	getRunIds() { return Object.keys(this.data); }
	getTagKeys(runId) { return this.data[runId] ? Object.keys(this.data[runId]) : []; }

	buildSinceStepMap(selectedRuns, selectedTags) {
		const map = {};
		for (const r of selectedRuns) {
			const tagMap = {};
			for (const t of selectedTags) {
				const d = this.get(r, t);
				tagMap[t] = d && d.steps.length ? d.steps[d.steps.length - 1] : -1;
			}
			map[r] = tagMap;
		}
		return map;
	}
}

/* ---------------- 描画 ---------------- */
class PlotlyController {
	constructor(app) {
		this.app = app;
		this.colors = getPlotlyColors();
	}

	renderBySelection(containerSel, runIds, tagKeys, cache) {
		const area = $(containerSel).empty();
		if (!runIds.length || !tagKeys.length) {
			area.append("<div style='color:#888;padding:12px;'>No selection.</div>");
			return false;
		}
		let drawn = false;
		for (const tagKey of tagKeys) {
			const safe = tagKey.replace(/[^\w-]/g, "_");
			const id = `graph-${safe}`;
			const $b = $(`<div class="graph-block"><div class="graph-title">${tagKey}</div><div id="${id}"></div></div>`);
			area.append($b);
			const traces = [];
			for (let i = 0; i < runIds.length; i++) {
				const r = runIds[i];
				const d = cache.get(r, tagKey);
				if (!d) continue;
				traces.push({
					x: d.steps, y: d.values, name: r, mode: "lines",
					line: { width: 2, color: this.app.runColorMap.get(r) }
				});
			}
			if (!traces.length) continue;
			Plotly.newPlot(id, traces, {
				margin: { t: 30, b: 15, l: 50, r: 10 }, height: 300,
				plot_bgcolor: "#111", paper_bgcolor: "#111", font: { color: "#ccc" },
				xaxis: { gridcolor: "#444" }, yaxis: { gridcolor: "#444" },
				showlegend: (runIds.length > 1),
			}, { displayModeBar: true, responsive: true, useResizeHandler: true });
			drawn = true;
		}
		if (!drawn) area.append("<div style='color:#888;padding:12px;'>No metrics data.</div>");
		return drawn;
	}

	resizeAll() { $(".graph-block div[id^='graph-']").each(function () { Plotly.Plots.resize(this); }); }
}

/* ---------------- UIController ---------------- */
class UIController {
	constructor(app) { this.app = app; }

	applyMode(mode) {
		const b = document.body;
		b.classList.remove("uninitialized", "metaLoading", "dataLoading", "error");
		if (mode === Mode.UNINITIALIZED) b.classList.add("uninitialized");
		if (mode === Mode.META_LOADING) b.classList.add("metaLoading");
		if (mode === Mode.DATA_LOADING) b.classList.add("dataLoading");
		if (mode === Mode.ERROR) b.classList.add("error");
		const disabled = (mode === Mode.META_LOADING || mode === Mode.ERROR);
		document.querySelectorAll("button, input, select").forEach(el => el.disabled = disabled);
	}

	renderRunList(runIds, selected, runColorMap) {
		const $list = $("#run-list").empty(), palette = getPlotlyColors();
		runColorMap.clear();
		runIds.sort();
		runIds.reverse();
		runIds.forEach((id, i) => {
			const c = palette[i % palette.length], chk = selected.includes(id) ? "checked" : "";
			runColorMap.set(id, c);
			$list.append(`<label class="run-row"><span class="run-color" style="background:${c};"></span>
				<input type="checkbox" class="run-check" value="${id}" ${chk}> ${id}</label><br>`);
		});
	}

	bindRunListEvents(runIds, selected) {
		const $list = $("#run-list");
		$list.find(".run-check").off("change").on("change", (e) => {
			const id = e.currentTarget.value, ok = e.currentTarget.checked;
			if (ok) { if (!selected.includes(id)) selected.push(id); }
			else {
				if (selected.length <= 1) { e.currentTarget.checked = true; return; }
				const i = selected.indexOf(id); if (i >= 0) selected.splice(i, 1);
			}
			this.app.onRunSelectionChanged();
		});

		$("#btn-select-all-runs").off("click").on("click", () => {
			this.app.selectedRuns = [...runIds];
			$list.find(".run-check").prop("checked", true);
			this.app.onRunSelectionChanged();
		});

		$("#btn-latest-only").off("click").on("click", () => {
			const latest = runIds[0];
			this.app.selectedRuns = latest ? [latest] : [];
			$list.find(".run-check").each((_, el) => { el.checked = (el.value === latest); });
			this.app.onRunSelectionChanged();
		});
	}

	updateRunColorChips(runColorMap) {
		const palette = getPlotlyColors();
		$("#run-list label .run-color").each((i, el) => {
			const id = $(el).next("input").val();
			el.style.background = runColorMap.get(id) || palette[i % palette.length];
		});
	}

	renderTagList(tagKeys, active) {
		const $ul = $("#tag-list").empty();
		active.clear();
		tagKeys.sort();
		tagKeys.forEach(k => {
			const id = `tag-${k.replace(/[^\w-]/g, "_")}`;
			$ul.append(`<li id="${id}">${k}</li>`);
		});
	}

	bindTagListEvents(tagKeys, active) {
		const $ul = $("#tag-list");

		// 個別クリック
		$ul.find("li").off("click").on("click", (e) => {
			const li = e.currentTarget;
			$(li).toggleClass("active");

			// DOMの状態を最新のtruth sourceとして反映
			this.app.activeTags = new Set($("#tag-list li.active").map((_, el) => $(el).text()).get());

			this.app.onTagSelectionChanged();
		});

		// 全選択
		$("#btn-select-all").off("click").on("click", () => {
			$ul.find("li").addClass("active");

			this.app.activeTags = new Set($("#tag-list li.active").map((_, el) => $(el).text()).get());

			this.app.onTagSelectionChanged();
		});

		// 全解除
		$("#btn-clear-all").off("click").on("click", () => {
			$ul.find("li").removeClass("active");

			this.app.activeTags = new Set(); // DOMから取得でも良いが空集合確定なので即代入

			this.app.onTagSelectionChanged();
		});

		// 並べ替え
		if ($("#tag-list").data("ui-sortable")) {
			$("#tag-list").sortable("option", "update", () => {
				this.app.activeTags = new Set($("#tag-list li.active").map((_, el) => $(el).text()).get());
				this.app.onTagSelectionChanged();
			});
		}
	}

	captureInitialTagList(active) {
		active.clear();
		$("#tag-list li.active").each((_, li) => active.add($(li).text()));
		this.bindTagListEvents([...active], active);
	}

	bindStaticControls() {
		$("#btn-reload").off("click").on("click", () => this.app.onReloadFull());
		$("#btn-auto-reload").off("click").on("click", () => this.app.onToggleAutoReload());
		$("#btn-screenshot").off("click").on("click", () => this.app.onToggleScreenshot());
		$("#btn-screenshot-toggle").off("click").on("click", () => this.app.onToggleScreenshot());
		$(window).off("resize.mv").on("resize.mv", () => this.app.plotly.resizeAll());
	}
}

/* ---------------- MetricsViewerClientApp (with debug logs) ---------------- */
class MetricsViewerClientApp {
	constructor() {
		this.fetcher = new DataFetcher();
		this.cache = new DataCache();
		this.plotly = new PlotlyController(this);
		this.ui = new UIController(this);
		this.mode = Mode.UNINITIALIZED;
		this.selectedRuns = [];
		this.activeTags = new Set();
		this.runColorMap = new Map();
		this.autoReloadEnabled = false;
		this.autoReloadTimer = null;
		this.colors = getPlotlyColors();
		console.log("[INIT] MetricsViewerClientApp constructed");
	}

	setMode(mode) {
		const prev = this.mode;
		console.log(`[MODE] ${prev} → ${mode}`);
		this.mode = mode;
		this.ui.applyMode(mode);
		console.log(`[MODE] applied: current=${this.mode}`);
	}

	async init() {
		try {
			console.log("[INIT] start");
			this.setMode(Mode.META_LOADING);

			const runsPayload = await this.fetcher.fetchRuns();
			console.log("[INIT] fetchRuns OK");

			const raw = Array.isArray(runsPayload?.runs) ? runsPayload.runs : [];
			const runIds = raw.map(r => typeof r === "string" ? r : (r.id ?? r.name ?? String(r)));
			if (!runIds.length) {
				this.cache.clear();
				$("#main-area").empty().append("<div style='color:#888;padding:12px;'>No runs.</div>");
				this.setMode(Mode.NORMAL);
				this.ui.bindStaticControls();
				Toast.show("No runs available。");
				console.log("[INIT] no runs found");
				return;
			}

			this._populateRuns(runIds);

			this.setMode(Mode.DATA_LOADING);
			const metricsPayload = await this.fetcher.fetchMetricsAll();
			this.cache.mergeAll(metricsPayload);
			console.log("[INIT] fetchMetricsAll OK");

			const latest = this.selectedRuns[this.selectedRuns.length - 1];
			const tags = latest ? this.cache.getTagKeys(latest) : [];
			if (tags.length) this._populateTags(tags, true);
			else this.ui.captureInitialTagList(this.activeTags);

			this._renderCurrent();
			this.setMode(Mode.NORMAL);
			this.ui.bindStaticControls();
			console.log("[INIT] completed normally");
		} catch (e) {
			console.error(e);
			this.setMode(Mode.ERROR);
			Toast.show("System error:" + e.message);
            console.log("[INIT] failed with error");
		}
	}

	_populateRuns(runIds) {
		if (!Array.isArray(this.selectedRuns)) this.selectedRuns = [];

		 // 最新が runIds[runIds.length - 1]
		 const latest = runIds[runIds.length - 1];

		 // selected が空 or 消失した場合 → 最新だけ選択
		 if (!this.selectedRuns.length || !runIds.includes(this.selectedRuns[0])) {
		     this.selectedRuns = [latest];
		 }
		 this.ui.renderRunList(runIds, this.selectedRuns, this.runColorMap);
		 this.ui.bindRunListEvents(runIds, this.selectedRuns);
		 		console.log(`[RUN] populateRuns → ${runIds.length} runs, selected=${this.selectedRuns.length}`);
	}

	_populateTags(tagKeys, allActive) {
		this.ui.renderTagList(tagKeys, this.activeTags);
		if (allActive) {
			this.activeTags = new Set(tagKeys);
			$("#tag-list li").addClass("active");
		}
		this.ui.bindTagListEvents(tagKeys, this.activeTags);
		console.log(`[TAG] populateTags → total=${tagKeys.length}, active=${this.activeTags.size}`);
	}

	_updateTagListByRuns() {
		const set = new Set();
		for (const r of this.selectedRuns) {
			for (const t of this.cache.getTagKeys(r) || []) set.add(t);
		}
		const keys = [...set].sort();
		const keep = [...this.activeTags].filter(t => keys.includes(t));
		this._populateTags(keys, false);
		this.activeTags = new Set(keep);
		$("#tag-list li").each((_, li) => {
			if (this.activeTags.has($(li).text())) $(li).addClass("active");
		});
		console.log(`[INTERNAL] updateTagListByRuns → ${keys.length} tags (keep=${this.activeTags.size})`);
	}

	_renderCurrent() {
		this.ui.updateRunColorChips(this.runColorMap);
		this.plotly.renderBySelection("#main-area", this.selectedRuns.slice(), [...this.activeTags], this.cache);
		this.plotly.resizeAll();
		console.log(`[DRAW] renderCurrent: runs=${this.selectedRuns.length}, tags=${this.activeTags.size}`);
	}

	/* ---------- イベントハンドラ群（onXXX統一） ---------- */

	onRunSelectionChanged() {
		console.log(`[RUN] selection changed → ${this.selectedRuns.length} runs`);
		this._updateTagListByRuns();
		this._renderCurrent();
	}

	onTagSelectionChanged() {
		console.log(`[TAG] selection changed → ${this.activeTags.size} tags`);
		this._renderCurrent();
	}

	onToggleAutoReload() {
		this.autoReloadEnabled = !this.autoReloadEnabled;
		if (this.autoReloadEnabled) {
			this.autoReloadTimer = setInterval(() => {
//				this.onReloadDiff();
				this.onReloadFull();
			}, AUTO_RELOAD_INTERVAL_MS);
			$("#btn-auto-reload").text("Auto Reload: ON");
			Toast.show("Auto-reload enabled.");
			console.log("[AUTO] toggled → ON");
		} else {
			clearInterval(this.autoReloadTimer);
			this.autoReloadTimer = null;
			$("#btn-auto-reload").text("Auto Reload: OFF");
			Toast.show("Auto-reload disabled.");
			console.log("[AUTO] toggled → OFF");
		}
	}

	onToggleScreenshot() {
		const b = document.body, h = document.documentElement, btn = document.getElementById("btn-screenshot-toggle");
		const on = b.classList.contains("screenshot-mode");
		if (on) {
			b.classList.remove("screenshot-mode");
			h.classList.remove("screenshot-mode");
			this.setMode(Mode.NORMAL);
			if (btn) btn.textContent = "⬅";
			const hd = document.getElementById("screenshot-header");
			if (hd) hd.style.display = "none";
			setTimeout(() => this.plotly.resizeAll(), 300);
			console.log("[SHOT] toggled → NORMAL");
		} else {
			b.classList.add("screenshot-mode");
			h.classList.add("screenshot-mode");
			this.setMode(Mode.SCREENSHOT);
			if (btn) btn.textContent = "➡";
			const title = this.selectedRuns.length === 1 ? `Metrics Viewer — ${this.selectedRuns[0]}` : "Metrics Viewer";
			const hd = document.getElementById("screenshot-header");
			if (hd) {
				hd.textContent = title;
				hd.style.display = "block";
			}
			setTimeout(() => this.plotly.resizeAll(), 300);
			console.log("[SHOT] toggled → SCREENSHOT");
		}
	}

	async onReloadFull() {
	    if (this.mode === Mode.SCREENSHOT) return;
		console.log("[RELOAD] full start");
		console.log(""+this.selectedRuns +" ", this.activeTags);
	    try {
	        this.fetcher.abortAll();
	        this.setMode(Mode.META_LOADING);

	        const runsPayload = await this.fetcher.fetchRuns();
	        console.log("[RELOAD] fetchRuns OK");

	        const raw = Array.isArray(runsPayload?.runs) ? runsPayload.runs : [];
	        const runIds = raw.map(r => typeof r === "string" ? r : (r.id ?? r.name ?? String(r)));
	        if (!runIds.length) {
	            this.cache.clear();
	            $("#main-area").empty().append("<div style='color:#888;padding:12px;'>No runs.</div>");
	            this.setMode(Mode.NORMAL);
	            Toast.show("No runs available。");
	            console.log("[RELOAD] no runs found");
	            return;
	        }

	        this.selectedRuns = this.selectedRuns.filter(r => runIds.includes(r));
	        if (!this.selectedRuns.length) this.selectedRuns = [runIds[runIds.length - 1]];
	        this._populateRuns(runIds);

	        this.setMode(Mode.DATA_LOADING);

	        const tagKeys = Array.from(this.activeTags);
	        const metricsPayload = await this.fetcher.fetchMetricsAll(this.selectedRuns, tagKeys);

	        console.log("[RELOAD] fetchMetricsAll OK");

	        // --- スクロール位置を保存・復元 ---
			const $scrollTarget = $("#main-area");
			const scrollTop = $scrollTarget.scrollTop();
			this._renderCurrent();
			$scrollTarget.scrollTop(scrollTop);
	        // -----------------------------------

	        this.setMode(Mode.NORMAL);
	    } catch (err) {
	        console.error("[RELOAD] failed:", err);
//	        this.setMode(Mode.ERROR);
			Toast.show("Reload failed.");
	    }
	}

	async onReloadDiff() {
		if (this.mode === Mode.SCREENSHOT) return;
		try {
			const since = this.cache.buildSinceStepMap(this.selectedRuns, [...this.activeTags]);
			console.log("[RELOAD] diff update ${this.selectedRuns} ${this.activeTags}");
			const payload = await this.fetcher.fetchMetricsDiff(this.selectedRuns, this.activeTags, since);
			this.cache.mergeDiff(payload);
			this._renderCurrent();
		} catch (e) {
			console.error(e);
			Toast.show("Reload failed.");
		}
	}
}

/* ---------------- 起動 ---------------- */
let app = null;
window.addEventListener("load", () => {
	app = new MetricsViewerClientApp();
	app.init();
});
