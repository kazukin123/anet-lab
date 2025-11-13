/* ============================================================
   Metrics Viewer - Phase1 リファクタリング（UIと状態の分離）
   ※ 挙動は変更せず、責務のみ整理
   ============================================================ */

const API_BASE_URL = (false) ? "/dummy_api" : "/api";
const AUTO_RELOAD_INTERVAL_MS = 10000;	// AutoReload間隔
const MAX_POINTS = 6000;	// 4Kモニタ想定
const MAX_SCATTER_GL = 0;	// あまり大きくするとグラフがでなくなる

const Mode = Object.freeze({
	UNINITIALIZED: "uninitialized",
	META_LOADING: "metaLoading",
	DATA_LOADING: "dataLoading",
	NORMAL: "normal",
	SCREENSHOT: "screenshot",
	ERROR: "error"
});

// パレット
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

/**
 * TypedArray対応の高速デシメーション
 * O(N/log N)程度のキャッシュ効率で動作
 * @param {{x: Int32Array|Float32Array, y: Float32Array}} trace 
 * @param {number} maxPoints 最大プロット点数
 * @param {[number, number]|null} range 表示範囲 [xmin, xmax]
 * @returns {{x: Int32Array|Float32Array, y: Float32Array}}
 */
function decimateTrace(trace, maxPoints = 8000, range = null) {
  if (!trace.x || trace.x.length <= maxPoints) return trace;

  const x = trace.x;
  const y = trace.y;
  const n = x.length;

  // 範囲抽出（二分探索）
  let startIdx = 0, endIdx = n;
  if (range) {
    const [xmin, xmax] = range;
    let lo = 0, hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (x[mid] < xmin) lo = mid + 1; else hi = mid;
    }
    startIdx = lo;
    lo = 0; hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >>> 1;
      if (x[mid] > xmax) hi = mid - 1; else lo = mid;
    }
    endIdx = hi + 1;
  }

  const visibleCount = endIdx - startIdx;
  if (visibleCount <= maxPoints) {
    return { ...trace, x: x.subarray(startIdx, endIdx), y: y.subarray(startIdx, endIdx) };
  }

  const step = Math.ceil(visibleCount / maxPoints);
  const outLen = Math.ceil(visibleCount / step);

  const outX = new (x.constructor)(outLen);
  const outY = new (y.constructor)(outLen);
  for (let i = 0, j = startIdx; j < endIdx && i < outLen; j += step, i++) {
    outX[i] = x[j];
    outY[i] = y[j];
  }

  // traceをクローンしてメタ情報維持
  return {
    ...trace,
    x: outX,
    y: outY
  };
}


// Toast
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

function base64ToFloat32Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function base64ToInt32Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int32Array(bytes.buffer);
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

	async fetchMetrics(cacheState) {
console.log("cacheState=", cacheState);
	    const c = this._ctrl();
	    const res = await fetch(`${API_BASE_URL}/metrics.json`, {
	        method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ runTagMap: cacheState }),
	        signal: c.signal
	    });

	    if (!res.ok) throw new Error(`Failed metrics.json: ${res.status}`);
		const json = await res.json();
console.log("json=", json);

		// --- Base64 → TypedArray の復元 ---
		for (const item of json.data ?? []) {
		    if (item.encodedSteps) {
		      item.steps = base64ToInt32Array(item.encodedSteps);
		      delete item.encodedSteps;
		    }
		    if (item.encodedValues) {
		      item.values = base64ToFloat32Array(item.encodedValues);
		      delete item.encodedValues;
		    }
		}

		return json;
	}

	abortAll() {
		for (const c of this.controllers) { try { c.abort(); } catch (_) {} }
		this.controllers = [];
	}
}

/* ---------------- キャッシュ ---------------- */
class DataCache {
	constructor() {
		this.runs = {};
		this.data = {};
	}

	clear() {
		this.runs = {};
		this.data = {};
	}

	updateRuns(runArray) {
		if (runArray == null) {
			this.runs = {};
			return;
		}
		this.runs = {};
		for (const run of runArray) {
			this.runs[run.id] = run;
		}
	}

	/** 差分データをマージ */
	merge(payload) {
console.log("payload=", payload);
		if (!payload || !Array.isArray(payload.data)) return;

		for (const m of payload.data) {
//console.log("run=", m);
			if (!this.data[m.runId]) this.data[m.runId] = {};

			let cur = this.data[m.runId][m.tagKey];
			if (!cur) {
				// 新規タグ → 新しいバッファとして作成
				const stepsBuf = new Int32Buffer(m.steps.length || 1024);
				const valuesBuf = new Float32Buffer(m.values.length || 1024);
				stepsBuf.append(m.steps);
				valuesBuf.append(m.values);
				this.data[m.runId][m.tagKey] = {
					steps: stepsBuf,
					values: valuesBuf,
					beginStep: m.beginStep ?? 0,
					endStep: m.endStep ?? (m.steps.at(-1) || 0)
				};
				continue;
			}

			// 既存データへの追記
			const last = cur.steps.length > 0 ? cur.steps.buffer[cur.steps.length - 1] : -Infinity;
			const newSteps = m.steps.filter(s => s > last);
			if (newSteps.length > 0) {
				const startIndex = m.steps.length - newSteps.length;
				const newVals = m.values.slice(startIndex);
				cur.steps.append(newSteps);
				cur.values.append(newVals);
			}
			// 取得済みステップ数範囲を更新（データ無しでもstep数だけ進む可能性を考慮）
			cur.beginStep = Math.min(cur.beginStep ?? Infinity, m.beginStep ?? Infinity);
			cur.endStep = Math.max(cur.endStep ?? -Infinity, m.endStep ?? -Infinity);
		}
console.log("data=", this.data);
	}

	/** 指定run/tagのデータ取得（TypedArray subarray） */
	get(runId, tagKey) {
	  const d = this.data[runId]?.[tagKey];
	  if (!d) return null;
	  return {
	    beginStep: d.beginStep,
	    endStep: d.endStep,
	    steps: d.steps.buffer.subarray(0, d.steps.length),
	    values: d.values.buffer.subarray(0, d.values.length)
	  };
	}

	getRuns() {
		return this.runs;
	}

	getRunIds() {
		return Object.keys(this.runs);
	}

	getTagKeys(runId) {
		const run = this.runs[runId];
		if (!run) return [];
		return run.tags.map(tag => tag.key);
	}

	buildCacheStateMap(targetRuns = null, targetTags = null) {	// デフォルトで全Run、全Tag
		const runIds = targetRuns ?? this.getRunIds();
		const map = {};
		for (const r of runIds) {
			const tagMap = {};
			const tags = targetTags ?? Object.keys(this.data[r] ?? {});
			for (const t of tags) {
				const d = this.data[r]?.[t];
				if (!d) {
					tagMap[t] = -1;
					continue;
				}
				// buffer再割り当て中でも安全な範囲コピー
				const safeEnd = (d.endStep != null)
				  ? d.endStep
				  : (d.steps.length ? d.steps.buffer[d.steps.length - 1] : -1);
				tagMap[t] = safeEnd;
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

	_makeTrace(runId, tagKey, steps, values, index) {
	  const traceType = (index < MAX_SCATTER_GL) ? 'scattergl' : 'scatter';
	  return {
		type: traceType,
	    x: steps,
	    y: values,
		name: `${runId}`,
	    mode: 'lines',
		line: { width: 1.5, color: this.app.runColorMap.get(runId) },
	    uid: `${runId}_${tagKey}`,
	  };
	}
	
	renderBySelection(containerSel, runIds, tagKeys, cache) {
		const area = $(containerSel).empty();
		Plotly.purge(area);
		if (!runIds.length || !tagKeys.length) {
			area.append("<div style='color:#888;padding:12px;'>No selection.</div>");
			return false;
		}
		
		let drawn = false;
		let numTraces = 0;
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
				const trace = this._makeTrace(r, tagKey, d.steps, d.values, numTraces);
				traces.push(trace);
				numTraces++;
			}
			if (!traces.length) continue;
			// 画面幅ピクセル数前提でデータ間引き処理（最大点数を超える場合に実行）
			const reducedTraces = traces.map(t => decimateTrace(t, MAX_POINTS));
			//グラフ作成
			const layout = 	{
					margin: { t: 30, b: 20, l: 50, r: 10 }, height: 300, width: area.width(), autosize:false,
					plot_bgcolor: "#111", paper_bgcolor: "#111", font: { color: "#ccc" },
					xaxis: { gridcolor: "#444" }, yaxis: { gridcolor: "#444" },
					showlegend: (runIds.length > 1)
				};
			Plotly.newPlot(id, reducedTraces, layout,
				{ displayModeBar:'hover', responsive: false, useResizeHandler: false });

			// --- ズーム追従処理 ---
			const plotDiv = document.getElementById(id);
			plotDiv.on('plotly_relayout', (e) => {
				if (!e['xaxis.range[0]'] || !e['xaxis.range[1]']) return;
				const xmin = e['xaxis.range[0]'], xmax = e['xaxis.range[1]'];
				const zoomedTraces = traces.map(t => decimateTrace(t, MAX_POINTS, [xmin, xmax]));
				Plotly.react(plotDiv, zoomedTraces, plotDiv.layout);
			});
			drawn = true;
		}
		if (!drawn) area.append("<div style='color:#888;padding:12px;'>No metrics data.</div>");
		return drawn;
	}

	resizeAll() {
		$(".graph-block div[id^='graph-']").each(function () {
			const rect = this.getBoundingClientRect();
			const newWidth = Math.floor(rect.width);
			const newHeight = Math.floor(rect.height);
			Plotly.relayout(this, { width: newWidth, height: newHeight });
			Plotly.Plots.resize(this);
		});
	}
}

/* ---------------- UIController ---------------- */
class UIController {
	constructor(app) { this.app = app; }
	
	setLoadingSpinner(active) {
	    const el = document.getElementById("loading-spinner");
	    if (!el) return;
	    if (active) el.classList.add("active");
	    else el.classList.remove("active");
	}

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

	renderRunList(runs, selectedRunIds, runColorMap) {
		const runIds = Object.keys(runs);
		const $list = $("#run-list").empty();
		const palette = getPlotlyColors();

		// --- まず古いRun→新しいRunの順で色割当（昇順） ---
		runIds.sort(); // 昇順：古いRunから順に
		for (const runId of runIds) {
			if (!runColorMap.has(runId)) {
				const c = palette[runColorMap.size % palette.length];
				runColorMap.set(runId, c);
			}
		}

		// --- 新しいRun→古いRunの順で表示（降順） ---
		const sortedDesc = [...runIds].reverse();
		sortedDesc.forEach((runId) => {
			const run = runs[runId];
			const c = runColorMap.get(runId);
			const chk = selectedRunIds.includes(runId) ? "checked" : "";
			const title = run.stats
				? Object.entries(run.stats)
					.map(([k, v]) => `${k}: ${v}`)
					.join("\n")
				: "";

			$list.append(`
				<label class="run-row" title="${title}">
					<input type="checkbox" class="run-check" value="${runId}" ${chk}>
					<span class="run-color" style="background:${c};"></span> ${runId}
				</label><br>
			`);
		});
	}

	bindRunListEvents(runIds) {
	  const $list = $("#run-list");
	  const selected = this.app.selectedRuns; // ← 常にアプリ本体の参照を使う
	  const sortedRunIds = runIds.slice().sort((a, b) => b.localeCompare(a));

	  // チェックボックス（複数選択）
	  $list.find(".run-check").off("change").on("change", (e) => {
	    const id = e.currentTarget.value;
	    const ok = e.currentTarget.checked;

	    if (ok) {
	      if (!selected.includes(id)) selected.push(id); // 破壊的更新
	    } else {
	      if (selected.length <= 1) { e.currentTarget.checked = true; return; }
	      const i = selected.indexOf(id);
	      if (i >= 0) selected.splice(i, 1);            // 破壊的更新
	    }
	    this.app.onRunSelectionChanged();                // 即時でOK（デバウンス不要）
	  });

	  // 行クリック（単独選択＝ラジオ動作）
	  $list.find(".run-row").off("click").on("click", (e) => {
	    if ($(e.target).hasClass("run-check")) return; // 二重反応防止

	    const $checkbox = $(e.currentTarget).find(".run-check");
	    const id = $checkbox.val();

	    // 破壊的に単独選択へ置換
	    selected.splice(0, selected.length, id);

	    $list.find(".run-check").prop("checked", false);
	    $checkbox.prop("checked", true);

	    this.app.onRunSelectionChanged();
	  });

	  // 全選択
	  $("#btn-select-all-runs").off("click").on("click", () => {
	    // 破壊的更新に統一（参照を保つ）
	    selected.splice(0, selected.length, ...runIds);
	    $list.find(".run-check").prop("checked", true);
	    this.app.onRunSelectionChanged();
	  });

	  // 最新のみ
	  $("#btn-latest-only").off("click").on("click", () => {
	    const latest = sortedRunIds[0];
	    selected.splice(0, selected.length, ...(latest ? [latest] : []));
	    $list.find(".run-check").each((_, el) => { el.checked = (el.value === latest); });
	    this.app.onRunSelectionChanged();
	  });
	}

	updateRunColorChips(runColorMap) {
		$("#run-list label .run-color").each((i, el) => {
			const runId = $(el).next("input").val();
			const c = runColorMap.get(runId);
			if (c) el.style.background = c;
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
		$("#btn-reload").off("click").on("click", () => this.app.onReload());
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
		if (mode == Mode.UNINITIALIZED || mode == Mode.META_LOADING || mode == Mode.DATA_LOADING) {
			this.ui.setLoadingSpinner(true);
		} else {
			this.ui.setLoadingSpinner(false);
		}
	}

	async init() {
		try {
			console.log("[INIT] start");
			this.setMode(Mode.META_LOADING);

			// Run情報(+タグ情報)を取得
			const runsPayload = await this.fetcher.fetchRuns();
			console.log("[INIT] fetchRuns OK");
			const runs = Array.isArray(runsPayload?.runs) ? runsPayload.runs : [];

			// Run無しの場合はキャッシュ全クリして終わり
			const runIds = runs.map(r => typeof r === "string" ? r : (r.id ?? r.name ?? String(r)));
			if (!runIds.length) {
				this.cache.clear();
				$("#main-area").empty().append("<div style='color:#888;padding:12px;'>No runs.</div>");
				this.setMode(Mode.NORMAL);
				this.ui.bindStaticControls();
				Toast.show("No runs available。");
				console.log("[INIT] no runs found");
				return;
			}

			// Run情報をキャッシュ保存
			this.cache.updateRuns(runs);

			// Runsを描画
			this._populateRuns();

			// Tagsを描画
			const latest = this.selectedRuns[this.selectedRuns.length - 1];
			const tags = latest ? this.cache.getTagKeys(latest) : [];
			if (tags.length) this._populateTags(tags, true);	// Tagがあったら表示
			else this.ui.captureInitialTagList(this.activeTags);	// なかったら現在

			// メトリクス情報を取得
			this.setMode(Mode.DATA_LOADING);
			const cacheState = this.cache.buildCacheStateMap();
			const metricsPayload = await this.fetcher.fetchMetrics(cacheState);
			console.log("metricsPayload=", metricsPayload);
			this.cache.merge(metricsPayload);
			console.log("[INIT] fetchMetrics OK");

			// グラフ描画
			this._renderCurrent();

			// UIイベントBind
			this.ui.bindStaticControls();

			// 初期化終わり
			this.setMode(Mode.NORMAL);
			console.log("[INIT] completed normally");
		} catch (e) {
			console.error(e);
			this.setMode(Mode.ERROR);
			Toast.show("System error:" + e.message);
            console.log("[INIT] failed with error");
		}
	}

	_populateRuns() {
		if (!Array.isArray(this.selectedRuns)) this.selectedRuns = [];

		const runs = this.cache.getRuns();
		const runIds = this.cache.getRunIds();

		// selected が空 or 消失した場合 → 最新だけ自動選択
		if (!this.selectedRuns.length || !runIds.includes(this.selectedRuns[0])) {
			runIds.sort().reverse();
			const latestRunId = runIds[0];
			this.selectedRuns = [latestRunId];
		 }

		 // Runsを描画
		this.ui.renderRunList(runs, this.selectedRuns, this.runColorMap);
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
		const t0 = performance.now();
		const t1 = performance.now();
		this.ui.updateRunColorChips(this.runColorMap);
		const t2 = performance.now();
		this.plotly.renderBySelection("#main-area", this.selectedRuns.slice(), [...this.activeTags], this.cache);
		const t3 = performance.now();
		this.plotly.resizeAll();
		const t4 = performance.now();
		console.log(`[DRAW] total=${(t4-t0).toFixed(1)}ms | ui=${(t2-t1).toFixed(1)}ms | plotly=${(t3-t2).toFixed(1)}ms | resize=${(t4-t3).toFixed(1)}ms`);
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
				this.onReload();
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

	async onReload() {
	    if (this.mode === Mode.SCREENSHOT) return;
		console.log("[RELOAD] full start");
		console.log(""+this.selectedRuns +" ", this.activeTags);
	    try {
			// 既存の通信を全てキャンセル
	        this.fetcher.abortAll();
	        this.setMode(Mode.META_LOADING);

			// Run情報(+Tags)を取得してキャシュ保存
	        const runsPayload = await this.fetcher.fetchRuns();
			const runs = Array.isArray(runsPayload?.runs) ? runsPayload.runs : [];
			this.cache.updateRuns(runs);
	        console.log("[RELOAD] fetchRuns OK");

			// Run情報がなかったらクリアして終わる
	        const runIds = runs.map(r => typeof r === "string" ? r : (r.id ?? r.name ?? String(r)));
	        if (!runIds.length) {
	            this.cache.clear();
	            $("#main-area").empty().append("<div style='color:#888;padding:12px;'>No runs.</div>");
	            this.setMode(Mode.NORMAL);
	            Toast.show("No runs available。");
	            console.log("[RELOAD] no runs found");
	            return;
	        }

			// 選択されていたRunのRun情報がなくなっていたら最新Runを再選択
			this.selectedRuns = this.selectedRuns.filter(r => runIds.includes(r));
	        if (!this.selectedRuns.length) this.selectedRuns = [runIds[runIds.length - 1]];

			// Runsを描画		
	        this._populateRuns();

			// Tagsを描画
			this._updateTagListByRuns();

			// Metricsを取得、キャッシュ登録
	        this.setMode(Mode.DATA_LOADING);
			const cacheState = this.cache.buildCacheStateMap();
	        const metricsPayload = await this.fetcher.fetchMetrics(cacheState);	// Reload時は選択状態関係無く全件再取得
			this.cache.merge(metricsPayload);

	        console.log("[RELOAD] fetchMetrics OK");

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
}

/* ---------------- 起動 ---------------- */
let app = null;
window.addEventListener("load", () => {
	app = new MetricsViewerClientApp();
	app.init();
});
