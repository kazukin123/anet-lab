/* Metrics Viewer: SQLite range cache client */

const API_BASE_URL = "/api";
const AUTO_RELOAD_INTERVAL_MS = 30000;
const INGEST_POLL_INTERVAL_MS = 4000;
const VIEWPORT_DEBOUNCE_MS = 150;
const RUN_SOLO_INTERVAL_MS = 350;
const GRAPH_SCROLL_LOCK_DRAG_THRESHOLD_PX = 1;
const HOVER_SCROLL_DELAY_MS = 300;
const MAX_SAFE_STEP = Number.MAX_SAFE_INTEGER;
const STORAGE_KEY_TAGS = "anet.metricsviewer.activeTags";
const STORAGE_KEY_KNOWN_TAGS = "anet.metricsviewer.knownTags";
const STORAGE_KEY_GRAPH_SCROLL_LOCK = "anet.metricsviewer.graphScrollLockEnabled";
const STORAGE_KEY_LOD_MODE = "anet.metricsviewer.lodDisplayMode";

const Mode = Object.freeze({
	UNINITIALIZED: "uninitialized",
	META_LOADING: "metaLoading",
	NORMAL: "normal",
	SCREENSHOT: "screenshot",
	ERROR: "error"
});

const LodDisplayMode = Object.freeze({
	MIN_MAX: "MinMax",
	MEAN: "Mean",
	BAND: "Band"
});

const RUN_COLORS = Object.freeze([
	"#2F7DE1", "#F2C230", "#7A5CFF", "#008B8B", "#F05A28",
	"#C678DD", "#FF9F1C", "#00A99D", "#A3C720", "#E23B4F",
	"#2FBF71", "#E75A9B", "#D1D83B", "#B83280", "#00B36B",
	"#A6761D", "#4656D9", "#C85A17", "#D83BD2", "#A65A2E"
]);

function getRunColors() {
	return RUN_COLORS;
}

function encodePlotlyTraceUidPart(value) {
	const bytes = new TextEncoder().encode(String(value));
	return Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}

function makePlotlyTraceUid(runId, tagKey, suffix = "") {
	return `mv_${encodePlotlyTraceUidPart(runId)}_${encodePlotlyTraceUidPart(tagKey)}_${suffix}`;
}

function graphId(tagKey) {
	return `graph-${encodePlotlyTraceUidPart(tagKey)}`;
}

function decodeBase64Bytes(input) {
	if (!input) return new Uint8Array(0);
	const chunks = typeof input === "string" ? [input] : input;
	if (!Array.isArray(chunks)) return new Uint8Array(0);
	let totalBytes = 0;
	for (const chunk of chunks) {
		let padding = 0;
		if (chunk.endsWith("==")) padding = 2;
		else if (chunk.endsWith("=")) padding = 1;
		totalBytes += chunk.length * 3 / 4 - padding;
	}
	const bytes = new Uint8Array(totalBytes);
	let offset = 0;
	for (const chunk of chunks) {
		const binary = atob(chunk);
		for (let i = 0; i < binary.length; i++) bytes[offset + i] = binary.charCodeAt(i);
		offset += binary.length;
	}
	return bytes;
}

function base64ToFloat32Array(input) {
	const bytes = decodeBase64Bytes(input);
	const values = new Float32Array(Math.floor(bytes.length / Float32Array.BYTES_PER_ELEMENT));
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
	return values;
}

function base64ToFloat64Array(input) {
	const bytes = decodeBase64Bytes(input);
	const values = new Float64Array(Math.floor(bytes.length / Float64Array.BYTES_PER_ELEMENT));
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	for (let i = 0; i < values.length; i++) values[i] = view.getFloat64(i * 8, true);
	return values;
}

function decodeProjection(projection) {
	if (!projection) return null;
	if (projection.kind === "raw") {
		return Object.freeze({
			kind: "raw",
			steps: base64ToFloat64Array(projection.steps),
			values: base64ToFloat32Array(projection.values)
		});
	}
	if (projection.kind !== "lod") return null;
	return Object.freeze({
		kind: "lod",
		minMax: Object.freeze({
			steps: base64ToFloat64Array(projection.minMax?.steps),
			values: base64ToFloat32Array(projection.minMax?.values)
		}),
		summary: Object.freeze({
			steps: base64ToFloat64Array(projection.summary?.steps),
			mins: base64ToFloat32Array(projection.summary?.mins),
			maxs: base64ToFloat32Array(projection.summary?.maxs),
			means: base64ToFloat32Array(projection.summary?.means),
			minSteps: base64ToFloat64Array(projection.summary?.minSteps),
			maxSteps: base64ToFloat64Array(projection.summary?.maxSteps)
		})
	});
}

function clampSafeStep(value) {
	if (!Number.isFinite(value)) return 0;
	return Math.max(-MAX_SAFE_STEP, Math.min(MAX_SAFE_STEP, Math.trunc(value)));
}

function colorWithAlpha(hex, alpha) {
	const value = hex.replace("#", "");
	const r = Number.parseInt(value.slice(0, 2), 16);
	const g = Number.parseInt(value.slice(2, 4), 16);
	const b = Number.parseInt(value.slice(4, 6), 16);
	return `rgba(${r},${g},${b},${alpha})`;
}

class Toast {
	static show(message, durationMs = 2500) {
		const element = document.createElement("div");
		element.textContent = message;
		Object.assign(element.style, {
			position: "fixed",
			top: "10px",
			left: "50%",
			transform: "translateX(-50%)",
			background: "rgba(255,64,64,0.92)",
			color: "#fff",
			padding: "8px 16px",
			borderRadius: "6px",
			fontSize: "13px",
			zIndex: 1000
		});
		document.body.appendChild(element);
		setTimeout(() => element.remove(), durationMs);
	}
}

class DataFetcher {
	constructor() {
		this.metadataController = null;
		this.metricsController = null;
		this.priorityTail = Promise.resolve();
	}

	async fetchRuns() {
		if (this.metadataController) this.metadataController.abort();
		this.metadataController = new AbortController();
		const response = await fetch(`${API_BASE_URL}/runs.json`, {
			signal: this.metadataController.signal
		});
		if (!response.ok) throw new Error(`Failed runs.json: ${response.status}`);
		return response.json();
	}

	async fetchIngestProgress() {
		const response = await fetch(`${API_BASE_URL}/runs.json`);
		if (!response.ok) throw new Error(`Failed runs.json: ${response.status}`);
		return response.json();
	}

	async fetchMetrics(series) {
		this.abortMetrics();
		this.metricsController = new AbortController();
		const response = await fetch(`${API_BASE_URL}/metrics.json`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ series }),
			signal: this.metricsController.signal
		});
		if (!response.ok) {
			const body = await response.text();
			throw new Error(`Failed metrics.json: ${response.status} ${body}`);
		}
		return response.json();
	}

	async prioritize(runIds) {
		const send = async () => {
			const response = await fetch(`${API_BASE_URL}/runs/prioritize`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ runIds })
			});
			if (!response.ok) throw new Error(`Failed prioritize: ${response.status}`);
		};
		this.priorityTail = this.priorityTail.catch(() => {}).then(send);
		return this.priorityTail;
	}

	abortMetrics() {
		if (this.metricsController) {
			this.metricsController.abort();
			this.metricsController = null;
		}
	}

	abortAll() {
		this.abortMetrics();
		if (this.metadataController) {
			this.metadataController.abort();
			this.metadataController = null;
		}
	}
}

class DataCache {
	constructor() {
		this.runs = new Map();
		this.windows = new Map();
	}

	updateRuns(runArray) {
		const next = new Map();
		for (const run of runArray ?? []) next.set(run.id, run);
		for (const [runId, previous] of this.runs) {
			const current = next.get(runId);
			if (!current || current.generation !== previous.generation) this.removeRun(runId);
		}
		this.runs = next;
	}

	removeRun(runId) {
		this.runs.delete(runId);
		for (const key of this.windows.keys()) {
			if (key.startsWith(`${runId}\u0000`)) this.windows.delete(key);
		}
	}

	clear() {
		this.runs.clear();
		this.windows.clear();
	}

	getRuns() {
		return Object.fromEntries(this.runs);
	}

	getRun(runId) {
		return this.runs.get(runId) ?? null;
	}

	getRunIds() {
		return [...this.runs.keys()];
	}

	getTag(runId, tagKey) {
		return this.getRun(runId)?.tags?.find(tag => tag.key === tagKey) ?? null;
	}

	getTagKeys(runId) {
		return (this.getRun(runId)?.tags ?? []).map(tag => tag.key);
	}

	getWindow(runId, tagKey) {
		return this.windows.get(this._key(runId, tagKey)) ?? null;
	}

	replaceWindow(result, requestedViewportWidth) {
		const window = Object.freeze({
			runId: result.runId,
			tagKey: result.tagKey,
			generation: result.generation,
			fromStep: result.fromStep,
			toStep: result.toStep,
			availability: result.availability,
			pointBudget: result.pointBudget,
			level: result.level,
			bucketWidth: result.bucketWidth,
			issues: Object.freeze([...(result.issues ?? [])]),
			projection: decodeProjection(result.projection),
			requestedViewportWidth
		});
		this.windows.set(this._key(result.runId, result.tagKey), window);
		return window;
	}

	needsFetch(runId, tagKey, target, viewport, force) {
		if (force) return true;
		const current = this.getWindow(runId, tagKey);
		if (!current || current.generation !== this.getRun(runId)?.generation) return true;
		if (current.fromStep > target.fromStep || current.toStep < target.toStep) return true;
		if (current.level > 0
				&& Number.isFinite(current.requestedViewportWidth)
				&& viewport.width < current.requestedViewportWidth * 0.75) {
			return true;
		}
		return false;
	}

	_key(runId, tagKey) {
		return `${runId}\u0000${tagKey}`;
	}
}

class PlotlyController {
	constructor(app) {
		this.app = app;
	}

	static signedLogValue(value) {
		if (value === 0 || !Number.isFinite(value)) return value;
		return Math.sign(value) * Math.log10(1 + Math.abs(value));
	}

	static signedLogRawValue(value) {
		if (value === 0 || !Number.isFinite(value)) return value;
		return Math.sign(value) * (Math.pow(10, Math.abs(value)) - 1);
	}

	_formatValue(value) {
		if (!Number.isFinite(value)) return String(value);
		if (value === 0) return "0";
		const absolute = Math.abs(value);
		if (absolute >= 10000 || absolute < 0.001) {
			return value.toExponential(3).replace("e+", "e");
		}
		return String(Number(value.toPrecision(6)));
	}

	_makeLineTrace(runId, tagKey, steps, values, suffix = "line") {
		return {
			type: "scatter",
			x: steps,
			y: values,
			name: runId,
			mode: "lines",
			line: { width: 1.5, color: this.app.runColorMap.get(runId) },
			uid: makePlotlyTraceUid(runId, tagKey, suffix),
			meta: { tagKey, runId },
			legendgroup: makePlotlyTraceUid(runId, tagKey, "legend"),
			visible: this.app.isLegendSeriesHidden(tagKey, runId) ? "legendonly" : true,
			opacity: this.app.selectedRuns.length > 1 ? 0.8 : 1.0
		};
	}

	_makeSeriesTraces(runId, tagKey, window) {
		const projection = window?.projection;
		if (!projection) return [];
		if (projection.kind === "raw") {
			return [this._makeLineTrace(runId, tagKey, projection.steps, projection.values, "raw")];
		}
		if (this.app.lodDisplayMode === LodDisplayMode.MIN_MAX) {
			return [this._makeLineTrace(
					runId,
					tagKey,
					projection.minMax.steps,
					projection.minMax.values,
					"minmax")];
		}
		if (this.app.lodDisplayMode === LodDisplayMode.MEAN) {
			return [this._makeLineTrace(
					runId,
					tagKey,
					projection.summary.steps,
					projection.summary.means,
					"mean")];
		}

		const color = this.app.runColorMap.get(runId);
		const summary = projection.summary;
		const minCustomData = Array.from(
				summary.mins,
				(value, index) => [summary.minSteps[index], value]);
		const maxCustomData = Array.from(
				summary.maxs,
				(value, index) => [summary.maxSteps[index], value]);
		const lower = {
			...this._makeLineTrace(runId, tagKey, summary.steps, summary.mins, "band-min"),
			name: `${runId} min`,
			line: { width: 0, color },
			showlegend: false,
			hovertemplate: "run=%{fullData.name}<br>bucket=%{x}<br>min step=%{customdata[0]}<br>min=%{customdata[1]:.6g}<extra></extra>",
			customdata: minCustomData
		};
		const upper = {
			...this._makeLineTrace(runId, tagKey, summary.steps, summary.maxs, "band-max"),
			name: `${runId} max`,
			line: { width: 0, color },
			fill: "tonexty",
			fillcolor: colorWithAlpha(color, 0.28),
			showlegend: false,
			hovertemplate: "run=%{fullData.name}<br>bucket=%{x}<br>max step=%{customdata[0]}<br>max=%{customdata[1]:.6g}<extra></extra>",
			customdata: maxCustomData
		};
		const mean = {
			...this._makeLineTrace(runId, tagKey, summary.steps, summary.means, "band-mean"),
			name: runId,
			line: { width: 1.5, color }
		};
		return [lower, upper, mean];
	}

	_toDisplayTrace(trace, signedLogScale) {
		if (!signedLogScale) return { ...trace };
		const transformed = new Float32Array(trace.y.length);
		for (let i = 0; i < trace.y.length; i++) {
			transformed[i] = PlotlyController.signedLogValue(Number(trace.y[i]));
		}
		return {
			...trace,
			y: transformed,
			customdata: trace.customdata ?? trace.y,
			hovertemplate: trace.hovertemplate
					?? "run=%{fullData.name}<br>step=%{x}<br>value=%{customdata:.6g}<extra></extra>"
		};
	}

	_toDisplayTraces(traces, signedLogScale) {
		return traces.map(trace => this._toDisplayTrace(trace, signedLogScale));
	}

	_makeLayout(width, showLegend, signedLogScale, traces, ranges = {}) {
		const layout = {
			margin: { t: 30, b: 20, l: 50, r: 10 },
			height: 300,
			width,
			autosize: false,
			plot_bgcolor: "#111",
			paper_bgcolor: "#111",
			font: { color: "#ccc" },
			xaxis: { gridcolor: "#444" },
			yaxis: this._makeYAxis(signedLogScale, traces, ranges),
			showlegend: showLegend,
			legend: { groupclick: "togglegroup" }
		};
		if (Array.isArray(ranges.xRange)) layout.xaxis.range = ranges.xRange.slice();
		if (Object.prototype.hasOwnProperty.call(ranges, "dragMode")) {
			layout.dragmode = ranges.dragMode;
		}
		return layout;
	}

	_makeYAxis(signedLogScale, traces, ranges) {
		const axis = { gridcolor: "#444", type: "linear" };
		if (Array.isArray(ranges.yRange)) axis.range = ranges.yRange.slice();
		if (!signedLogScale) return axis;
		const ticks = this._makeSignedLogTicks(traces, ranges);
		return {
			...axis,
			zeroline: true,
			zerolinecolor: "#666",
			tickmode: "array",
			tickvals: ticks.map(PlotlyController.signedLogValue),
			ticktext: ticks.map(value => this._formatValue(value))
		};
	}

	_makeSignedLogTicks(traces, ranges) {
		let min = Infinity;
		let max = -Infinity;
		if (Array.isArray(ranges.yRange)) {
			const raw0 = PlotlyController.signedLogRawValue(Number(ranges.yRange[0]));
			const raw1 = PlotlyController.signedLogRawValue(Number(ranges.yRange[1]));
			min = Math.min(raw0, raw1);
			max = Math.max(raw0, raw1);
		} else {
			for (const trace of traces) {
				for (const raw of trace.y) {
					const value = Number(raw);
					if (!Number.isFinite(value)) continue;
					min = Math.min(min, value);
					max = Math.max(max, value);
				}
			}
		}
		if (!Number.isFinite(min) || !Number.isFinite(max)) return [0];
		const ticks = new Set();
		if (min <= 0 && max >= 0) ticks.add(0);
		const maxAbs = Math.max(Math.abs(min), Math.abs(max));
		if (maxAbs > 0) {
			const minPower = Math.min(0, Math.floor(Math.log10(maxAbs)));
			const maxPower = Math.max(0, Math.floor(Math.log10(maxAbs)));
			for (let power = minPower; power <= maxPower; power++) {
				const value = Math.pow(10, power);
				if (min <= -value && -value <= max) ticks.add(-value);
				if (min <= value && value <= max) ticks.add(value);
			}
		}
		if (ticks.size < 2) {
			ticks.add(min);
			ticks.add(max);
		}
		return [...ticks].sort((a, b) => a - b);
	}

	renderBySelection(containerSelector, runIds, tagKeys, cache) {
		const area = document.querySelector(containerSelector);
		this.capturePlotState(area);
		this.app.pruneLegendVisibility(runIds, tagKeys);
		this.app.prunePlotDragModes(tagKeys);
		for (const plot of area.querySelectorAll(".js-plotly-plot")) Plotly.purge(plot);
		area.replaceChildren();
		if (!runIds.length) {
			this._empty(area, "No selection.");
			return false;
		}
		if (!tagKeys.length) {
			this._empty(area, "No metrics data.");
			return false;
		}

		let drawn = false;
		const sortedRuns = runIds.slice().sort((a, b) => a.localeCompare(b));
		for (const tagKey of tagKeys) {
			const traces = [];
			for (const runId of sortedRuns) {
				const window = cache.getWindow(runId, tagKey);
				if (window?.availability !== "ok") continue;
				traces.push(...this._makeSeriesTraces(runId, tagKey, window));
			}
			if (!traces.length) continue;
			drawn = true;

			const block = document.createElement("div");
			block.className = "graph-block";
			const header = document.createElement("div");
			header.className = "graph-header";
			const title = document.createElement("div");
			title.className = "graph-title";
			title.textContent = tagKey;
			const logButton = document.createElement("button");
			logButton.type = "button";
			logButton.className = "graph-log-toggle";
			logButton.textContent = "Log";
			logButton.title = "Toggle signed log scale";
			const signedLogScale = this.app.logScaleTags.has(tagKey);
			logButton.classList.toggle("active", signedLogScale);
			logButton.setAttribute("aria-pressed", signedLogScale ? "true" : "false");
			header.append(title, logButton);

			const issue = this.app.issueForTag(tagKey);
			if (issue) {
				const warning = document.createElement("span");
				warning.className = "graph-warning";
				warning.textContent = "⚠";
				warning.title = issue;
				header.append(warning);
			}
			const stats = this.app.combinedStats(tagKey);
			if (stats) {
				const statsElement = document.createElement("span");
				statsElement.className = "graph-stats";
				statsElement.textContent =
						`Min ${this._formatValue(stats.min)} / Max ${this._formatValue(stats.max)}`
						+ ` / Avg ${this._formatValue(stats.mean)} / Std ${this._formatValue(stats.stdDev)}`;
				statsElement.title = `count=${stats.count}\nmin=${stats.min}\nmax=${stats.max}`
						+ `\navg=${stats.mean}\nstd=${stats.stdDev}`;
				header.append(statsElement);
			}

			const plot = document.createElement("div");
			plot.id = graphId(tagKey);
			block.append(header, plot);
			area.append(block);

			const viewport = this.app.explicitViewport(tagKey);
			const layout = this._makeLayout(
					this._plotWidth(block),
					runIds.length > 1,
					signedLogScale,
					traces,
					{
						xRange: viewport?.range ?? null,
						dragMode: this.app.graphScrollLockEnabled
								? false
								: this.app.plotDragMode(tagKey)
					});
			if (layout.dragmode === undefined) delete layout.dragmode;
			Plotly.newPlot(
					plot,
					this._toDisplayTraces(traces, signedLogScale),
					layout,
					{
						displayModeBar: "hover",
						responsive: false,
						useResizeHandler: false
					});

			logButton.addEventListener("click", event => {
				event.stopPropagation();
				this.app.onToggleLog(tagKey);
			});
			plot.on("plotly_relayout", event => {
				if (plot.__mvUpdatingPlot) return;
				if (this.app.graphScrollLockEnabled
						&& Object.prototype.hasOwnProperty.call(event ?? {}, "dragmode")
						&& event.dragmode !== false) {
					Plotly.relayout(plot, { dragmode: false });
					return;
				}
				const xRange = this._readRange(plot.layout?.xaxis, "xaxis", event);
				const yRange = this._readRange(plot.layout?.yaxis, "yaxis", event);
				const xChanged = this._hasRangeEvent(event, "xaxis");
				const yChanged = this._hasRangeEvent(event, "yaxis");
				if (!xChanged && !yChanged) return;
				this.capturePlotState(plot);
				const nextLayout = this._makeLayout(
						this._plotWidth(block),
						runIds.length > 1,
						this.app.logScaleTags.has(tagKey),
						traces,
						{
							xRange,
							yRange,
							dragMode: plot.layout?.dragmode ?? plot._fullLayout?.dragmode
						});
				this._reactPlot(
						plot,
						this._toDisplayTraces(
								this._applyLegendVisibility(traces),
								this.app.logScaleTags.has(tagKey)),
						nextLayout);
				if (xChanged) {
					this.app.onViewportChanged(
							tagKey,
							xRange,
							Boolean(event?.["xaxis.autorange"]));
				}
			});
		}
		if (!drawn) this._empty(area, "No metrics data.");
		return drawn;
	}

	capturePlotState(root) {
		const plots = root?.matches?.(".js-plotly-plot")
				? [root]
				: Array.from(root?.querySelectorAll?.(".js-plotly-plot") ?? []);
		for (const plot of plots) {
			const tagKey = Array.from(plot.data ?? [])
					.find(trace => typeof trace.meta?.tagKey === "string")?.meta.tagKey;
			const dragMode = plot._fullLayout?.dragmode ?? plot.layout?.dragmode;
			if (typeof tagKey === "string" && typeof dragMode === "string") {
				this.app.setPlotDragMode(tagKey, dragMode);
			}
			for (const trace of plot.data ?? []) {
				if (trace.showlegend === false) continue;
				const traceTagKey = trace.meta?.tagKey;
				const runId = trace.meta?.runId;
				if (typeof traceTagKey !== "string" || typeof runId !== "string") continue;
				this.app.setLegendSeriesHidden(traceTagKey, runId, trace.visible === "legendonly");
			}
		}
	}

	_applyLegendVisibility(traces) {
		return traces.map(trace => ({
			...trace,
			visible: this.app.isLegendSeriesHidden(trace.meta?.tagKey, trace.meta?.runId)
					? "legendonly"
					: true
		}));
	}

	_readRange(axisLayout, axisName, event) {
		const combined = event?.[`${axisName}.range`];
		if (Array.isArray(combined)) return combined.slice();
		const first = event?.[`${axisName}.range[0]`];
		const second = event?.[`${axisName}.range[1]`];
		if (first != null && second != null) return [first, second];
		if (event?.[`${axisName}.autorange`]) return null;
		return Array.isArray(axisLayout?.range) ? axisLayout.range.slice() : null;
	}

	_hasRangeEvent(event, axisName) {
		return Object.keys(event ?? {}).some(key =>
			key === `${axisName}.range`
			|| key === `${axisName}.range[0]`
			|| key === `${axisName}.range[1]`
			|| key === `${axisName}.autorange`);
	}

	_reactPlot(plot, traces, layout) {
		plot.__mvUpdatingPlot = true;
		Promise.resolve(Plotly.react(plot, traces, layout))
				.finally(() => { plot.__mvUpdatingPlot = false; });
	}

	_plotWidth(block) {
		return Math.max(1, Math.floor(block.getBoundingClientRect().width));
	}

	_empty(area, text) {
		const message = document.createElement("div");
		message.style.cssText = "color:#888;padding:12px;";
		message.textContent = text;
		area.append(message);
	}

	applyGraphScrollLock(enabled) {
		for (const plot of document.querySelectorAll(".graph-block .js-plotly-plot")) {
			if (enabled) {
				if (!Object.prototype.hasOwnProperty.call(plot, "__mvScrollLockPreviousDragMode")) {
					const tagKey = Array.from(plot.data ?? [])
							.find(trace => typeof trace.meta?.tagKey === "string")?.meta.tagKey;
					plot.__mvScrollLockPreviousDragMode =
							this.app.plotDragMode(tagKey)
							?? plot.layout?.dragmode
							?? plot._fullLayout?.dragmode
							?? "zoom";
				}
				Plotly.relayout(plot, { dragmode: false });
			} else {
				if (!Object.prototype.hasOwnProperty.call(
						plot,
						"__mvScrollLockPreviousDragMode")) continue;
				const previous = plot.__mvScrollLockPreviousDragMode;
				delete plot.__mvScrollLockPreviousDragMode;
				Plotly.relayout(plot, { dragmode: previous || "zoom" });
			}
		}
	}

	resizeAll() {
		for (const plot of document.querySelectorAll(".graph-block .js-plotly-plot")) {
			const rectangle = plot.getBoundingClientRect();
			const block = plot.closest(".graph-block");
			Plotly.relayout(plot, {
				width: block ? this._plotWidth(block) : Math.floor(rectangle.width),
				height: Math.floor(rectangle.height)
			});
			Plotly.Plots.resize(plot);
		}
	}

	async resetView() {
		const plots = Array.from(document.querySelectorAll(".graph-block .js-plotly-plot"));
		await Promise.all(plots.map(async plot => {
			// 凡例で隠した系列を戻してから、表示対象全体に対して軸を再計算する。
			await Plotly.restyle(plot, { visible: true });
			await Plotly.relayout(plot, {
				"xaxis.autorange": true,
				"yaxis.autorange": true
			});
		}));
	}
}

class UIController {
	constructor(app) {
		this.app = app;
		this.lastRunClick = { runId: null, at: -Infinity };
	}

	setLoadingSpinner(active) {
		document.getElementById("loading-spinner")?.classList.toggle("active", active);
	}

	renderUpdateStatus(failures) {
		const status = document.getElementById("update-status");
		const details = [];
		if (failures.metadata) details.push(`Metadata: ${failures.metadata}`);
		if (failures.metrics) details.push(`Metrics: ${failures.metrics}`);
		status.hidden = details.length === 0;
		status.textContent = details.length ? "Update failed" : "";
		status.title = details.join("\n");
	}

	applyMode(mode) {
		document.body.classList.remove("uninitialized", "metaLoading", "error");
		if (mode === Mode.UNINITIALIZED) document.body.classList.add("uninitialized");
		if (mode === Mode.META_LOADING) document.body.classList.add("metaLoading");
		if (mode === Mode.ERROR) document.body.classList.add("error");
	}

	renderRunList(runs, selectedRunIds, runColorMap) {
		const list = document.getElementById("run-list");
		list.replaceChildren();
		const runIds = Object.keys(runs).sort();
		for (const runId of runIds) {
			if (!runColorMap.has(runId)) {
				runColorMap.set(runId, getRunColors()[runColorMap.size % getRunColors().length]);
			}
		}
		for (const runId of runIds.reverse()) {
			const run = runs[runId];
			const row = document.createElement("div");
			row.className = `run-row ${run.ingest?.state ?? "pending"}`;
			row.classList.toggle("active", selectedRunIds.includes(runId));
			row.dataset.runId = runId;
			const percentage = this._ingestPercentage(run.ingest);
			const quarantineCount = (run.tags ?? []).filter(tag => tag.status === "error").length;
			const issueMessages = [];
			if (run.ingest?.error) issueMessages.push(
					`${run.ingest.error.code}: ${run.ingest.error.message}`);
			for (const tag of run.tags ?? []) {
				if (tag.error) issueMessages.push(`${tag.key}: ${tag.error.code}: ${tag.error.message}`);
			}
			const ingestLabel = percentage >= 100
					? (run.ingest?.state ?? "ready")
					: `${percentage}% · ${run.ingest?.state ?? "pending"}`;
			row.title = ingestLabel
					+ (issueMessages.length ? `\n${issueMessages.join("\n")}` : "");

			const chip = document.createElement("span");
			chip.className = "run-color";
			chip.style.background = runColorMap.get(runId);
			const name = document.createElement("span");
			name.className = "run-name";
			name.textContent = runId;
			row.append(chip, name);
			this._applyRunProgress(row, run.ingest);
			if (run.ingest?.error || quarantineCount > 0) {
				const warning = document.createElement("span");
				warning.className = "run-warning";
				warning.textContent = quarantineCount > 0 ? `⚠${quarantineCount}` : "⚠";
				row.append(warning);
			}
			list.append(row);
		}
	}

	updateRunProgress(runs) {
		const runsById = new Map(runs.map(run => [run.id, run]));
		let needsPolling = false;
		for (const row of document.querySelectorAll("#run-list .run-row")) {
			const run = runsById.get(row.dataset.runId);
			if (!run) continue;
			needsPolling = this._applyRunProgress(row, run.ingest) || needsPolling;
		}
		return needsPolling;
	}

	_applyRunProgress(row, ingest) {
		const percentage = this._ingestPercentage(ingest);
		const active = ["pending", "converting"].includes(ingest?.state);
		row.style.setProperty("--ingest-progress", percentage < 100 ? `${percentage}%` : "0%");

		let progress = row.querySelector(".run-progress");
		if (active && percentage < 100) {
			if (!progress) {
				progress = document.createElement("span");
				progress.className = "run-progress";
				const warning = row.querySelector(".run-warning");
				row.insertBefore(progress, warning);
			}
			progress.textContent = `${percentage}%`;
		} else {
			progress?.remove();
		}
		return active;
	}

	_ingestPercentage(ingest) {
		const rawPercentage = Number(ingest?.percentage ?? 0);
		return Number.isFinite(rawPercentage)
				? Math.max(0, Math.min(100, rawPercentage))
				: 0;
	}

	bindRunListEvents() {
		const list = document.getElementById("run-list");
		for (const row of list.querySelectorAll(".run-row")) {
			row.addEventListener("click", () => {
				const runId = row.dataset.runId;
				const now = performance.now();
				const solo = this.lastRunClick.runId === runId
						&& now - this.lastRunClick.at <= RUN_SOLO_INTERVAL_MS;
				this.lastRunClick = { runId, at: now };
				if (solo) {
					this.app.setSelectedRuns([runId]);
					return;
				}
				const selected = new Set(this.app.selectedRuns);
				if (selected.has(runId)) selected.delete(runId);
				else selected.add(runId);
				this.app.setSelectedRuns([...selected]);
			});
		}
		document.getElementById("btn-select-all-runs").onclick =
				() => this.app.setSelectedRuns(this.app.cache.getRunIds());
		document.getElementById("btn-latest-only").onclick = () => {
			const latest = this.app.cache.getRunIds().sort().at(-1);
			this.app.setSelectedRuns(latest ? [latest] : []);
		};
	}

	renderTagList(tagKeys) {
		const list = document.getElementById("tag-list");
		list.replaceChildren();
		for (const tagKey of tagKeys.slice().sort()) {
			const item = document.createElement("li");
			item.dataset.tagKey = tagKey;
			item.classList.toggle("active", this.app.activeTags.has(tagKey));
			if (this.app.isTagsLocked && !this.app.activeTags.has(tagKey)) item.hidden = true;
			const label = document.createElement("span");
			label.className = "tag-label";
			label.textContent = tagKey;
			item.append(label);
			const issue = this.app.issueForTag(tagKey);
			if (issue) {
				const warning = document.createElement("span");
				warning.className = "tag-warning";
				warning.textContent = " ⚠";
				warning.title = issue;
				item.append(warning);
			}
			list.append(item);
		}
	}

	bindTagListEvents() {
		const list = document.getElementById("tag-list");
		let hoverTimer = null;
		const clearHover = () => {
			if (hoverTimer) clearTimeout(hoverTimer);
			hoverTimer = null;
		};
		const startHover = item => {
			clearHover();
			if (HOVER_SCROLL_DELAY_MS <= 0 || !item.classList.contains("active")) return;
			hoverTimer = setTimeout(() => {
				document.getElementById(graphId(item.dataset.tagKey))
						?.closest(".graph-block")
						?.scrollIntoView({ behavior: "smooth", block: "start" });
			}, HOVER_SCROLL_DELAY_MS);
		};
		for (const item of list.querySelectorAll("li")) {
			item.addEventListener("click", () => {
				const tagKey = item.dataset.tagKey;
				if (this.app.activeTags.has(tagKey)) this.app.activeTags.delete(tagKey);
				else this.app.activeTags.add(tagKey);
				this.app.onTagSelectionChanged();
				if (this.app.activeTags.has(tagKey)) startHover(item);
				else clearHover();
			});
			item.addEventListener("mouseenter", () => startHover(item));
			item.addEventListener("mouseleave", clearHover);
		}
		document.getElementById("btn-select-all").onclick = () => {
			for (const item of list.querySelectorAll("li")) {
				this.app.activeTags.add(item.dataset.tagKey);
			}
			this.app.onTagSelectionChanged();
		};
		document.getElementById("btn-clear-all").onclick = () => {
			for (const item of list.querySelectorAll("li")) {
				this.app.activeTags.delete(item.dataset.tagKey);
			}
			this.app.onTagSelectionChanged();
		};
		const filter = document.getElementById("chk-lock-tags");
		filter.checked = this.app.isTagsLocked;
		filter.onchange = () => {
			this.app.isTagsLocked = filter.checked;
			this.app.refreshLists();
		};
		document.getElementById("btn-select-all").disabled = this.app.isTagsLocked;
		document.getElementById("btn-clear-all").disabled = this.app.isTagsLocked;
	}

	bindStaticControls() {
		document.getElementById("btn-reload").onclick = () => this.app.onReload();
		document.getElementById("btn-auto-reload").onclick = () => this.app.onToggleAutoReload();
		document.getElementById("btn-graph-scroll-lock").onclick =
				() => this.app.onToggleGraphScrollLock();
		document.getElementById("btn-reset-view").onclick =
				() => this.app.onResetView().catch(error => console.error(error));
		document.getElementById("btn-screenshot").onclick = () => this.app.onToggleScreenshot();
		document.getElementById("btn-screenshot-toggle").onclick =
				() => this.app.onToggleScreenshot();
		const lodMode = document.getElementById("lod-display-mode");
		lodMode.value = this.app.lodDisplayMode;
		lodMode.onchange = () => this.app.onLodDisplayModeChanged(lodMode.value);
		window.onresize = () => this.app.plotly.resizeAll();

		const mainArea = document.getElementById("main-area");
		mainArea.onclick = event => {
			if (event.detail !== 2) return;
			if (!(event.target instanceof Element)) return;
			if (event.target.closest("button,select") || !event.target.closest(".graph-block")) return;
			this.app.onReload();
		};
		mainArea.__mvGraphDblClickReloadHandler = mainArea.onclick;
		this.bindGraphScrollLockDrag(mainArea);
	}

	bindGraphScrollLockDrag(mainArea) {
		const state = { active: false, scrolling: false, startY: 0, lastY: 0, touchId: null };
		const reset = () => {
			state.active = false;
			state.scrolling = false;
			state.touchId = null;
		};
		const isGraphTarget = target => target instanceof Element
				&& Boolean(target.closest(".js-plotly-plot"))
				&& !target.closest(".modebar");
		const begin = (target, clientY, touchId = null) => {
			if (!this.app.graphScrollLockEnabled || !isGraphTarget(target)) return;
			state.active = true;
			state.startY = clientY;
			state.lastY = clientY;
			state.touchId = touchId;
		};
		const move = (event, clientY) => {
			if (!state.active || !this.app.graphScrollLockEnabled) return;
			const total = clientY - state.startY;
			if (!state.scrolling && Math.abs(total) < GRAPH_SCROLL_LOCK_DRAG_THRESHOLD_PX) return;
			state.scrolling = true;
			const delta = state.lastY - clientY;
			state.lastY = clientY;
			mainArea.scrollTop += delta;
			if (event.cancelable) event.preventDefault();
			event.stopPropagation();
		};
		mainArea.addEventListener("mousedown", event => {
			if (event.button === 0) begin(event.target, event.clientY);
		}, true);
		document.addEventListener("mousemove", event => {
			if ((event.buttons & 1) === 1) move(event, event.clientY);
		}, true);
		document.addEventListener("mouseup", reset, true);
		mainArea.addEventListener("touchstart", event => {
			if (event.touches.length !== 1) return;
			const touch = event.touches[0];
			begin(event.target, touch.clientY, touch.identifier);
		}, { capture: true, passive: true });
		document.addEventListener("touchmove", event => {
			for (const touch of event.touches) {
				if (touch.identifier === state.touchId) move(event, touch.clientY);
			}
		}, { capture: true, passive: false });
		document.addEventListener("touchend", reset, true);
		document.addEventListener("touchcancel", reset, true);
	}
}

class MetricsViewerClientApp {
	constructor() {
		this.fetcher = new DataFetcher();
		this.cache = new DataCache();
		this.plotly = new PlotlyController(this);
		this.ui = new UIController(this);
		this.mode = Mode.UNINITIALIZED;
		this.selectedRuns = [];
		this.activeTags = new Set();
		this.knownTags = new Set();
		this.logScaleTags = new Set();
		this.hiddenLegendSeries = new Map();
		this.plotDragModes = new Map();
		this.runColorMap = new Map();
		this.viewports = new Map();
		this.graphScrollLockEnabled = false;
		this.lodDisplayMode = LodDisplayMode.MIN_MAX;
		this.isTagsLocked = false;
		this.autoReloadEnabled = false;
		this.autoReloadTimer = null;
		this.ingestPollTimer = null;
		this.polling = false;
		this.initialSelectionApplied = false;
		this.queryRevision = 0;
		this.metadataRevision = 0;
		this.viewportDebounceTimer = null;
		this.updateFailures = { metadata: null, metrics: null };
	}

	async init() {
		this._loadState();
		this.ui.bindStaticControls();
		this._syncGraphScrollLockUi();
		this.setMode(Mode.META_LOADING);
		try {
			await this.refreshMetadata({ initial: true, requestData: false });
			this.setMode(Mode.NORMAL);
			await this.requestVisibleData({ force: true });
		} catch (error) {
			if (error.name !== "AbortError") {
				console.error(error);
				this.setMode(Mode.ERROR);
				Toast.show(`System error: ${error.message}`);
			}
		}
	}

	setMode(mode) {
		this.mode = mode;
		this.ui.applyMode(mode);
	}

	async refreshMetadata({ initial = false, requestData = true } = {}) {
		const revision = ++this.metadataRevision;
		let metadataSucceeded = false;
		this.ui.setLoadingSpinner(true);
		try {
			const payload = await this.fetcher.fetchRuns();
			if (revision !== this.metadataRevision) return;
			metadataSucceeded = true;
			this._setUpdateFailure("metadata", null);
			const runs = Array.isArray(payload?.runs) ? payload.runs : [];
			const previousSelection = this.selectedRuns.join("\u0000");
			const previousGenerations = new Map(
					this.selectedRuns.map(runId => [runId, this.cache.getRun(runId)?.generation]));
			this.cache.updateRuns(runs);
			const runIds = this.cache.getRunIds();
			for (const runId of this.runColorMap.keys()) {
				if (!runIds.includes(runId)) this.runColorMap.delete(runId);
			}
			if (!this.initialSelectionApplied && initial) {
				const latest = runIds.slice().sort().at(-1);
				this.selectedRuns = latest ? [latest] : [];
				this.initialSelectionApplied = true;
			} else {
				this.selectedRuns = this.selectedRuns.filter(runId => runIds.includes(runId));
			}
			const generationChanged = this.selectedRuns.some(runId =>
				previousGenerations.has(runId)
					&& previousGenerations.get(runId) !== this.cache.getRun(runId)?.generation);
			if (previousSelection !== this.selectedRuns.join("\u0000") || generationChanged) {
				this._bumpQueryRevision();
				this.fetcher.prioritize(this.selectedRuns).catch(error => console.warn(error));
			}

			this._activateNewVisibleTags();
			this.refreshLists();
			this._renderCurrent();
			this._syncIngestPoller();
			if (requestData) await this.requestVisibleData();
		} catch (error) {
			if (!metadataSucceeded && revision === this.metadataRevision) {
				this._setUpdateFailure("metadata", error);
			}
			throw error;
		} finally {
			if (revision === this.metadataRevision) this.ui.setLoadingSpinner(false);
		}
	}

	refreshLists() {
		this.ui.renderRunList(this.cache.getRuns(), this.selectedRuns, this.runColorMap);
		this.ui.bindRunListEvents();
		this.ui.renderTagList([...this._visibleTagSet()]);
		this.ui.bindTagListEvents();
	}

	setSelectedRuns(runIds) {
		this.selectedRuns = [...new Set(runIds)].filter(runId => this.cache.getRun(runId));
		this._bumpQueryRevision();
		this._activateNewVisibleTags();
		this.refreshLists();
		this._renderCurrent();
		this.fetcher.prioritize(this.selectedRuns).catch(error => console.warn(error));
		this.requestVisibleData().catch(error => this._handleQueryError(error));
	}

	onTagSelectionChanged() {
		this._saveSets();
		this._bumpQueryRevision();
		this.refreshLists();
		this._renderCurrent();
		this.requestVisibleData().catch(error => this._handleQueryError(error));
	}

	onViewportChanged(tagKey, range, autorange) {
		if (autorange || !Array.isArray(range)) {
			this.viewports.set(tagKey, { autorange: true });
		} else {
			const from = Math.min(Number(range[0]), Number(range[1]));
			const to = Math.max(Number(range[0]), Number(range[1]));
			this.viewports.set(tagKey, { autorange: false, from, to });
		}
		this._bumpQueryRevision();
		clearTimeout(this.viewportDebounceTimer);
		this.viewportDebounceTimer = setTimeout(() => {
			this.requestVisibleData().catch(error => this._handleQueryError(error));
		}, VIEWPORT_DEBOUNCE_MS);
	}

	onToggleLog(tagKey) {
		if (this.logScaleTags.has(tagKey)) this.logScaleTags.delete(tagKey);
		else this.logScaleTags.add(tagKey);
		this._renderCurrent();
	}

	onLodDisplayModeChanged(mode) {
		if (!Object.values(LodDisplayMode).includes(mode)) return;
		this.lodDisplayMode = mode;
		localStorage.setItem(STORAGE_KEY_LOD_MODE, mode);
		this._renderCurrent();
	}

	async requestVisibleData({ force = false, eligibleRunIds = null, followOnly = false } = {}) {
		const revision = this.queryRevision;
		const series = [];
		const requestContext = [];
		for (const tagKey of this._visibleSelectedTags()) {
			const viewport = this._viewportFor(tagKey);
			if (!viewport) continue;
			const target = this._windowFor(viewport);
			for (const runId of this.selectedRuns) {
				const run = this.cache.getRun(runId);
				if (!run || !this.cache.getTag(runId, tagKey)) continue;
				if (eligibleRunIds && !eligibleRunIds.has(runId)) continue;
				if (followOnly && !this._isFollowing(tagKey, viewport)) continue;
				if (!this.cache.needsFetch(runId, tagKey, target, viewport, force)) continue;
				series.push({
					runId,
					tagKey,
					fromStep: target.fromStep,
					toStep: target.toStep
				});
				requestContext.push({
					runId,
					tagKey,
					generation: run.generation,
					viewportWidth: viewport.width
				});
			}
		}
		if (!series.length) return;

		this.ui.setLoadingSpinner(true);
		try {
			const payload = await this.fetcher.fetchMetrics(series);
			if (revision !== this.queryRevision) return;
			this._setUpdateFailure("metrics", null);
			const contextBySeries = new Map(requestContext.map(expected => [
				`${expected.runId}\u0000${expected.tagKey}`,
				expected
			]));
			for (const result of payload.data ?? []) {
				const expected = contextBySeries.get(`${result.runId}\u0000${result.tagKey}`);
				if (!expected) continue;
				const run = this.cache.getRun(result.runId);
				if (!run
						|| !this.selectedRuns.includes(result.runId)
						|| !this.activeTags.has(result.tagKey)
						|| run.generation !== expected.generation
						|| result.generation !== expected.generation) {
					continue;
				}
				this.cache.replaceWindow(result, expected.viewportWidth);
			}
			this._renderCurrent();
		} catch (error) {
			if (revision === this.queryRevision) this._setUpdateFailure("metrics", error);
			throw error;
		} finally {
			if (revision === this.queryRevision) this.ui.setLoadingSpinner(false);
		}
	}

	_viewportFor(tagKey) {
		const explicit = this.viewports.get(tagKey);
		if (explicit && !explicit.autorange) {
			return {
				from: explicit.from,
				to: explicit.to,
				width: Math.max(1, explicit.to - explicit.from),
				autorange: false
			};
		}
		let min = Infinity;
		let max = -Infinity;
		for (const runId of this.selectedRuns) {
			const stats = this.cache.getTag(runId, tagKey)?.stats;
			if (!stats) continue;
			min = Math.min(min, Number(stats.minStep));
			max = Math.max(max, Number(stats.maxStep));
		}
		if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
		return { from: min, to: max, width: Math.max(1, max - min), autorange: true };
	}

	_windowFor(viewport) {
		return {
			fromStep: clampSafeStep(Math.floor(viewport.from - viewport.width)),
			toStep: clampSafeStep(Math.ceil(viewport.to + viewport.width))
		};
	}

	explicitViewport(tagKey) {
		const viewport = this.viewports.get(tagKey);
		if (!viewport || viewport.autorange) return null;
		return { range: [viewport.from, viewport.to] };
	}

	_isFollowing(tagKey, viewport = this._viewportFor(tagKey)) {
		if (!viewport) return false;
		if (viewport.autorange) return true;
		let latest = -Infinity;
		for (const runId of this.selectedRuns) {
			const tag = this.cache.getTag(runId, tagKey);
			if (tag?.stats) latest = Math.max(latest, Number(tag.stats.maxStep));
		}
		return Number.isFinite(latest)
				&& Math.abs(latest - viewport.to) <= viewport.width * 0.05;
	}

	_visibleTagSet() {
		const tags = new Set();
		for (const runId of this.selectedRuns) {
			for (const tagKey of this.cache.getTagKeys(runId)) tags.add(tagKey);
		}
		return tags;
	}

	_visibleSelectedTags() {
		const visible = this._visibleTagSet();
		return [...this.activeTags].filter(tag => visible.has(tag)).sort();
	}

	_activateNewVisibleTags() {
		let changed = false;
		for (const tagKey of this._visibleTagSet()) {
			if (!this.knownTags.has(tagKey)) {
				this.knownTags.add(tagKey);
				this.activeTags.add(tagKey);
				changed = true;
			}
		}
		if (changed) this._saveSets();
	}

	combinedStats(tagKey) {
		let count = 0;
		let mean = 0;
		let m2 = 0;
		let min = Infinity;
		let max = -Infinity;
		for (const runId of this.selectedRuns) {
			const stats = this.cache.getTag(runId, tagKey)?.stats;
			if (!stats || Number(stats.count) <= 0) continue;
			const nextCount = Number(stats.count);
			const nextMean = Number(stats.mean);
			const nextM2 = Number(stats.variance) * nextCount;
			if (count === 0) {
				count = nextCount;
				mean = nextMean;
				m2 = nextM2;
			} else {
				const delta = nextMean - mean;
				const combinedCount = count + nextCount;
				mean += delta * nextCount / combinedCount;
				m2 += nextM2 + delta * delta * count * nextCount / combinedCount;
				count = combinedCount;
			}
			min = Math.min(min, Number(stats.minValue));
			max = Math.max(max, Number(stats.maxValue));
		}
		if (count === 0) return null;
		return { count, min, max, mean, stdDev: Math.sqrt(Math.max(0, m2 / count)) };
	}

	issueForTag(tagKey) {
		const issues = [];
		for (const runId of this.selectedRuns) {
			const run = this.cache.getRun(runId);
			if (run?.ingest?.error) {
				issues.push(`${runId}: ${run.ingest.error.code}: ${run.ingest.error.message}`);
			}
			const tag = this.cache.getTag(runId, tagKey);
			if (tag?.error) issues.push(`${runId}: ${tag.error.code}: ${tag.error.message}`);
			for (const issue of this.cache.getWindow(runId, tagKey)?.issues ?? []) {
				issues.push(`${runId}: ${issue.code}: ${issue.message}`);
			}
		}
		return issues.length ? issues.join("\n") : null;
	}

	isLegendSeriesHidden(tagKey, runId) {
		return this.hiddenLegendSeries.get(tagKey)?.has(runId) ?? false;
	}

	setLegendSeriesHidden(tagKey, runId, hidden) {
		if (!hidden) {
			const runIds = this.hiddenLegendSeries.get(tagKey);
			runIds?.delete(runId);
			if (runIds?.size === 0) this.hiddenLegendSeries.delete(tagKey);
			return;
		}
		let runIds = this.hiddenLegendSeries.get(tagKey);
		if (!runIds) {
			runIds = new Set();
			this.hiddenLegendSeries.set(tagKey, runIds);
		}
		runIds.add(runId);
	}

	pruneLegendVisibility(runIds, tagKeys) {
		const selectedRuns = new Set(runIds);
		const selectedTags = new Set(tagKeys);
		for (const [tagKey, hiddenRuns] of this.hiddenLegendSeries) {
			if (!selectedTags.has(tagKey)) {
				this.hiddenLegendSeries.delete(tagKey);
				continue;
			}
			for (const runId of hiddenRuns) {
				if (!selectedRuns.has(runId)) hiddenRuns.delete(runId);
			}
			if (hiddenRuns.size === 0) this.hiddenLegendSeries.delete(tagKey);
		}
	}

	plotDragMode(tagKey) {
		return this.plotDragModes.get(tagKey);
	}

	setPlotDragMode(tagKey, dragMode) {
		this.plotDragModes.set(tagKey, dragMode);
	}

	prunePlotDragModes(tagKeys) {
		const selectedTags = new Set(tagKeys);
		for (const tagKey of this.plotDragModes.keys()) {
			if (!selectedTags.has(tagKey)) this.plotDragModes.delete(tagKey);
		}
	}

	async onResetView() {
		this.hiddenLegendSeries.clear();
		await this.plotly.resetView();
	}

	_renderCurrent() {
		const main = document.getElementById("main-area");
		const scrollTop = main.scrollTop;
		this.plotly.renderBySelection(
				"#main-area",
				this.selectedRuns.slice(),
				this._visibleSelectedTags(),
				this.cache);
		main.scrollTop = scrollTop;
		this._syncGraphScrollLockUi();
	}

	_bumpQueryRevision() {
		this.queryRevision++;
		this.fetcher.abortMetrics();
	}

	_syncIngestPoller() {
		const needsPolling = this.cache.getRunIds().some(runId =>
			["pending", "converting"].includes(this.cache.getRun(runId)?.ingest?.state));
		if (needsPolling && !this.ingestPollTimer) {
			this.ingestPollTimer = setInterval(() => this._pollIngest(), INGEST_POLL_INTERVAL_MS);
		} else if (!needsPolling && this.ingestPollTimer) {
			clearInterval(this.ingestPollTimer);
			this.ingestPollTimer = null;
		}
	}

	async _pollIngest() {
		if (this.polling || this.mode === Mode.SCREENSHOT) return;
		this.polling = true;
		const metadataRevision = this.metadataRevision;
		try {
			const payload = await this.fetcher.fetchIngestProgress();
			if (metadataRevision !== this.metadataRevision) return;
			this._setUpdateFailure("metadata", null);
			const runs = Array.isArray(payload?.runs) ? payload.runs : [];
			const needsPolling = this.ui.updateRunProgress(runs);
			if (!needsPolling && this.ingestPollTimer) {
				clearInterval(this.ingestPollTimer);
				this.ingestPollTimer = null;
			}
		} catch (error) {
			if (metadataRevision === this.metadataRevision) {
				this._setUpdateFailure("metadata", error);
			}
			this._handleQueryError(error);
		} finally {
			this.polling = false;
		}
	}

	onToggleGraphScrollLock() {
		this.graphScrollLockEnabled = !this.graphScrollLockEnabled;
		localStorage.setItem(
				STORAGE_KEY_GRAPH_SCROLL_LOCK,
				this.graphScrollLockEnabled ? "true" : "false");
		this._syncGraphScrollLockUi();
	}

	_syncGraphScrollLockUi() {
		document.body.classList.toggle("graph-scroll-locked", this.graphScrollLockEnabled);
		const button = document.getElementById("btn-graph-scroll-lock");
		if (button) {
			button.textContent = this.graphScrollLockEnabled
					? "Scroll Lock: ON"
					: "Scroll Lock: OFF";
			button.classList.toggle("active", this.graphScrollLockEnabled);
			button.setAttribute("aria-pressed", this.graphScrollLockEnabled ? "true" : "false");
		}
		this.plotly.applyGraphScrollLock(this.graphScrollLockEnabled);
	}

	onToggleAutoReload() {
		this.autoReloadEnabled = !this.autoReloadEnabled;
		const button = document.getElementById("btn-auto-reload");
		button.textContent = this.autoReloadEnabled ? "Auto Reload: ON" : "Auto Reload: OFF";
		button.classList.toggle("active", this.autoReloadEnabled);
		button.setAttribute("aria-pressed", this.autoReloadEnabled ? "true" : "false");
		if (this.autoReloadEnabled) {
			this.autoReloadTimer = setInterval(async () => {
				try {
					await this.refreshMetadata({ requestData: false });
					await this.requestVisibleData({ force: true, followOnly: true });
				} catch (error) {
					this._handleQueryError(error);
				}
			}, AUTO_RELOAD_INTERVAL_MS);
		} else {
			clearInterval(this.autoReloadTimer);
			this.autoReloadTimer = null;
		}
	}

	async onReload() {
		if (this.mode === Mode.SCREENSHOT) return;
		const recoveringFromInitialError = this.mode === Mode.ERROR;
		try {
			await this.refreshMetadata({
				initial: recoveringFromInitialError,
				requestData: false
			});
			await this.requestVisibleData({ force: true });
			if (recoveringFromInitialError) this.setMode(Mode.NORMAL);
		} catch (error) {
			this._handleQueryError(error);
			Toast.show("Reload failed.");
		}
	}

	onToggleScreenshot() {
		if (this.mode === Mode.ERROR) return;
		const enabled = document.body.classList.toggle("screenshot-mode");
		document.documentElement.classList.toggle("screenshot-mode", enabled);
		this.setMode(enabled ? Mode.SCREENSHOT : Mode.NORMAL);
		document.getElementById("btn-screenshot-toggle").textContent = enabled ? "➡" : "⬅";
		const header = document.getElementById("screenshot-header");
		header.textContent = this.selectedRuns.length === 1
				? `Metrics Viewer — ${this.selectedRuns[0]}`
				: "Metrics Viewer";
		header.style.display = enabled ? "block" : "none";
		setTimeout(() => this.plotly.resizeAll(), 300);
	}

	_loadState() {
		this.activeTags = this._loadSet(STORAGE_KEY_TAGS);
		this.knownTags = this._loadSet(STORAGE_KEY_KNOWN_TAGS);
		this.graphScrollLockEnabled =
				localStorage.getItem(STORAGE_KEY_GRAPH_SCROLL_LOCK) === "true";
		const storedMode = localStorage.getItem(STORAGE_KEY_LOD_MODE);
		if (Object.values(LodDisplayMode).includes(storedMode)) this.lodDisplayMode = storedMode;
	}

	_loadSet(key) {
		try {
			const stored = localStorage.getItem(key);
			return stored ? new Set(JSON.parse(stored)) : new Set();
		} catch (error) {
			console.warn(`Failed to load ${key}`, error);
			return new Set();
		}
	}

	_saveSets() {
		localStorage.setItem(STORAGE_KEY_TAGS, JSON.stringify([...this.activeTags]));
		localStorage.setItem(STORAGE_KEY_KNOWN_TAGS, JSON.stringify([...this.knownTags]));
	}

	_setUpdateFailure(kind, error) {
		if (error?.name === "AbortError") return;
		this.updateFailures[kind] = error ? (error.message ?? String(error)) : null;
		this.ui.renderUpdateStatus(this.updateFailures);
	}

	_handleQueryError(error) {
		if (error?.name === "AbortError") return;
		console.error(error);
	}
}

let app = null;
window.addEventListener("load", () => {
	app = new MetricsViewerClientApp();
	app.init();
});
