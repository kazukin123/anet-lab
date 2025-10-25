#!/usr/bin/env python3
"""
metrics_viewer_dash_full_v13.py
---------------------------------------
- JSONL差分読込＋Parquetキャッシュ
- 自動/手動更新・タグフィルタ
- 1点グラフ対応・ズーム保持
- type=json の全Meta情報をページ末尾に表示
- Meta情報にtimestamp併記＋Runごとの色同期
- グラフとMetaのRun色統一（Plotlyパレット）
---------------------------------------
"""

import os, json, re, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, dcc, html, Input, Output, State
from dash.dependencies import ALL

RUN_CACHE = {}
RUN_COLORS = {}


def get_run_color(run_name):
    """Run名に基づき一貫した色を返す"""
    if run_name not in RUN_COLORS:
        palette = pc.qualitative.Plotly
        idx = len(RUN_COLORS) % len(palette)
        RUN_COLORS[run_name] = palette[idx]
    return RUN_COLORS[run_name]


def read_incremental_jsonl(jsonl_path: str):
    """metrics.jsonl の差分を読み込み、Parquetにキャッシュ"""
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()
    run_dir = os.path.dirname(jsonl_path)
    run_name = os.path.basename(run_dir)
    parquet_path = os.path.join(run_dir, "metrics_cache.parquet")
    mtime = os.path.getmtime(jsonl_path)
    cached = RUN_CACHE.get(run_name)
    last_pos = cached["pos"] if cached else 0
    df_existing = cached["df"] if cached else pd.DataFrame()
    if cached and cached["mtime"] == mtime:
        return df_existing

    new_records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        f.seek(last_pos)
        for line in f:
            try:
                j = json.loads(line)
                if j.get("type") == "scalar":
                    new_records.append(j)
            except json.JSONDecodeError:
                continue
        new_pos = f.tell()

    if not new_records and cached:
        return df_existing

    df_new = pd.DataFrame(new_records)
    if not df_new.empty:
        mask = df_new["tag"].str.startswith("episode/")
        if mask.any():
            df_new.loc[mask, "episode"] = df_new.loc[mask, "step"]

    df_all = pd.concat([df_existing, df_new], ignore_index=True)
    df_all.drop_duplicates(subset=["step", "tag"], inplace=True)

    table = pa.Table.from_pandas(df_all)
    pq.write_table(table, parquet_path)
    RUN_CACHE[run_name] = {"mtime": mtime, "pos": new_pos, "df": df_all}
    return df_all


def load_selected_runs(root, selected_runs):
    runs = {}
    for run_name in selected_runs:
        jsonl_path = os.path.join(root, run_name, "metrics.jsonl")
        df = read_incremental_jsonl(jsonl_path)
        if not df.empty:
            runs[run_name] = df
    return runs


def extract_tags(run_data):
    tags = set()
    for df in run_data.values():
        tags |= set(df["tag"].unique())
    return sorted(tags)


def detect_axis_column(df, tag):
    return "episode" if tag.startswith("episode/") and "episode" in df.columns else "step"


def safe_tag(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", tag)


def make_tag_fig(run_data, selected_runs, tag, zoom_state):
    fig = go.Figure()
    MAX_POINTS = 2000
    for run, df in run_data.items():
        if run not in selected_runs:
            continue
        run_color = get_run_color(run)
        axis_col = detect_axis_column(df, tag)
        sub = df[df["tag"] == tag]
        if len(sub) > MAX_POINTS:
            sub = sub.iloc[::max(1, len(sub)//MAX_POINTS)]
        if sub.empty:
            continue

        if len(sub) == 1:
            xval = sub[axis_col].iloc[0]
            yval = sub["value"].iloc[0]
            sub = pd.DataFrame([
                {axis_col: xval - 0.5, "value": yval, "tag": tag},
                {axis_col: xval + 0.5, "value": yval, "tag": tag}
            ])
            fig.add_trace(go.Scatter(
                x=sub[axis_col],
                y=sub["value"],
                mode="lines+markers",
                name=run,
                line=dict(width=2, dash="dot", color=run_color),
                marker=dict(size=7, color=run_color, symbol="circle")
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sub[axis_col],
                y=sub["value"],
                mode="lines",
                name=run,
                line=dict(color=run_color, width=2)
            ))

    fig.update_layout(
        template="plotly_dark",
        title=tag,
        xaxis_title=axis_col,
        yaxis_title=tag,
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=len(selected_runs) > 1
    )

    if zoom_state and tag in zoom_state:
        zr = zoom_state[tag]
        if zr.get("xrange"):
            fig.update_xaxes(range=zr["xrange"])
        if zr.get("yrange"):
            fig.update_yaxes(range=zr["yrange"])

    return fig


def load_json_meta_for_run(run_dir):
    """type=json の全エントリを抽出"""
    metas = []
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(jsonl_path):
        return []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if '"type":"json"' not in line:
                continue
            try:
                j = json.loads(line)
                metas.append({
                    "tag": j.get("tag", ""),
                    "timestamp": j.get("timestamp", ""),
                    "data": j.get("data", {})
                })
            except json.JSONDecodeError:
                continue
    return metas


def create_app(log_root):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H2("📊 Metrics Viewer — Multi-run with Color Sync"),

        html.Div([
            html.Label("Select runs:"),
            dcc.Dropdown(
                id="run-select",
                options=[], value=[],
                multi=True,
                placeholder="Select runs",
                style={"width": "400px", "marginBottom": "10px"}
            ),
        ]),

        html.Div([
            html.Button("手動更新", id="refresh-btn", n_clicks=0, style={"marginRight": "10px"}),
            html.Button("自動更新: OFF", id="toggle-auto", n_clicks=0),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Label("表示タグフィルタ:"),
            dcc.Dropdown(
                id="tag-filter",
                options=[], value=[],
                multi=True,
                placeholder="（未指定なら全タグ表示）",
                style={"width": "500px", "marginBottom": "15px"}
            ),
        ]),

        dcc.Store(id="zoom-store", data={}),
        dcc.Store(id="auto-flag", data=False),
        html.Div(id="graphs-container"),
        html.Div(id="dummy-output", style={"display": "none"}),
        dcc.Interval(id="tick", interval=5000, n_intervals=0, disabled=True)
    ])

    @app.callback(
        Output("tick", "disabled"),
        Output("toggle-auto", "children"),
        Output("auto-flag", "data"),
        Input("toggle-auto", "n_clicks"),
        State("auto-flag", "data"),
        prevent_initial_call=True
    )
    def toggle_auto(n_clicks, current_flag):
        new_flag = not current_flag
        label = "自動更新: ON" if new_flag else "自動更新: OFF"
        return (not new_flag, label, new_flag)

    @app.callback(
        Output("graphs-container", "children"),
        Output("run-select", "options"),
        Output("run-select", "value"),
        Output("tag-filter", "options"),
        Input("tick", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
        Input("run-select", "value"),
        State("tag-filter", "value"),
        State("zoom-store", "data"),
        prevent_initial_call=False
    )
    def update_graphs(n_auto, n_refresh, selected_runs, selected_tags, zoom_state):
        run_names = sorted([d for d in os.listdir(log_root)
                            if os.path.isdir(os.path.join(log_root, d))])
        if not run_names:
            return [html.Div("No runs found.", style={"color": "gray"})], [], [], []

        latest_run = run_names[-1]
        if not selected_runs:
            selected_runs = [latest_run]

        run_data = load_selected_runs(log_root, selected_runs)
        if not run_data:
            return [html.Div("No data found.", style={"color": "gray"})], \
                   [{"label": r, "value": r} for r in run_names], selected_runs, []

        all_tags = extract_tags(run_data)
        if not all_tags:
            return [html.Div("No tags found.", style={"color": "gray"})], \
                   [{"label": r, "value": r} for r in run_names], selected_runs, []

        display_tags = selected_tags or all_tags

        graphs = []
        for tag in all_tags:
            if tag not in display_tags:
                continue
            safe_id = safe_tag(tag)
            fig = make_tag_fig(run_data, selected_runs, tag, zoom_state)
            graphs.append(
                dcc.Graph(
                    id={"type": "metric-graph", "tag": safe_id},
                    figure=fig,
                    config={"displayModeBar": True, "responsive": False},
                    style={"height": "300px", "minHeight": "300px"}
                )
            )

        # ---- Meta情報（末尾） ----
        meta_blocks = []
        for run in selected_runs:
            metas = load_json_meta_for_run(os.path.join(log_root, run))
            if not metas:
                continue
            run_color = get_run_color(run)
            run_header = html.Div(
                [
                    html.Div(style={
                        "backgroundColor": run_color,
                        "width": "12px",
                        "height": "100%",
                        "float": "left",
                        "marginRight": "8px",
                        "borderTopLeftRadius": "6px",
                        "borderBottomLeftRadius": "6px"
                    }),
                    html.Span(f"Run: {run}", style={
                        "color": "#fff",
                        "fontWeight": "bold",
                        "fontSize": "16px",
                    }),
                ],
                style={
                    "backgroundColor": "#222",
                    "padding": "6px 10px",
                    "borderTopLeftRadius": "6px",
                    "borderTopRightRadius": "6px",
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                }
            )

            run_container = []
            for meta in metas:
                tag = meta["tag"]
                ts = meta.get("timestamp", "")
                data = meta["data"]

                header_line = html.Div([
                    html.Span(f"Tag: {tag}", style={
                        "color": run_color,
                        "fontWeight": "bold",
                        "marginRight": "10px"
                    }),
                    html.Span(f"({ts})", style={
                        "color": "#aaa",
                        "fontSize": "12px"
                    })
                ], style={"marginBottom": "4px", "marginTop": "6px"})

                rows = [
                    html.Tr([
                        html.Th(k, style={"textAlign": "left", "paddingRight": "10px", "color": "#fff"}),
                        html.Td(str(v), style={"textAlign": "left", "color": "#fff"})
                    ]) for k, v in data.items()
                ]

                table = html.Table(rows, style={
                    "borderCollapse": "collapse",
                    "border": f"1px solid {run_color}",
                    "marginBottom": "8px",
                    "width": "auto",
                    "backgroundColor": "#111",
                    "fontSize": "14px",
                    "wordBreak": "break-all",
                    "padding": "4px 8px"
                })

                run_container.append(html.Div([header_line, table],
                                              style={"marginLeft": "16px", "marginBottom": "10px"}))

            meta_blocks.append(html.Div(
                [run_header] + run_container,
                style={
                    "border": f"1px solid {run_color}",
                    "borderRadius": "6px",
                    "padding": "10px",
                    "marginBottom": "24px",
                    "backgroundColor": "#1a1a1a",
                    "overflow": "hidden",
                    "boxShadow": f"0 0 6px {run_color}66"
                }
            ))

        run_options = [{"label": r, "value": r} for r in run_names]
        tag_options = [{"label": t, "value": t} for t in all_tags]
        return graphs + meta_blocks, run_options, selected_runs, tag_options

    @app.callback(
        Output("zoom-store", "data"),
        Input({"type": "metric-graph", "tag": ALL}, "relayoutData"),
        State("zoom-store", "data"),
        prevent_initial_call=True
    )
    def store_zoom(relayout_list, store):
        store = store or {}
        for idx, relayout in enumerate(relayout_list):
            if not relayout:
                continue
            xr = [relayout.get("xaxis.range[0]"), relayout.get("xaxis.range[1]")]
            yr = [relayout.get("yaxis.range[0]"), relayout.get("yaxis.range[1]")]
            tag_key = f"tag_{idx}"
            store[tag_key] = {"xrange": xr, "yrange": yr}
        return store

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()
    print(f"[INFO] Starting viewer — {args.logdir}")
    app = create_app(args.logdir)
    app.run(debug=False)
