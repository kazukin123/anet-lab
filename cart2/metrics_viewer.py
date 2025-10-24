#!/usr/bin/env python3
"""
metrics_viewer.py  (v32)
----------------------------------------
Dash4 安定版
 - CSSは assets/custom.css に分離
 - Dashが自動読み込み
 - 自己リロード (--watch)
----------------------------------------
"""

import os, json, re, time, sys, subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, dcc, html, Input, Output, State

VIEWER_VERSION = "v32"
RUN_CACHE = {}
RUN_COLORS = {}


def get_run_color(run_name):
    if run_name not in RUN_COLORS:
        palette = pc.qualitative.Plotly
        RUN_COLORS[run_name] = palette[len(RUN_COLORS) % len(palette)]
    return RUN_COLORS[run_name]


def read_incremental_jsonl(jsonl_path):
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
    pq.write_table(pa.Table.from_pandas(df_all), parquet_path)
    RUN_CACHE[run_name] = {"mtime": mtime, "pos": new_pos, "df": df_all}
    return df_all


def load_selected_runs(root, runs):
    out = {}
    for r in runs:
        p = os.path.join(root, r, "metrics.jsonl")
        df = read_incremental_jsonl(p)
        if not df.empty:
            out[r] = df
    return out


def extract_tags(run_data):
    tags = set()
    for df in run_data.values():
        tags |= set(df["tag"].unique())
    return sorted(tags)


def detect_axis_column(df, tag):
    return "episode" if tag.startswith("episode/") and "episode" in df.columns else "step"


def make_tag_fig(run_data, selected_runs, tag):
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
            x, y = sub[axis_col].iloc[0], sub["value"].iloc[0]
            sub = pd.DataFrame([{axis_col: x-0.5, "value": y}, {axis_col: x+0.5, "value": y}])
            fig.add_trace(go.Scatter(
                x=sub[axis_col], y=sub["value"],
                mode="lines+markers", name=run,
                line=dict(width=2, dash="dot", color=run_color),
                marker=dict(size=7, color=run_color)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=sub[axis_col], y=sub["value"], mode="lines",
                name=run, line=dict(color=run_color, width=2)
            ))
    fig.update_layout(template="plotly_dark", height=300,
                      margin=dict(l=40, r=20, t=20, b=40),
                      showlegend=len(selected_runs) > 1)
    return fig


def create_app(log_root):
    app = Dash(__name__, title="Metrics Viewer")

    app.layout = html.Div([
        html.Div([
            html.H2("📊 Metrics Viewer",
                    style={"marginTop": "0", "marginBottom": "10px", "color": "#fff"}),

            html.Div([
                html.Label("Runs", style={"marginRight": "6px", "color": "#eee"}),
                dcc.Dropdown(
                    id="run-select", options=[], value=[],
                    multi=True, placeholder="Select runs",
                    style={"minWidth": "150px", "maxWidth": "250px",
                           "flex": "0 1 auto", "marginRight": "12px"}
                ),
                html.Label("Tags", style={"marginRight": "6px", "color": "#eee"}),
                dcc.Dropdown(
                    id="tag-filter", options=[], value=[],
                    multi=True, placeholder="All tags",
                    style={"minWidth": "200px", "maxWidth": "300px", "flex": "0 1 auto"}
                )
            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "10px", "flexWrap": "wrap"}),

            html.Div([
                html.Button("手動更新", id="refresh-btn", n_clicks=0,
                            style={"marginRight": "10px", "boxShadow": "0 0 3px #000"}),
                html.Button("自動更新: OFF", id="toggle-auto", n_clicks=0,
                            style={"boxShadow": "0 0 3px #000"})
            ])
        ], style={
            "backgroundColor": "#2d2d2d", "border": "1px solid #555",
            "borderRadius": "8px", "padding": "12px 16px 14px 16px",
            "marginBottom": "18px", "boxShadow": "0 0 8px #0008"
        }),

        dcc.Store(id="auto-flag", data=False),
        html.Div(id="graphs-container"),
        dcc.Interval(id="tick", interval=5000, n_intervals=0, disabled=True),

        html.Div(f"Viewer {VIEWER_VERSION}", style={
            "position": "fixed", "top": "8px", "right": "12px",
            "backgroundColor": "#000", "color": "#fff", "fontSize": "13px",
            "padding": "3px 8px", "borderRadius": "4px", "zIndex": 9999,
            "fontFamily": "monospace", "fontWeight": "bold", "boxShadow": "0 0 4px #000"
        })
    ])

    @app.callback(
        Output("tick", "disabled"),
        Output("toggle-auto", "children"),
        Output("auto-flag", "data"),
        Input("toggle-auto", "n_clicks"),
        State("auto-flag", "data"),
        prevent_initial_call=True
    )
    def toggle_auto(n_clicks, current):
        new_flag = not current
        return (not new_flag, f"自動更新: {'ON' if new_flag else 'OFF'}", new_flag)

    @app.callback(
        Output("graphs-container", "children"),
        Output("run-select", "options"),
        Output("run-select", "value"),
        Output("tag-filter", "options"),
        Input("tick", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
        Input("run-select", "value"),
        State("tag-filter", "value"),
        prevent_initial_call=False
    )
    def update_graphs(n_auto, n_clicks, selected_runs, selected_tags):
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
        display_tags = selected_tags or all_tags
        graphs = []
        for tag in display_tags:
            fig = make_tag_fig(run_data, selected_runs, tag)
            tag_label = html.Div(tag, style={
                "color": "#fff", "fontSize": "14px", "fontWeight": "bold",
                "position": "absolute", "top": "2px", "left": "6px",
                "zIndex": "10", "backgroundColor": "rgba(0,0,0,0.5)",
                "padding": "1px 4px", "borderRadius": "3px"
            })
            graphs.append(html.Div([
                tag_label,
                dcc.Graph(figure=fig,
                          config={"displayModeBar": True, "responsive": False},
                          style={"height": "300px"})
            ], style={"position": "relative", "marginBottom": "12px"}))
        return graphs, [{"label": r, "value": r} for r in run_names], selected_runs, \
               [{"label": t, "value": t} for t in all_tags]

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if not args.watch:
        print(f"[INFO] Starting viewer {VIEWER_VERSION} — {args.logdir}")
        app = create_app(args.logdir)
        app.run(debug=False)
    else:
        print(f"[INFO] Starting viewer {VIEWER_VERSION} in WATCH mode — {args.logdir}")
        target = os.path.abspath(__file__)
        last_mtime = os.path.getmtime(target)
        proc = subprocess.Popen([sys.executable, target, "--logdir", args.logdir])
        try:
            while True:
                time.sleep(1)
                new_mtime = os.path.getmtime(target)
                if new_mtime != last_mtime:
                    print("[WATCH] Detected change. Restarting viewer...")
                    last_mtime = new_mtime
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    proc = subprocess.Popen([sys.executable, target, "--logdir", args.logdir])
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt. Exiting watch mode...")
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("[INFO] Done.")
