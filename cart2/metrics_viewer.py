#!/usr/bin/env python3
# metrics_viewer.py
# ---------------------------------
# Metrics Viewer
#   ・metrics_logger.hpp が出力する JSONL ログを読み込み
#   ・各 run ごとに scalar 値を Plotly グラフで可視化
#   ・複数 run / tag の選択、手動・自動更新に対応
#   ・type=="json" のメタ情報は下部に別表示
#   ・Runs ドロップダウンに色付きインジケータを左側に表示（全Run共通）
#   ・ヘッダ固定レイアウト、スクロール領域分離
# ---------------------------------

import os, sys, json, time, subprocess, argparse
import pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, dcc, html, Input, Output, State

RUN_CACHE = {}
RUN_COLORS = {}
VERSION = "v17.6"


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
    mtime = os.path.getmtime(jsonl_path)
    cached = RUN_CACHE.get(run_name)
    if cached and cached["mtime"] == mtime:
        return cached["df"]

    df_new = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                if j.get("type") in ("scalar", "json"):
                    df_new.append(j)
            except json.JSONDecodeError:
                continue

    if not df_new:
        return pd.DataFrame()
    df = pd.DataFrame(df_new)
    mask = df["tag"].str.startswith("episode/")
    if mask.any():
        df.loc[mask, "episode"] = df.loc[mask, "step"]
    pq.write_table(pa.Table.from_pandas(df), os.path.join(run_dir, "metrics_cache.parquet"))
    RUN_CACHE[run_name] = {"mtime": mtime, "df": df}
    return df


def load_selected_runs(root, selected_runs):
    out = {}
    for r in selected_runs:
        path = os.path.join(root, r, "metrics.jsonl")
        df = read_incremental_jsonl(path)
        if not df.empty:
            out[r] = df
    return out


def extract_tags(run_data):
    tags = set()
    for df in run_data.values():
        # type=="json" を除外
        if "type" in df.columns and "tag" in df.columns:
            valid = df[df["type"] != "json"]
            tags |= set(valid["tag"].unique())
    return sorted(tags)


def detect_axis(df, tag):
    return "episode" if tag.startswith("episode/") and "episode" in df.columns else "step"


def make_tag_fig(run_data, selected_runs, tag):
    fig = go.Figure()
    multi = len(selected_runs) > 1

    if multi:
        run_order = []
        all_values = []
        for run in selected_runs:
            if run in run_data:
                df = run_data[run]
                sub = df[(df["tag"] == tag) & (df["type"] == "scalar")]
                if not sub.empty:
                    median_val = sub["value"].median()
                    run_order.append((run, median_val))
                    all_values.extend(sub["value"].tolist())

        sign_mean = sum(all_values) / len(all_values) if all_values else 0.0
        if sign_mean >= 0:
            run_order.sort(key=lambda x: x[1], reverse=True)
        else:
            run_order.sort(key=lambda x: x[1], reverse=False)
        sorted_runs = [r for r, _ in run_order]
    else:
        sorted_runs = selected_runs

    for run in sorted_runs:
        if run not in run_data:
            continue
        df = run_data[run]
        sub = df[(df["tag"] == tag) & (df["type"] == "scalar")]

        if sub.empty:
            fig.add_trace(go.Scatter(
                x=[0], y=[None],
                name=f"{run} (no data)",
                mode="lines",
                line=dict(color=get_run_color(run), width=1, dash="dot"),
                showlegend=True,
                hoverinfo="skip"
            ))
            continue

        axis = detect_axis(df, tag)
        color = get_run_color(run)
        opacity = 0.8 if multi else 1.0
        width = 2.5 if not multi or run == sorted_runs[-1] else 1.5

        fig.add_trace(go.Scatter(
            x=sub[axis], y=sub["value"],
            mode="lines",
            name=run,
            line=dict(color=color, width=width),
            opacity=opacity,
            showlegend=True
        ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=len(selected_runs) > 1,
        plot_bgcolor="rgb(25,25,25)",
        paper_bgcolor="rgb(15,15,15)",
        xaxis=dict(showgrid=True, gridcolor="rgba(100,100,100,0.3)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(100,100,100,0.3)")
    )
    return fig


def render_meta_info(run_data):
    blocks = []
    for run, df in run_data.items():
        meta_df = df[df["type"] == "json"]
        if meta_df.empty:
            continue
        run_color = get_run_color(run)
        run_section = [
            html.H4(f"Run: {run}", style={
                "marginTop": "0px", "marginBottom": "6px",
                "color": "#fff", "fontWeight": "bold"
            })
        ]
        for _, row in meta_df.iterrows():
            tag = row.get("tag", "")
            data = row.get("data", {})
            run_section.append(html.Div([
                html.Div(tag, style={"color": "#fff", "fontSize": "12px", "marginBottom": "2px"}),
                html.Pre(json.dumps(data, indent=2, ensure_ascii=False),
                         style={
                             "backgroundColor": "#111", "color": "#ddd",
                             "padding": "4px 6px", "borderRadius": "4px",
                             "whiteSpace": "pre-wrap", "marginTop": "0px"
                         })
            ], style={"marginBottom": "6px"}))
        blocks.append(html.Div(run_section, style={
            "border": f"5px solid {run_color}",
            "padding": "8px", "marginTop": "10px",
            "backgroundColor": "#181818", "borderRadius": "6px"
        }))
    return blocks


def create_app(log_root):
    app = Dash(__name__)
    app.title = "Metrics Viewer"

    app.layout = html.Div([
        html.Div([
            html.Div([
                html.H2("📊 Metrics Viewer", style={"margin": "0", "display": "inline-block"}),
                html.Span(VERSION, style={"color": "#aaa", "fontSize": "13px", "marginLeft": "8px"})
            ], style={"display": "inline-flex", "alignItems": "center"})
        ], style={
            "position": "fixed", "top": "0", "left": "0", "right": "0",
            "zIndex": "1000", "backgroundColor": "inherit",
            "padding": "8px 12px",
            "display": "flex", "justifyContent": "space-between", "alignItems": "center"
        }),

        html.Div([
            html.Label("Runs", style={"marginRight": "6px"}),
            dcc.Dropdown(id="run-select", options=[], value=[], multi=True,
                         placeholder="Select runs",
                         style={"width": "220px", "display": "inline-block", "marginRight": "10px"}),

            html.Label("Tags", style={"marginRight": "6px"}),
            dcc.Dropdown(id="tag-filter", options=[], value=[], multi=True,
                         placeholder="All tags",
                         style={"width": "260px", "display": "inline-block"}),

            html.Button("Manual Refresh", id="refresh-btn", n_clicks=0,
                        style={"marginLeft": "12px", "height": "28px"}),
            html.Button("Auto Refresh: OFF", id="toggle-auto", n_clicks=0,
                        style={"marginLeft": "6px", "height": "28px"})
        ], style={
            "position": "fixed", "top": "48px", "left": "0", "right": "0",
            "zIndex": "999", "backgroundColor": "inherit",
            "padding": "6px 12px 11px 12px",
            "display": "flex", "alignItems": "center", "gap": "6px"
        }),

        html.Div(id="scroll-container", children=[
            html.Div(id="graphs-container", style={"padding": "8px"})
        ], style={
            "position": "absolute", "top": "101px", "bottom": "0",
            "left": "0", "right": "0", "overflowY": "auto"
        }),

        dcc.Store(id="auto-flag", data=False),
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
    def toggle_auto(n, current):
        new_flag = not current
        return (not new_flag, f"Auto Refresh: {'ON' if new_flag else 'OFF'}", new_flag)

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
    def update_graphs(n_auto, n_manual, selected_runs, selected_tags):
        runs = sorted([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))])
        if not runs:
            return [html.Div("No runs.", style={"color": "gray"})], [], [], []
        if not selected_runs:
            selected_runs = [runs[-1]]

        run_data = load_selected_runs(log_root, selected_runs)
        if not run_data:
            return [html.Div("No data.", style={"color": "gray"})], [], selected_runs, []

        all_tags = extract_tags(run_data)
        if not all_tags:
            return [html.Div("No tags.", style={"color": "gray"})], [], selected_runs, []

        display_tags = selected_tags or all_tags
        graphs = []
        for tag in display_tags:
            any_json = any((df[(df["tag"] == tag)]["type"] == "json").any() for df in run_data.values())
            if any_json:
                continue
            fig = make_tag_fig(run_data, selected_runs, tag)
            graphs.append(html.Div([
                html.Div(tag, style={
                    "position": "absolute", "top": "1px", "left": "8px",
                    "backgroundColor": "rgba(0,0,0,0.7)", "color": "white",
                    "fontFamily": "monospace", "fontSize": "13px",
                    "padding": "2px 6px", "borderRadius": "3px", "zIndex": "10"
                }),
                dcc.Graph(figure=fig, config={"displayModeBar": True},
                          style={"height": "300px", "position": "relative"})
            ], style={"marginBottom": "8px", "position": "relative"}))

        meta_blocks = render_meta_info(run_data)
        if meta_blocks:
            graphs.append(html.Hr(style={"borderTop": "1px solid #555"}))
            graphs.extend(meta_blocks)

        # --- Runs ドロップダウンに色付きマークを追加（左側、全Run） ---
        run_options = []
        for r in runs:
            color = get_run_color(r)
            run_options.append({
                "label": html.Span([
                    html.Span("■", style={
                        "color": color,
                        "fontWeight": "bold",
                        "marginRight": "6px",
                        "display": "inline-block",
                        "width": "10px"
                    }),
                    html.Span(r)
                ], style={"display": "inline-flex", "alignItems": "center"}),
                "value": r
            })

        tag_options = [{"label": t, "value": t} for t in all_tags]

        return graphs, run_options, selected_runs, tag_options

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if args.serve:
        print(f"[INFO] Starting Metrics Viewer {VERSION} — {args.logdir}")
        app = create_app(args.logdir)
        app.run(debug=False)
    else:
        target = os.path.abspath(__file__)
        last = os.path.getmtime(target)
        print(f"[INFO] Watching {target}")
        proc = subprocess.Popen([sys.executable, target, "--serve", "--logdir", args.logdir])
        try:
            while True:
                time.sleep(1)
                mtime = os.path.getmtime(target)
                if mtime != last:
                    print("[RELOAD] metrics_viewer.py updated — restarting...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    proc = subprocess.Popen([sys.executable, target, "--serve", "--logdir", args.logdir])
                    last = mtime
        except KeyboardInterrupt:
            proc.terminate()
            print("\n[INFO] Exit watch mode.")
