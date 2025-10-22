#!/usr/bin/env python3
"""
metrics_viewer.py
---------------------------------------
TensorBoard風メトリクスビューア（Plotly Dash）
- 手動更新ボタン
- 自動更新 ON/OFF
- タグフィルタ
- ズーム保持（zoom-store）
- サンプリング + Scattergl（高速描画）
- 初期白抜け対策（複数回 Plotly.resize）
---------------------------------------
"""

import os, json, glob, re
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.dependencies import ALL


# -------------------------
# JSONL 読み込み
# -------------------------
def read_jsonl(path):
    rec = []
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                if j.get("type") == "scalar":
                    rec.append(j)
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rec)
    if df.empty:
        return df
    mask = df["tag"].str.startswith("episode/")
    if mask.any():
        df.loc[mask, "episode"] = df.loc[mask, "step"]
    return df


def load_all_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        path = os.path.join(d, "metrics.jsonl")
        if not os.path.exists(path):
            continue
        df = read_jsonl(path)
        if not df.empty:
            runs[os.path.basename(d)] = df
    return runs


def extract_tags(run_data):
    tags = set()
    for df in run_data.values():
        tags |= set(df["tag"].unique())
    return sorted(tags)


def detect_axis_column(df, tag):
    return "episode" if tag.startswith("episode/") and "episode" in df.columns else "step"


def safe_tag(tag: str) -> str:
    """Dash IDで安全な形式に変換"""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", tag)


def make_tag_fig(run_data, selected_runs, tag, zoom_state):
    fig = go.Figure()
    MAX_POINTS = 2000  # 負荷対策
    for run, df in run_data.items():
        if run not in selected_runs:
            continue
        axis_col = detect_axis_column(df, tag)
        sub = df[df["tag"] == tag]
        if len(sub) > MAX_POINTS:
            sub = sub.iloc[::max(1, len(sub)//MAX_POINTS)]
        if sub.empty:
            continue
        #fig.add_trace(go.Scattergl(
        fig.add_trace(go.Scatter(
            x=sub[axis_col],
            y=sub["value"],
            mode="lines",
            name=f"{run}"
        ))

    fig.update_layout(
        template="plotly_dark",
        title=tag,
        xaxis_title=detect_axis_column(df, tag),
        yaxis_title=tag,
        height=300,
        margin=dict(l=40, r=20, t=40, b=40)
    )

    if zoom_state and tag in zoom_state:
        zr = zoom_state[tag]
        if zr.get("xrange"):
            fig.update_xaxes(range=zr["xrange"])
        if zr.get("yrange"):
            fig.update_yaxes(range=zr["yrange"])

    return fig


# -------------------------
# Dash アプリ
# -------------------------
def create_app(log_root):
    app = Dash(__name__)

    app.layout = html.Div([
        html.H2("📊 Metrics Viewer"),

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
            html.Button("手動更新", id="refresh-btn", n_clicks=0,
                        style={"marginRight": "10px"}),
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
        html.Div(id="dummy-output", style={"display": "none"}),  # JS連携用
        dcc.Interval(id="tick", interval=5000, n_intervals=0, disabled=True)
    ])

    # 自動更新トグル
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

    # メイン更新（手動 or 自動）
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
        run_data = load_all_runs(log_root)
        if not run_data:
            return [html.Div("No runs found.", style={"color": "gray"})], [], selected_runs, []

        run_names = sorted(run_data.keys())
        latest_run = run_names[-1]
        if not selected_runs:
            selected_runs = [latest_run]

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

        run_options = [{"label": r, "value": r} for r in run_names]
        tag_options = [{"label": t, "value": t} for t in all_tags]

        return graphs, run_options, selected_runs, tag_options

    # ズーム状態の保持
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

    # 初期白抜け対策（複数回 Plotly.resize）
    app.clientside_callback(
        """
        function(children) {
            function forceRedraw(){
                const graphs = document.querySelectorAll('.js-plotly-plot');
                graphs.forEach(p=>{
                    if(p.clientHeight < 50){
                        try{window.Plotly.Plots.resize(p);}catch(e){}
                    }
                });
            }
            [100,300,800,1500,3000].forEach(t=>setTimeout(forceRedraw,t));
            return null;
        }
        """,
        Output("dummy-output", "children"),
        Input("graphs-container", "children")
    )

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()

    print(f"[INFO] Starting viewer — {args.logdir}")
    app = create_app(args.logdir)
    app.run(debug=False)
