from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from market_state_lab.models import MarketStateResult, StyleResult


def _figure_html(figure: go.Figure, include_plotly: bool | str) -> str:
    figure.update_layout(
        template="plotly_white",
        height=330,
        margin=dict(l=42, r=18, t=50, b=35),
        font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#20242a"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs=include_plotly,
        config={"displaylogo": False, "responsive": True},
    )


def _absolute_suffix(latest: dict[str, Any]) -> str:
    """Show the expanding-window percentile next to the rolling one.

    A rolling percentile puts ~38% of every period in the high band, so on its
    own it cannot distinguish a calm year from a crisis.
    """
    expanding = latest.get("risk_percentile_expanding")
    band = latest.get("vix_band")
    parts: list[str] = []
    if isinstance(expanding, (int, float)):
        parts.append(f"{float(expanding):.1%}")
    if band:
        parts.append(html.escape(str(band)))
    return f" <span class='sub'>/ {' | '.join(parts)}</span>" if parts else ""


def _table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return '<p class="muted">No data available.</p>'
    view = frame[columns] if columns else frame
    return view.to_html(index=False, border=0, classes="data-table", float_format=lambda x: f"{x:.4f}")


def write_dashboard(
    path: Path,
    state: MarketStateResult,
    style: StyleResult,
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    years = int(config["reporting"].get("history_years", 5))
    cutoff = state.history.index.max() - pd.DateOffset(years=years)
    state_history = state.history.loc[cutoff:]
    style_history = style.history.loc[cutoff:]

    probability_figure = go.Figure()
    probability_figure.add_trace(go.Scatter(x=state_history.index, y=state_history["p_low_risk"], name="Low risk", line=dict(color="#16845b")))
    probability_figure.add_trace(go.Scatter(x=state_history.index, y=state_history["p_mid_risk"], name="Mid risk", line=dict(color="#c08a13")))
    probability_figure.add_trace(go.Scatter(x=state_history.index, y=state_history["p_high_risk"], name="High risk", line=dict(color="#ba3b46")))
    probability_figure.update_layout(title="Walk-forward market-state probabilities", yaxis=dict(range=[0, 1], tickformat=".0%"))

    risk_figure = go.Figure()
    risk_figure.add_trace(go.Scatter(x=state_history.index, y=state_history["risk_percentile"], name="Risk percentile", line=dict(color="#4058a5")))
    risk_figure.add_hline(y=0.62, line_dash="dot", line_color="#ba3b46")
    risk_figure.add_hline(y=0.38, line_dash="dot", line_color="#16845b")
    risk_figure.update_layout(title="Observable risk percentile", yaxis=dict(range=[0, 1], tickformat=".0%"))

    style_figure = go.Figure()
    for column in [column for column in style_history if column.startswith("score_")]:
        style_figure.add_trace(
            go.Scatter(
                x=style_history.index,
                y=style_history[column],
                name=column.replace("score_", "").replace("_", " ").title(),
            )
        )
    style_figure.add_hline(y=0, line_color="#8b9098", line_width=1)
    style_figure.update_layout(title="Trailing-only market-style evidence")

    latest = state.latest
    style_table = style.latest.copy()
    if not style_table.empty:
        style_table["favored_strength"] = style_table["favored_strength"].map(lambda value: f"{value:.1%}")
        style_table["dimension"] = style_table["dimension"].str.replace("_", " ").str.title()
        style_table["favored_style"] = style_table["favored_style"].str.replace("_", " ").str.title()
    manifest_view = manifest.copy()
    if "error" in manifest_view:
        manifest_view["error"] = manifest_view["error"].fillna("").map(lambda value: str(value)[:120])
    vintage_warning = (
        "Latest-vintage or retrospectively revised inputs are present. Historical rows must not be treated as a strict backtest."
        if latest.get("history_is_latest_vintage")
        else "All enabled historical inputs passed the configured point-in-time contract."
    )
    news_overlay = latest.get("news_overlay", {})
    news_features = news_overlay.get("features", {}) if isinstance(news_overlay, dict) else {}
    news_table = pd.DataFrame(
        [{"feature": key, "value": value} for key, value in news_features.items()]
    )
    news_status = (
        str(news_overlay.get("status") or news_overlay.get("llm", {}).get("status") or "not enabled")
        if isinstance(news_overlay, dict)
        else "not enabled"
    )

    # The report used to publish a headline state and a set of decision weights
    # that can point the other way, with nothing saying which to act on.
    half_life = float(config["models"]["market_state"].get("decision_half_life_days", 15))
    decision_source = str(latest.get("decision_weight_source", "ensemble"))
    decision_weights = latest.get("decision_weights", {}) or {}
    if decision_weights:
        leading = max(decision_weights, key=lambda name: decision_weights[name])
        decision_line = (
            f"Act on the decision weights: <strong>{html.escape(leading.replace('_', ' '))}</strong> "
            f"at {decision_weights[leading]:.1%}, from the {html.escape(decision_source)} "
            f"probabilities smoothed with a {half_life:g}-day half-life. The headline state above is "
            f"the unsmoothed read and will lead the decision weights by roughly {half_life:g} "
            f"trading days at a turn - when the two disagree, that lag is the reason."
        )
    else:
        decision_line = "No decision weights were produced for this run."

    # A synthetic run has to announce itself on the page, not only in the JSON.
    is_fixture = str(latest.get("point_in_time_status", "")) == "synthetic_fixture"
    fixture_banner = (
        '<section class="fixture">'
        "<strong>SYNTHETIC FIXTURE</strong> - every number on this page comes from the offline "
        "test fixture, not from market data. Run <code>scripts/run_daily.py</code> for the live "
        "report.</section>"
        if is_fixture
        else ""
    )
    title = "SYNTHETIC - Market State Lab" if is_fixture else "Market State Lab"

    skill = latest.get("forward_skill", {}) or {}
    if skill:
        verdict = "beats" if skill.get("beats_climatology") else "does NOT beat"
        skill_line = (
            f"Forward skill: the calibrated ensemble {verdict} the no-skill climatology on "
            f"{html.escape(str(skill.get('target', '')))} - Brier "
            f"{float(skill.get('forward_brier', float('nan'))):.3f} against "
            f"{float(skill.get('climatology_brier', float('nan'))):.3f} over "
            f"{int(skill.get('observations', 0)):,} paired days. The self-consistency Brier in the "
            f"table below is not accuracy; it only measures agreement with the percentile rule."
        )
    else:
        skill_line = "Forward skill was not evaluated for this run."

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#20242a; --muted:#68707b; --line:#d9dde3; --paper:#ffffff; --band:#f4f6f8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--band); color:var(--ink); font-family:"Segoe UI",Arial,sans-serif; letter-spacing:0; }}
header {{ background:#17212b; color:#fff; padding:22px max(24px, calc((100vw - 1180px)/2)); }}
header h1 {{ margin:0 0 4px; font-size:25px; font-weight:650; letter-spacing:0; }}
header p {{ margin:0; color:#cdd5dc; font-size:13px; }}
main {{ max-width:1180px; margin:0 auto; padding:22px 24px 40px; }}
.summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-bottom:18px; }}
.metric {{ background:var(--paper); border:1px solid var(--line); border-radius:6px; padding:14px; min-height:88px; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:7px; }}
.metric strong {{ display:block; font-size:21px; line-height:1.2; overflow-wrap:anywhere; }}
section {{ background:var(--paper); border-top:1px solid var(--line); margin:0 0 16px; padding:16px 18px; }}
section h2 {{ font-size:16px; margin:0 0 12px; letter-spacing:0; }}
.charts {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.data-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
.data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:8px 7px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
.data-table th {{ background:#f7f8fa; font-weight:600; }}
.muted,.note {{ color:var(--muted); font-size:12px; line-height:1.5; }}
.callout {{ border-top:3px solid #17212b; margin-bottom:16px; }}
.callout p {{ margin:0 0 8px; font-size:13px; line-height:1.6; }}
.callout p:last-child {{ margin-bottom:0; color:var(--muted); }}
.metric .sub {{ font-size:13px; color:var(--muted); font-weight:400; }}
.fixture {{ background:#fdf2d0; border-top:3px solid #b8860b; color:#5a4405; font-size:13px; line-height:1.6; margin-bottom:16px; }}
.fixture code {{ background:#f5e6b8; padding:1px 5px; border-radius:3px; }}
@media (max-width:780px) {{ .summary,.charts {{ grid-template-columns:1fr; }} main {{ padding:14px 10px 28px; }} }}
</style>
</head>
<body>
<header><h1>Market State Lab</h1><p>US close-to-close research dashboard | read-only data workflow</p></header>
<main>
{fixture_banner}
<div class="summary">
  <div class="metric"><span>As of</span><strong>{html.escape(str(latest['as_of']))}</strong></div>
  <div class="metric"><span>Risk state</span><strong>{html.escape(str(latest['market_state']).replace('_', ' ').title())}</strong></div>
  <div class="metric"><span>State confidence</span><strong>{latest['confidence']:.1%}</strong></div>
  <div class="metric"><span>Risk percentile (rolling / all history)</span><strong>{latest['risk_percentile']:.1%}{_absolute_suffix(latest)}</strong></div>
</div>
<section class="callout"><p>{decision_line}</p><p>{skill_line}</p></section>
<div class="charts">
  <section>{_figure_html(probability_figure, 'inline' if config['reporting'].get('inline_plotly', True) else 'cdn')}</section>
  <section>{_figure_html(risk_figure, False)}</section>
</div>
<section>{_figure_html(style_figure, False)}</section>
<section><h2>Model comparison</h2>{_table(state.comparison)}</section>
<section><h2>Risk decision value</h2><p class="muted">All methods share the same dates and one-day-lagged exposures, net of transaction costs, with Sharpe measured excess of cash. significant_vs_buy_and_hold is the only column that answers whether a difference is real; a bootstrap interval spanning zero means the gap is noise. Diagnostics, not a strategy backtest.</p>{_table(state.decision_value)}</section>
<section><h2>Latest style evidence</h2>{_table(style_table, ['dimension','favored_style','favored_strength','score','ff_score','etf_score','source_mode','source_agreement','as_of','age_business_days','data_status'])}</section>
<section><h2>News overlay</h2><p class="muted">Status: {html.escape(news_status)}. This challenger signal is not an input to the state ensemble.</p>{_table(news_table)}</section>
<section><h2>Data quality and freshness</h2>{_table(manifest_view)}</section>
<section class="note">Information date: {html.escape(str(latest.get('information_date', latest['as_of'])))}. Run date: {html.escape(str(latest.get('run_date', '')))}. {html.escape(vintage_warning)} State probabilities are an expanding-window ensemble; decision weights use a separate {float(config['models']['market_state'].get('decision_half_life_days', 15)):g}-day half-life. Style values are scores and soft signs, not calibrated win probabilities. No output in this report is an automatic trade instruction.</section>
</main>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
