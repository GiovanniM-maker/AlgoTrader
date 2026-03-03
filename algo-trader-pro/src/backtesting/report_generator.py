"""
src/backtesting/report_generator.py

HTML Report Generator
======================
Produces a self-contained, single-file HTML report for a completed backtest.

The report includes:
  * Dark-theme CSS (inlined — no external stylesheet dependency)
  * Chart.js loaded from CDN for interactive charts
  * Summary metrics card
  * Equity curve (line chart with drawdown shading)
  * Monte Carlo chart: 20 sample paths + percentile band lines
  * Trade table (last 50 closed trades)

The file is written to ``reports/backtest_{timestamp}.html`` by default, and
the method returns the absolute path so callers can open or serve the file.

Usage
-----
    from src.backtesting.report_generator import ReportGenerator

    rg = ReportGenerator()
    path = rg.generate_html_report(backtest_result, monte_carlo_result)
    print(f"Report written to {path}")
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHARTJS_CDN = (
    "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
)

_DARK_CSS = """
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242736;
    --border: #2e3150;
    --accent: #6c63ff;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #f59e0b;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --font: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.6;
    padding: 24px;
  }
  h1, h2, h3 { font-weight: 600; letter-spacing: -0.02em; }
  h1 { font-size: 24px; margin-bottom: 4px; }
  h2 { font-size: 18px; margin: 32px 0 16px; }
  .subtitle { color: var(--text-muted); font-size: 13px; margin-bottom: 32px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 32px;
  }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
  }
  .card .label {
    color: var(--text-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
  }
  .card .value {
    font-size: 22px;
    font-weight: 700;
  }
  .card .value.pos { color: var(--green); }
  .card .value.neg { color: var(--red); }
  .card .value.neu { color: var(--yellow); }
  .chart-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 32px;
  }
  canvas { max-height: 380px; }
  .mc-canvas { max-height: 420px; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }
  thead th {
    background: var(--surface2);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
  }
  tbody tr:nth-child(even) { background: var(--surface2); }
  tbody td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
  }
  tbody tr:last-child td { border-bottom: none; }
  .pos { color: var(--green); font-weight: 600; }
  .neg { color: var(--red); font-weight: 600; }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-long  { background: rgba(34,197,94,0.15); color: var(--green); }
  .badge-short { background: rgba(239,68,68,0.15); color: var(--red); }
  .section-header { display: flex; align-items: center; gap: 10px; }
  .section-header small { color: var(--text-muted); font-size: 12px; font-weight: 400; }
  footer {
    margin-top: 48px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
  }
"""


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """
    Generates self-contained HTML backtest reports.

    No external dependencies beyond the Python standard library are required
    at report-generation time.  Chart.js is pulled from a CDN, so an internet
    connection is needed to *view* the charts; all data is embedded as JSON.
    """

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        backtest_result: Any,
        monte_carlo_result: Any,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Build a self-contained HTML report and write it to disk.

        Parameters
        ----------
        backtest_result :
            ``BacktestResult`` dataclass instance (from backtest_engine.py).
        monte_carlo_result :
            ``MonteCarloResult`` dataclass instance (from monte_carlo.py).
        output_path : str, optional
            Where to write the file.  Defaults to
            ``reports/backtest_{timestamp}.html`` relative to the current
            working directory.

        Returns
        -------
        str
            Absolute path to the written HTML file.
        """
        # Resolve output path
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("reports", f"backtest_{ts}.html")

        # Ensure parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent_dir, exist_ok=True)

        # Build HTML
        html = self._build_html(backtest_result, monte_carlo_result)

        # Write file
        abs_path = os.path.abspath(output_path)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        return abs_path

    def generate_summary_dict(
        self,
        backtest_result: Any,
        monte_carlo_result: Any,
    ) -> dict:
        """
        Return a JSON-serialisable summary combining backtest and Monte Carlo results.

        Parameters
        ----------
        backtest_result :
            ``BacktestResult`` dataclass instance.
        monte_carlo_result :
            ``MonteCarloResult`` dataclass instance.

        Returns
        -------
        dict
            Combined summary suitable for API responses or logging.
        """
        bt_metrics = getattr(backtest_result, "metrics", {}) or {}
        mc_dict = monte_carlo_result.to_dict() if hasattr(monte_carlo_result, "to_dict") else {}

        return {
            "generated_at": datetime.now().isoformat(),
            "backtest": {
                "start_equity": getattr(backtest_result, "start_equity", None),
                "end_equity": getattr(backtest_result, "end_equity", None),
                "total_trades": getattr(backtest_result, "total_trades", None),
                "metrics": bt_metrics,
            },
            "monte_carlo": mc_dict,
        }

    # ------------------------------------------------------------------
    # HTML building
    # ------------------------------------------------------------------

    def _build_html(self, bt: Any, mc: Any) -> str:
        """Assemble the full HTML document string."""
        metrics = getattr(bt, "metrics", {}) or {}
        trades = getattr(bt, "trades", []) or []
        equity_curve = getattr(bt, "equity_curve", []) or []

        # ---- Derived metadata ----
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        symbol = metrics.get("symbol", getattr(bt, "symbol", "N/A"))
        timeframe = metrics.get("timeframe", getattr(bt, "timeframe", "N/A"))

        # ---- Build chart data ----
        ec_labels, ec_equity, ec_drawdown = self._parse_equity_curve(equity_curve)
        mc_data = mc.to_dict() if hasattr(mc, "to_dict") else {}
        mc_sample_paths = mc_data.get("sample_paths", [])
        mc_percentiles = {
            str(k): v
            for k, v in mc_data.get("final_equity_percentiles", {}).items()
        }

        # ---- Build summary cards ----
        cards_html = self._build_summary_cards(metrics, mc_data)

        # ---- Build trade table ----
        table_html = self._build_trade_table(trades[-50:])  # last 50

        # ---- Serialise chart data ----
        ec_labels_json = json.dumps(ec_labels)
        ec_equity_json = json.dumps(ec_equity)
        ec_drawdown_json = json.dumps(ec_drawdown)
        mc_paths_json = json.dumps(mc_sample_paths)
        mc_pct_json = json.dumps(mc_percentiles)
        initial_cap = mc_data.get("initial_capital", getattr(bt, "start_equity", 10000))
        initial_cap_json = json.dumps(initial_cap)
        n_sims_json = json.dumps(mc_data.get("n_simulations", 0))

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AlgoTrader Pro — Backtest Report</title>
<script src="{_CHARTJS_CDN}"></script>
<style>{_DARK_CSS}</style>
</head>
<body>

<h1>AlgoTrader Pro — Backtest Report</h1>
<p class="subtitle">
  Symbol: <strong>{self._esc(str(symbol))}</strong> &nbsp;|&nbsp;
  Timeframe: <strong>{self._esc(str(timeframe))}</strong> &nbsp;|&nbsp;
  Generated: {generated_at}
</p>

<h2>Performance Summary</h2>
{cards_html}

<h2>Equity Curve</h2>
<div class="chart-wrap">
  <canvas id="equityChart"></canvas>
</div>

<h2>Monte Carlo Simulation
  <small>({n_sims_json} bootstrap runs — bootstrapped from {len(trades)} historical trades)</small>
</h2>
<div class="chart-wrap">
  <canvas id="mcChart" class="mc-canvas"></canvas>
</div>

<h2>
  <div class="section-header">
    <span>Recent Trades</span>
    <small>(showing last {min(50, len(trades))} of {len(trades)} total)</small>
  </div>
</h2>
{table_html}

<footer>
  AlgoTrader Pro &mdash; Paper Trading Simulation &mdash; {generated_at}
</footer>

<script>
// =====================================================================
// Equity Curve
// =====================================================================
(function() {{
  const labels    = {ec_labels_json};
  const equity    = {ec_equity_json};
  const drawdown  = {ec_drawdown_json};

  const ctx = document.getElementById('equityChart').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: 'Equity (USDT)',
          data: equity,
          borderColor: '#6c63ff',
          backgroundColor: 'rgba(108,99,255,0.08)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          yAxisID: 'y',
          tension: 0.3,
        }},
        {{
          label: 'Drawdown (%)',
          data: drawdown,
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239,68,68,0.10)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          yAxisID: 'y1',
          tension: 0.3,
        }}
      ]
    }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ color: '#e2e8f0' }} }},
        tooltip: {{ backgroundColor: '#1a1d27', borderColor: '#2e3150', borderWidth: 1 }}
      }},
      scales: {{
        x: {{
          ticks: {{ color: '#94a3b8', maxTicksLimit: 12, maxRotation: 0 }},
          grid: {{ color: '#2e3150' }}
        }},
        y: {{
          type: 'linear', position: 'left',
          ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }},
          grid: {{ color: '#2e3150' }}
        }},
        y1: {{
          type: 'linear', position: 'right',
          ticks: {{ color: '#ef4444', callback: v => v.toFixed(1) + '%' }},
          grid: {{ drawOnChartArea: false }}
        }}
      }}
    }}
  }});
}})();

// =====================================================================
// Monte Carlo
// =====================================================================
(function() {{
  const paths       = {mc_paths_json};
  const pctiles     = {mc_pct_json};
  const initCap     = {initial_cap_json};
  const nSims       = {n_sims_json};

  const ctx = document.getElementById('mcChart').getContext('2d');

  const PALETTE = [
    '#6c63ff44','#22c55e44','#f59e0b44','#ef444444','#06b6d444',
    '#a855f744','#ec489944','#84cc1644','#14b8a644','#f9731644',
    '#0ea5e944','#8b5cf644','#10b98144','#fb923c44','#a78bfa44',
    '#34d39944','#fb718544','#38bdf844','#fbbf2444','#4ade8044'
  ];

  const datasets = [];

  // Sample paths (faint)
  paths.forEach((path, i) => {{
    const nSteps = path.length;
    datasets.push({{
      label: i === 0 ? 'Sample paths' : undefined,
      data: path,
      borderColor: PALETTE[i % PALETTE.length],
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      showInLegend: i === 0,
    }});
  }});

  // Percentile lines
  const pctLineColors = {{
    '5':  '#ef4444',
    '25': '#f59e0b',
    '50': '#22c55e',
    '75': '#f59e0b',
    '95': '#6c63ff'
  }};

  if (paths.length > 0) {{
    const nSteps = paths[0].length;
    Object.entries(pctiles).forEach(([pct, finalVal]) => {{
      // Interpolate a straight line from initCap to finalVal
      const lineData = Array.from({{length: nSteps}}, (_, i) =>
        initCap + (finalVal - initCap) * (i / (nSteps - 1 || 1))
      );
      datasets.push({{
        label: 'p' + pct,
        data: lineData,
        borderColor: pctLineColors[pct] || '#ffffff',
        borderWidth: 2,
        borderDash: [6, 3],
        pointRadius: 0,
        fill: false,
      }});
    }});
  }}

  const labels = paths.length > 0
    ? Array.from({{length: paths[0].length}}, (_, i) => 'Trade ' + i)
    : [];

  new Chart(ctx, {{
    type: 'line',
    data: {{ labels, datasets }},
    options: {{
      responsive: true,
      animation: false,
      plugins: {{
        legend: {{
          labels: {{ color: '#e2e8f0', filter: item => item.text !== undefined }}
        }},
        tooltip: {{ backgroundColor: '#1a1d27', borderColor: '#2e3150', borderWidth: 1 }}
      }},
      scales: {{
        x: {{
          ticks: {{ color: '#94a3b8', maxTicksLimit: 10 }},
          grid: {{ color: '#2e3150' }}
        }},
        y: {{
          ticks: {{ color: '#94a3b8', callback: v => '$' + v.toLocaleString() }},
          grid: {{ color: '#2e3150' }}
        }}
      }}
    }}
  }});
}})();
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Summary cards
    # ------------------------------------------------------------------

    def _build_summary_cards(self, metrics: dict, mc_data: dict) -> str:
        """Render the top-level metric cards as an HTML string."""

        def _fmt_pct(v: Any, decimals: int = 2) -> str:
            if v is None:
                return "N/A"
            try:
                return f"{float(v):.{decimals}f}%"
            except (TypeError, ValueError):
                return "N/A"

        def _fmt_float(v: Any, decimals: int = 2, prefix: str = "") -> str:
            if v is None:
                return "N/A"
            try:
                return f"{prefix}{float(v):.{decimals}f}"
            except (TypeError, ValueError):
                return "N/A"

        def _fmt_int(v: Any) -> str:
            if v is None:
                return "N/A"
            try:
                return str(int(v))
            except (TypeError, ValueError):
                return "N/A"

        def _color(v: Any, pos_good: bool = True) -> str:
            """Return CSS class for positive/negative coloring."""
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return "neu"
            if fv > 0:
                return "pos" if pos_good else "neg"
            if fv < 0:
                return "neg" if pos_good else "pos"
            return "neu"

        total_return = metrics.get("total_return_pct") or metrics.get("total_return")
        sharpe = metrics.get("sharpe_ratio")
        sortino = metrics.get("sortino_ratio")
        max_dd = metrics.get("max_drawdown_pct") or metrics.get("max_drawdown")
        win_rate = metrics.get("win_rate") or metrics.get("win_rate_pct")
        profit_factor = metrics.get("profit_factor")
        total_trades = metrics.get("total_trades")
        calmar = metrics.get("calmar_ratio")
        final_equity = metrics.get("final_equity") or metrics.get("end_equity")
        cagr = metrics.get("cagr") or metrics.get("cagr_pct")

        prob_profit = mc_data.get("probability_of_profit")
        prob_ruin = mc_data.get("probability_of_ruin")
        p50_equity = mc_data.get("median_final_equity")

        # Win rate: normalise to percentage if stored as 0-1 fraction
        if win_rate is not None:
            try:
                win_rate_f = float(win_rate)
                if 0 < win_rate_f <= 1.0:
                    win_rate = win_rate_f * 100.0
            except (TypeError, ValueError):
                pass

        cards = [
            ("Total Return",     _fmt_pct(total_return),                     _color(total_return)),
            ("Final Equity",     _fmt_float(final_equity, prefix="$"),        "neu"),
            ("Sharpe Ratio",     _fmt_float(sharpe, 3),                       _color(sharpe)),
            ("Sortino Ratio",    _fmt_float(sortino, 3),                      _color(sortino)),
            ("Max Drawdown",     _fmt_pct(max_dd),                            _color(max_dd, pos_good=False)),
            ("Win Rate",         _fmt_pct(win_rate),                          _color(win_rate)),
            ("Profit Factor",    _fmt_float(profit_factor, 3),                _color(profit_factor)),
            ("Calmar Ratio",     _fmt_float(calmar, 3),                       _color(calmar)),
            ("CAGR",             _fmt_pct(cagr),                              _color(cagr)),
            ("Total Trades",     _fmt_int(total_trades),                      "neu"),
            ("MC Prob. Profit",  _fmt_pct(prob_profit),                       _color(prob_profit)),
            ("MC Prob. Ruin",    _fmt_pct(prob_ruin),                         _color(prob_ruin, pos_good=False)),
            ("MC Median Equity", _fmt_float(p50_equity, prefix="$"),          "neu"),
        ]

        card_html_parts = []
        for label, value, css_class in cards:
            card_html_parts.append(
                f'<div class="card">'
                f'<div class="label">{self._esc(label)}</div>'
                f'<div class="value {css_class}">{self._esc(value)}</div>'
                f'</div>'
            )

        return f'<div class="grid">{"".join(card_html_parts)}</div>'

    # ------------------------------------------------------------------
    # Trade table
    # ------------------------------------------------------------------

    def _build_trade_table(self, trades: List[dict]) -> str:
        """Render the last-50-trades table as an HTML string."""
        if not trades:
            return '<p style="color:var(--text-muted);padding:12px">No trades to display.</p>'

        rows_html: List[str] = []
        for t in trades:
            symbol = self._esc(str(t.get("symbol", "")))
            direction = str(t.get("direction", "")).upper()
            badge_class = "badge-long" if direction == "LONG" else "badge-short"
            badge = f'<span class="badge {badge_class}">{self._esc(direction)}</span>'

            entry_price = self._fmt_price(t.get("entry_price"))
            exit_price = self._fmt_price(t.get("exit_price"))
            entry_time = self._fmt_ts(t.get("entry_time"))
            exit_time = self._fmt_ts(t.get("exit_time"))

            net_pnl = t.get("net_pnl") or t.get("pnl")
            pnl_pct = t.get("pnl_pct")
            pnl_css = ""
            try:
                if float(net_pnl or 0) > 0:
                    pnl_css = "pos"
                elif float(net_pnl or 0) < 0:
                    pnl_css = "neg"
            except (TypeError, ValueError):
                pass

            net_pnl_str = (
                f'<span class="{pnl_css}">'
                f'{self._fmt_currency(net_pnl)}'
                f'</span>'
            )
            pnl_pct_str = (
                f'<span class="{pnl_css}">'
                f'{self._fmt_pct_val(pnl_pct)}'
                f'</span>'
            )

            exit_reason = self._esc(str(t.get("exit_reason", "")).replace("_", " ").title())
            confidence = self._fmt_float_val(t.get("confidence_score"), decimals=1)
            duration = t.get("duration_minutes")
            duration_str = (
                self._fmt_duration(int(duration)) if duration is not None else "—"
            )

            rows_html.append(
                f"<tr>"
                f"<td>{symbol}</td>"
                f"<td>{badge}</td>"
                f"<td>{entry_time}</td>"
                f"<td>{exit_time}</td>"
                f"<td>{entry_price}</td>"
                f"<td>{exit_price}</td>"
                f"<td>{net_pnl_str}</td>"
                f"<td>{pnl_pct_str}</td>"
                f"<td>{exit_reason}</td>"
                f"<td>{confidence}</td>"
                f"<td>{duration_str}</td>"
                f"</tr>"
            )

        headers = [
            "Symbol", "Dir", "Entry Time", "Exit Time",
            "Entry $", "Exit $", "Net P&amp;L", "P&amp;L %",
            "Exit Reason", "Conf", "Duration",
        ]
        thead = (
            "<thead><tr>"
            + "".join(f"<th>{h}</th>" for h in headers)
            + "</tr></thead>"
        )

        return (
            f"<div style='overflow-x:auto;margin-bottom:32px'>"
            f"<table>{thead}<tbody>"
            + "".join(rows_html)
            + "</tbody></table></div>"
        )

    # ------------------------------------------------------------------
    # Data parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_equity_curve(
        equity_curve: List[dict],
    ) -> tuple[List[str], List[float], List[float]]:
        """
        Extract chart-ready lists from the equity curve records.

        Parameters
        ----------
        equity_curve : list of dict
            Each dict must contain ``timestamp`` (str), ``equity`` (float),
            and ``drawdown_pct`` (float).

        Returns
        -------
        labels : list of str
        equity_values : list of float
        drawdown_values : list of float
        """
        labels: List[str] = []
        equity_vals: List[float] = []
        dd_vals: List[float] = []

        for row in equity_curve:
            ts = row.get("timestamp", "")
            # Shorten ISO timestamps for the axis
            if isinstance(ts, str) and len(ts) >= 16:
                ts = ts[:16].replace("T", " ")
            labels.append(str(ts))
            equity_vals.append(float(row.get("equity", 0)))
            dd_vals.append(float(row.get("drawdown_pct", 0)))

        return labels, equity_vals, dd_vals

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _esc(text: str) -> str:
        """Minimal HTML entity escaping."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _fmt_price(v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"${float(v):,.4f}"
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _fmt_currency(v: Any) -> str:
        if v is None:
            return "—"
        try:
            fv = float(v)
            sign = "+" if fv > 0 else ""
            return f"{sign}${fv:,.2f}"
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _fmt_pct_val(v: Any) -> str:
        if v is None:
            return "—"
        try:
            fv = float(v)
            sign = "+" if fv > 0 else ""
            return f"{sign}{fv:.2f}%"
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _fmt_float_val(v: Any, decimals: int = 2) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _fmt_ts(v: Any) -> str:
        if v is None:
            return "—"
        s = str(v)
        if len(s) >= 16:
            return s[:16].replace("T", " ")
        return s

    @staticmethod
    def _fmt_duration(minutes: int) -> str:
        """Convert minutes to human-readable duration string."""
        if minutes < 60:
            return f"{minutes}m"
        if minutes < 1440:
            h, m = divmod(minutes, 60)
            return f"{h}h {m}m"
        d, rem = divmod(minutes, 1440)
        h = rem // 60
        return f"{d}d {h}h"
