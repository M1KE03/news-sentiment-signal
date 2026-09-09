"""Loading and hygiene for the two raw inputs: the news dump and SPY/VIX.

Produces `interim/headlines.parquet` and `interim/market.parquet` exactly to the
schemas in §4. Nothing here knows about scoring, alignment or regressions.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

import config

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

HEADLINE_COLUMNS = ["headline_id", "text", "text_norm", "ts_utc", "ts_et", "tickers"]

# Column name and source timezone per candidate dataset. Both are asserted in
# the Stage 0 audit rather than trusted -- see notebooks/01_data_audit.ipynb.
_SOURCE_SPEC = {
    "fnspid": {"text": "Article_title", "ts": "Date", "tickers": "Stock_symbol", "tz": "UTC"},
    "benzinga": {"text": "title", "ts": "date", "tickers": "stock", "tz": "America/New_York"},
}


def normalize_text(s: pd.Series) -> pd.Series:
    """Dedup key only. Never scored -- scorers see the original `text`."""
    out = s.fillna("").str.lower()
    out = out.map(lambda t: _PUNCT.sub(" ", t))
    return out.map(lambda t: _WS.sub(" ", t).strip())


def _headline_id(text_norm: str, ts_utc: pd.Timestamp) -> str:
    key = f"{text_norm}|{ts_utc.isoformat()}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_news(path: str | Path, source: Literal["fnspid", "benzinga"]) -> pd.DataFrame:
    """Read a raw news dump into the §4 `headlines` schema.

    The source timezone is applied explicitly from `_SOURCE_SPEC`, never
    inferred from the data. Timezone-naive source stamps are localized; aware
    ones are converted.
    """
    if source not in _SOURCE_SPEC:
        raise ValueError(f"unknown source {source!r}; expected one of {list(_SOURCE_SPEC)}")
    spec = _SOURCE_SPEC[source]

    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    missing = [spec[k] for k in ("text", "ts") if spec[k] not in raw.columns]
    if missing:
        raise KeyError(f"{path.name} is missing expected column(s) {missing}")

    ts = pd.to_datetime(raw[spec["ts"]], errors="coerce", utc=False)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(spec["tz"], ambiguous="NaT", nonexistent="NaT")
    ts_utc = ts.dt.tz_convert("UTC")

    tick_col = spec.get("tickers")
    if tick_col and tick_col in raw.columns:
        tickers = raw[tick_col].fillna("").map(
            lambda v: [t.strip().upper() for t in str(v).split(",") if t.strip()]
        )
    else:
        tickers = pd.Series([[] for _ in range(len(raw))], index=raw.index)

    df = pd.DataFrame(
        {
            "text": raw[spec["text"]].astype("string").str.strip(),
            "ts_utc": ts_utc,
            "tickers": tickers,
        }
    )
    df = df[df["text"].notna() & (df["text"].str.len() > 0) & df["ts_utc"].notna()]
    df["text_norm"] = normalize_text(df["text"])
    df = df[df["text_norm"].str.len() > 0]
    df["ts_et"] = df["ts_utc"].dt.tz_convert(config.TZ_MARKET)
    df["headline_id"] = [
        _headline_id(tn, ts) for tn, ts in zip(df["text_norm"], df["ts_utc"])
    ]
    df["text"] = df["text"].astype(str)

    return df.loc[:, HEADLINE_COLUMNS].sort_values("ts_utc").reset_index(drop=True)


def _token_set_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def dedup(
    df: pd.DataFrame,
    near_dupe: bool = True,
    window_days: int = 3,
    overlap: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """Drop exact `text_norm` repeats within `window_days`, then near-duplicates.

    Returns the surviving frame plus the counts, so the report can quote a dedup
    rate instead of asserting one.
    """
    n_in = len(df)
    df = df.sort_values("ts_utc").reset_index(drop=True)

    # Exact repeats within the window: keep the first occurrence.
    first_seen = df.groupby("text_norm")["ts_utc"].transform("min")
    within = (df["ts_utc"] - first_seen) <= pd.Timedelta(days=window_days)
    is_first = ~df.duplicated("text_norm", keep="first")
    keep_exact = is_first | ~within
    df_exact = df[keep_exact].reset_index(drop=True)
    n_exact_dropped = n_in - len(df_exact)

    n_near_dropped = 0
    if near_dupe and len(df_exact):
        keep = np.ones(len(df_exact), dtype=bool)
        ts = df_exact["ts_utc"].to_numpy()
        norms = df_exact["text_norm"].to_numpy()
        window = pd.Timedelta(days=window_days).to_timedelta64()
        j_start = 0
        for i in range(len(df_exact)):
            while ts[i] - ts[j_start] > window:
                j_start += 1
            for j in range(j_start, i):
                if keep[j] and _token_set_overlap(norms[i], norms[j]) >= overlap:
                    keep[i] = False
                    break
        n_near_dropped = int((~keep).sum())
        df_exact = df_exact[keep].reset_index(drop=True)

    stats = {
        "n_in": n_in,
        "n_out": len(df_exact),
        "n_exact_dropped": int(n_exact_dropped),
        "n_near_dropped": n_near_dropped,
        "dedup_rate": 0.0 if n_in == 0 else 1 - len(df_exact) / n_in,
        "window_days": window_days,
        "overlap_threshold": overlap,
    }
    return df_exact, stats


def trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """NYSE session dates in [start, end], tz-naive dates.

    Raises if `pandas_market_calendars` is unavailable. It previously fell back
    to `pd.bdate_range` with a warning, which treats every market holiday as a
    session -- roughly nine phantom sessions a year, silently shifting what
    "the next trading day" means, behind a warning that scrolls past (B05
    defect D-3). A wrong calendar is not a degraded mode of this pipeline; it
    is a wrong answer, so it is a hard failure.
    """
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "pandas_market_calendars is required: every lag, lead and "
            "session mapping in this project is defined against the exchange "
            "calendar, and substituting business days would silently treat ~9 "
            "market holidays a year as trading sessions.\n"
            "    pip install pandas-market-calendars"
        ) from exc

    sched = mcal.get_calendar(config.MARKET_CALENDAR).schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index).normalize()


def load_market(start: str, end: str) -> pd.DataFrame:
    """SPY daily bars + ^VIX close -> the §4 `market` schema."""
    import yfinance as yf

    spy = yf.download(
        config.MARKET_TICKER, start=start, end=end, auto_adjust=True, progress=False
    )
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if spy.empty:
        raise RuntimeError(f"no {config.MARKET_TICKER} data returned for {start}..{end}")

    vix = yf.download(config.VIX_TICKER, start=start, end=end, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    out = pd.DataFrame(index=pd.DatetimeIndex(spy.index).normalize())
    out.index.name = "date"
    out["close_adj"] = spy["Close"].to_numpy()
    out["ret"] = np.log(out["close_adj"]).diff()
    out["parkinson"] = (np.log(spy["High"] / spy["Low"]).to_numpy() ** 2) / (4 * np.log(2))
    out["volume"] = spy["Volume"].to_numpy()
    out["log_turnover"] = np.log(out["volume"].replace(0, np.nan))
    out["vix_close"] = vix["Close"].reindex(out.index).to_numpy() if not vix.empty else np.nan

    return out.reset_index()
