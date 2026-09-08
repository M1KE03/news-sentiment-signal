"""Every constant from §3 of the implementation plan lives here and nowhere else.

Importing a number from anywhere but this module is a bug. Two cells are marked
PENDING: they are resolved once, at Stage 0, from the dataset audit, and frozen.
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

# ---------------------------------------------------------------- paths ----

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"

HEADLINES_PARQUET = DATA_INTERIM / "headlines.parquet"
SCORES_PARQUET = DATA_INTERIM / "scores.parquet"
MARKET_PARQUET = DATA_INTERIM / "market.parquet"
PANEL_PARQUET = DATA_PROCESSED / "daily_panel.parquet"

LM_DICT_PATH = DATA_RAW / "LoughranMcDonald_MasterDictionary.csv"

# ------------------------------------------------------- D1 news source ----
# PENDING (Stage 0). One of "fnspid" | "benzinga", chosen by the timestamp
# audit in notebooks/01_data_audit.ipynb. Criteria, in priority order:
#   1. usable intraday timestamps with a documented timezone
#   2. coverage / duplication quality
#   3. sample length
# Do not trust any description of these datasets, including the plan's.
NEWS_SOURCE: str | None = None
NEWS_RAW_PATH: Path | None = None

# D2 fallback: if neither candidate passes the timestamp audit, set this True,
# map news dated d to trading day d+1, and drop RQ2 (the contemporaneous act)
# entirely. Decision deadline: end of day 1.
DATE_ONLY_FALLBACK = False

# ------------------------------------------------------ D3 market data ----

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"          # context plots only, never a regressor
MARKET_CALENDAR = "NYSE"

# ----------------------------------------------------- D4 sample window ----
# PENDING (Stage 0). Longest span with stable news coverage.
# Target >= 5 years / >= 1,250 trading days. Locked at Stage 0, then frozen.
SAMPLE_START: str | None = None
SAMPLE_END: str | None = None
MIN_TRADING_DAYS = 1250

# ------------------------------------------------- D5 unit of analysis ----
# Market level: all headlines in the window aggregated per trading day, tested
# against SPY. Single-name analysis appears only as a Stage 6 spot check.
UNIT_OF_ANALYSIS = "market"

# ------------------------------------ D6 timestamp -> trading day rule ----
# A headline stamped s (in America/New_York) belongs to trading day t iff
#   s in (close(t-1), close(t)]        with close = 16:00 ET.
# News on non-trading days rolls forward into the next trading day's window.
TZ_MARKET = "America/New_York"
MARKET_CLOSE_ET = time(16, 0)
# D6 fixes the close at 16:00 ET even on half-days. Using each session's actual
# close is strictly more accurate but is a change to a locked decision -- if you
# flip this, log the change and the date in future-work.md.
USE_ACTUAL_SESSION_CLOSE = False

# ---------------------------------------------------------- D7 scorers ----

SCORERS = ("lm", "vader", "finbert")

FINBERT_MODEL = "ProsusAI/finbert"
# Pin the exact model revision once it is downloaded (Stage 1): an unpinned
# model id is not a reproducible artifact.
FINBERT_REVISION: str | None = None
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 64      # truncation is safe: these are headlines

# Pin the dictionary release actually downloaded from the SRAF site.
LM_DICT_VERSION: str | None = None

# --------------------------------------------- D8/D9 daily aggregation ----

AGG = "mean"                 # equal-weighted mean of the day's headline scores
MIN_HEADLINES_FOR_DISPERSION = 5   # d_t is NaN below this
# Trading days with n_t == 0 are dropped from every regression, and the count
# is reported. Days with 1 <= n_t < 5 keep S_t but are excluded from d_t specs.

# ------------------------------------------------- D10 standard errors ----

NW_MAXLAGS = 5
NW_MAXLAGS_SENSITIVITY = 10  # Stage 6

# ------------------------------------------------------ D11 lag family ----

HORIZONS = (1, 2, 3, 4, 5)
FDR_Q = 0.05                 # Benjamini-Hochberg, within each scorer's family

# ---------------------------------------------------------- D12 placebo ----

PERMUTATION_DRAWS = 1000
PERMUTATION_BLOCK = 21       # trading days
PLACEBO_HORIZON = 1

# ------------------------------------------------ D13 PhraseBank setup ----

PHRASEBANK_SUBSETS = (
    "sentences_50agree",
    "sentences_66agree",
    "sentences_75agree",
    "sentences_allagree",
)
PHRASEBANK_PRIMARY_SUBSET = "sentences_75agree"
PHRASEBANK_METRIC = "macro_f1"
THRESHOLD_FIT_FRACTION = 0.20   # stratified; the other 80% is the eval set

# ----------------------------------------- D14 classifier comparison ----

MCNEMAR_EXACT = True
MCNEMAR_PRIMARY = ("finbert", "lm")
MCNEMAR_SECONDARY = ("finbert", "vader")

# ------------------------------------------------------ D15 effect size ----

BPS_PER_UNIT = 10_000
TRANSACTION_COST_BPS = 5.0   # ~5 bps one-way, the economic-significance bar

# --------------------------------------------- D16 seeds / reproducibility ----

SEED = 20260830

LABELS = ("negative", "neutral", "positive")
