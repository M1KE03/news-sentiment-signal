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

# The analysis input: filtered to the universe AND deduplicated.
HEADLINES_PARQUET = DATA_INTERIM / "headlines.parquet"
# Filtered but not deduplicated, kept so the dedup rate stays checkable.
HEADLINES_RAW_PARQUET = DATA_INTERIM / "headlines_raw.parquet"
# One row per RAW row: which cluster it joined, whether it survived, and what
# eliminated it. Makes the dedup rate recomputable from the artifacts instead of
# only reproducible by re-running the pass (R03b). The writer derives the actual
# path from the clean corpus it accompanies (`download.lineage_path`), so this
# names the default location rather than dictating it.
DEDUP_LINEAGE_PARQUET = DATA_INTERIM / "dedup_lineage.parquet"
SCORES_PARQUET = DATA_INTERIM / "scores.parquet"
MARKET_PARQUET = DATA_INTERIM / "market.parquet"
PANEL_PARQUET = DATA_PROCESSED / "daily_panel.parquet"

LM_DICT_PATH = DATA_RAW / "LoughranMcDonald_MasterDictionary.csv"

# ------------------------------------------------------- news source ----
# Resolved at B04. Evidence: docs/data-audit-fnspid.md.
#
# FNSPID's All_external.csv is five-plus heterogeneous sub-corpora concatenated
# (Reuters, Benzinga, SeekingAlpha, Zacks, Bloomberg, lenta.ru, ...) with
# different languages, schemas and timestamp behaviour. The Benzinga block is
# selected on relevance and coverage: US-equity, English, 100% ticker-tagged,
# single editorial source (so the source mix cannot drift under the aggregate).
NEWS_SOURCE = "fnspid_benzinga"
NEWS_RAW_FILE = "Stock_news/All_external.csv"
NEWS_HF_REPO = "Zihan1004/FNSPID"
NEWS_HF_REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
# CORRECTED 2026-09-09 (R03d). This field previously held
# dde529189c87048a8be1f73d17ecd5e211fc7809032e60c72cd22645f111c470, which is
# **not** a SHA-256 of the file: it is HuggingFace's `xetHash`, a Xet
# content-addressing digest, recorded at B04 under a SHA-256 name. The value
# below is the real SHA-256 of the file's bytes, confirmed two independent ways:
# HF's paths-info API reports it as the Git LFS `oid` for this exact revision
# (LFS OIDs are sha256-of-content), and streaming the 5.73 GB file and hashing
# it as it was consumed produced the same digest.
#
# Nothing about the data changed. The revision, the byte length and the content
# are identical; only the pin was mislabelled, and R02's verification is what
# surfaced it -- the first assembly run after R02 refused to publish and wrote
# nothing, which is the behaviour that guard exists for.
NEWS_FILE_SHA256 = "5d4c018036bd82ca821da71b7a9c0c7db3289642e0fc6f897ea69f4a0c5135c3"
# Kept for traceability: the value B04 recorded, and what it actually is.
NEWS_FILE_XET_HASH = "dde529189c87048a8be1f73d17ecd5e211fc7809032e60c72cd22645f111c470"
NEWS_FILE_BYTES = 5_731_397_037
NEWS_LICENCE = "CC BY-NC 4.0 (non-commercial)"

# Rows are kept only when their Url host matches. This is mandatory, not
# hygiene: lenta.ru is a Russian-language general news site present in both
# FNSPID files, and scoring it with an English financial model yields numbers
# with no meaning. Taking "all headlines" is not an available option.
NEWS_SOURCE_DOMAINS = ("benzinga.com",)
# Candidates for a later pooled sensitivity check, recorded now so the choice
# cannot be made after seeing a coefficient:
NEWS_SOURCE_DOMAINS_POOLED = ("benzinga.com", "zacks.com", "seekingalpha.com")

# Date-only fallback: ACTIVE. No FNSPID sub-corpus is both intraday-stamped and
# relevant to US equities -- the relevant blocks are 96-100% midnight, and the
# one intraday block (Reuters) is a global general newswire with no ticker tags,
# weighted to European hours. So RQ2 (same-day association) is dropped and a
# headline dated d maps to the next trading session at or after d+1.
#
# Implemented at B09: align.map_date_to_session applies the deferred mapping,
# aggregate_daily dispatches on this flag, inference.contemporaneous refuses to
# run while RQ2_ADMISSIBLE is False, and run_all.py writes no Table 2.
DATE_ONLY_FALLBACK = True
DATE_ONLY_FALLBACK_IMPLEMENTED = True
RQ2_ADMISSIBLE = False

# ------------------------------------------------------ D3 market data ----

MARKET_TICKER = "SPY"
VIX_TICKER = "^VIX"          # context plots only, never a regressor
MARKET_CALENDAR = "NYSE"

# ----------------------------------------------------- sample window ----
# PROVISIONAL (B04). Ten full years, ~2,500 sessions, against a >=1,250 target.
# 2009 excluded as a partial ramp-up year; 2020 excluded because its timestamp
# regime changes (25% intraday), which makes it a different measurement.
#
# Provisional because it is bounded from per-source spans in a *cluster* sample:
# the byte-range slices are ticker-ordered, so market-wide headlines-per-session
# and coverage stability cannot be estimated from them. Freeze this after the
# corpus is assembled and before any sentiment-return coefficient is examined,
# and log the freeze.
SAMPLE_START = "2010-01-01"
SAMPLE_END = "2019-12-31"
# FROZEN on the assembled corpus (census: docs/data-audit-fnspid.md §11), before
# any sentiment-return coefficient was examined. 2,516 sessions; 2 zero-news
# sessions in ten years; no year falls below the stability tolerance.
SAMPLE_WINDOW_PROVISIONAL = False
SAMPLE_WINDOW_FROZEN_ON = "2026-09-09"
# 2010 carries ~half the median coverage of the other years (median 170 vs 344),
# which roughly doubles the sampling variance of S_t that year. It clears the
# tolerance and is kept, but the drop-2010 variant is declared HERE, in advance,
# as a prespecified sensitivity so it cannot be chosen after seeing a result.
SENSITIVITY_DROP_FIRST_YEAR = "2011-01-01"
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
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"   # B04, inspected
# Published label order, verified from the checkpoint's config.json. It is NOT
# negative/neutral/positive: code that assumes an index order silently inverts
# every score. src/scoring.py resolves positions from model.config.id2label;
# B12 must preserve that when it exposes class probabilities.
FINBERT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
# The checkpoint post-dates the whole 2010-2019 window. This is retrospective
# measurement and is never described as tooling available at the time (P17).
FINBERT_POSTDATES_SAMPLE = True
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 64      # measured at R11: truncates 3,921 headlines (0.45%)

# How batches are formed. "length_sorted" groups headlines of similar token
# length so batches are not padded to an outlier's length: measured 2.20x
# faster on this corpus, with padding falling from 2.33x the real token count
# to 1.01x. It is NOT bit-identical to "input" order -- batch composition moves
# individual scores by up to 4.2e-6 -- so this setting is part of the FinBERT
# fingerprint and changing it invalidates the cached column (R11).
FINBERT_BATCH_ORDER = "length_sorted"

# Acquired from the CSV linked by Notre Dame SRAF on 2026-09-10.
# This release postdates the sample and is used for retrospective measurement.
# Full acquisition evidence: docs/lm-dictionary-provenance.md.
LM_DICT_VERSION = "1993-2025 (March 2026)"
LM_DICT_SHA256 = "e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d"

# --------------------------------------------- D8/D9 daily aggregation ----

# How S_t summarises a session's headline scores. Read by
# `align.aggregate_daily` and recorded in the panel's provenance, so a median
# run cannot be mistaken for the primary one. Until R12 this constant was inert
# -- `aggregate_daily` hardcoded the mean -- so changing it changed nothing
# while appearing to change the specification (audit A15).
# D8 fixes "mean" as primary; "median" is the Stage 6 robustness variant and is
# selected by passing it explicitly, never by editing this line.
AGG = "mean"                 # equal-weighted mean of the day's headline scores
AGG_CHOICES = ("mean", "median")
MIN_HEADLINES_FOR_DISPERSION = 5   # d_t is NaN below this
# Trading days with n_t == 0 are dropped from every regression, and the count
# is reported. Days with 1 <= n_t < 5 keep S_t but are excluded from d_t specs.

# ------------------------------------------------- D10 standard errors ----

NW_MAXLAGS = 5
NW_MAXLAGS_SENSITIVITY = 10  # Stage 6

# What the prespecified bandwidth L = 5 counts (A09 / amendment M7, settled at
# R07c on 2026-09-10). "One trading week" is a statement about exchange
# sessions, so a lag counts sessions: pairs more than L sessions apart get zero
# weight even when they are adjacent rows of a gapped analysis sample. The
# alternative -- lag = L retained ROWS, which is what statsmodels computes after
# an index reset, and whose own documentation assumes equally spaced periods --
# is always computed and reported alongside as a sensitivity, never substituted.
# See docs/hac-spacing-decision.md.
HAC_CONVENTION = "session_indexed"

# ------------------------------------------------------ D11 lag family ----

HORIZONS = (1, 2, 3, 4, 5)
FDR_Q = 0.05                 # BH across the 14 secondary return tests; BY sensitivity

# ------------------------------------------- D12 timing diagnostic ----

# The block-resampling placebo was removed at R08c: it drew blocks WITH
# replacement, so it was not a permutation, and its output was labelled and
# plotted as a p-value, which the inference protocol forbids. Its settings
# (`PERMUTATION_DRAWS`, `PERMUTATION_BLOCK`) are deleted with it. The
# replacement is a full circular shift over the retained analysis rows -- an
# exhaustive, deterministic sweep of every k in 1..n-1, so there is nothing to
# draw and no seed to set.
TIMING_DIAGNOSTIC_HORIZON = 1
TIMING_TIE_PRECISION_BPS = 0.1   # the reporting precision ties are judged at

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
SESOI_BPS = 5.0             # Small-effect yardstick for association, not profitability
TRANSACTION_COST_BPS = SESOI_BPS  # Compatibility alias for historical configurations

# --------------------------------------------- D16 seeds / reproducibility ----

SEED = 20260830

LABELS = ("negative", "neutral", "positive")

# ------------------------------------------------ B11 / R06a annotation ------
# The blind evaluation sample. These artifacts are TRACKED in Git, unlike
# everything under data/raw|interim|processed: the labels are the study's own
# experimental data and the most expensive thing in the project to reproduce.
ANNOTATION_DIR = ROOT / "data" / "annotation"

ANNOTATION_N_TOTAL = 800
ANNOTATION_N_CALIBRATION = 200      # thresholds, rubric refinement, the pilot
ANNOTATION_N_EVALUATION = 600       # every reported Act 1 number. Touched once.
ANNOTATION_PILOT_N = 60             # validation protocol section 8
ANNOTATION_SECOND_FRACTION = 0.20   # independent second annotator, for kappa

# Presentation order is shuffled so neither time order nor source clusters cue
# the annotator. Distinct from SEED so that reshuffling the running order can
# never disturb which rows were drawn.
ANNOTATION_ORDER_SEED = 20260831

# Two headlines are in the same ARTICLE GROUP when they share a normalized text
# or their token sets overlap at least this much. Unlike the dedup pass this is
# NOT windowed: a story republished ten days later survives dedup twice but is
# still one article for leakage purposes, and a group must never straddle the
# calibration/evaluation boundary.
ANNOTATION_GROUP_OVERLAP = 0.90

# Filled only after the human calibration labels have been validated. Record
# the calibration provenance and thresholds before scoring evaluation text.
# Frozen 2026-09-10 from the 200 CALIBRATION items only, never the evaluation
# set. See report/tables/calibration_record.json, which also records that the
# LM band is one of 1,600 grid pairs tied at the best macro-F1 -- all of which
# classify identically, because LM takes only three values on headline text.
VALIDATION_THRESHOLDS = {"lm": [-1.0, 0.0], "vader": [-0.125, 0.2250000000000001]}
