# Data audit: FNSPID

Date: 2026-09-09. Increment: **B03**. Source notes and verified facts.

Scope: a bounded sample of the FNSPID news dataset, audited for the questions B03 requires — publication versus update/collection time, timezone evidence, duplication, company/source concentration, and coverage. **No return relationships were examined**; nothing in this document was chosen by looking at a p-value.

Companion: [B04 decisions](#8-b04-decisions-universe-window-and-artifacts) below, recorded into `config.py`.

---

## 1. What was downloaded, and why only a sample

| Field | Value |
|---|---|
| Repository | `Zihan1004/FNSPID` (HuggingFace, dataset) |
| Repo revision | `bf9189c41527198897d1af3e17b1a0095279fc45` |
| Last modified | 2024-04-09 |
| Licence | CC BY-NC 4.0 — **non-commercial**; commercial use prohibited without authorisation |
| Citation | Dong, Fan & Peng (2024), arXiv:2402.06698 |
| File audited | `Stock_news/All_external.csv`, 5,731,397,037 bytes (5.7 GB) |
| File sha256 (etag) | `dde529189c87048a8be1f73d17ecd5e211fc7809032e60c72cd22645f111c470` |
| Second file probed | `Stock_news/nasdaq_exteral_data.csv`, 23.2 GB |

Neither file was downloaded whole. `All_external.csv` was sampled with **24 HTTP range slices of 6 MB each, evenly spaced** — 144 MB, 2.5% of the file, 356,123 parsed rows. `nasdaq_exteral_data.csv` got a lighter 8 × 4 MB probe, 6,671 rows.

**The sample is a cluster sample, not a random one, and this constrains what may be concluded from it.** The file is physically ordered by source block and then by ticker, so a slice is a contiguous run of rows from one source and a handful of tickers. That makes the sample sound for *per-source* properties — timestamp behaviour, language, content type, column availability — and **unsound for market-wide rates**, such as headlines per trading day across all tickers. No statistic of the latter kind is claimed below.

Reproduce with `python data/raw/download.py --fnspid-sample --slices 24 --slice-mb 6`.

## 2. Schema, as verified from the bytes

```text
Date, Article_title, Stock_symbol, Url, Publisher, Author,
Article, Lsa_summary, Luhn_summary, Textrank_summary, Lexrank_summary
```

`Date` values are formatted `YYYY-MM-DD HH:MM:SS UTC`. **99.95% conform exactly**; the 180 that do not are fragments of article body text (`"$1.34 billion dollar bridge loan last week as the value of"`, `"with the matter.  Bakrie & Brothers"`), which proves that the `Article` field contains **embedded newlines**. Any loader that splits on newlines rather than parsing CSV quoting will silently corrupt rows. The loader drops non-conforming `Date` rows explicitly and reports the count.

## 3. Timezone evidence

The timezone is **stated in the data itself** — every conforming value carries a literal `UTC` suffix — rather than inferred from the distribution. That is the strongest form of the evidence criterion B03 asks for, and it is the one thing about FNSPID's timestamps that is unambiguous.

It is also the *only* thing. The suffix establishes the zone, not the meaning.

## 4. Publication versus update versus collection time — **UNRESOLVED**

Neither the [dataset card](https://huggingface.co/datasets/Zihan1004/FNSPID) nor the [GitHub repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) documents what `Date` represents. There is no column-level documentation at all: no statement that it is the publication timestamp, the last-update timestamp, or the time the scraper collected the row.

**This is recorded as an unknown, not resolved by assumption** (P13). It matters because a correctly localised timestamp can still describe a moment other than when the content became readable, and the entire information-boundary argument in Act 2 depends on that moment.

Its practical force is limited by §5: for the sub-corpus actually selected, the timestamps are date-only, so the question of intraday publication semantics does not arise. It would become decisive if the intraday Reuters block were ever used.

## 5. The file is five corpora, not one — and this is the decisive finding

Classifying rows by URL domain reveals that `All_external.csv` concatenates heterogeneous sub-corpora with **different schemas, different languages, and different timestamp behaviour**:

| Source | Rows sampled | Midnight share | Distinct minutes-of-day | Ticker tag | Span |
|---|---:|---:|---:|---:|---|
| reuters.com | 213,568 | **0.2%** | **1440** | 0% | 2007–2016 |
| benzinga.com | 57,207 | 96.7% | 697 | 100% | 2009–2020 |
| seekingalpha.com | 29,665 | 100% | 1 | 100% | 2014–2019 |
| **lenta.ru** | 19,075 | 100% | 1 | 0% | 2003–2018 |
| zacks.com | 13,707 | 100% | 1 | 100% | 2011–2020 |
| bloomberg.com | 12,014 | 100% | 1 | 0% | 2007–2013 |
| 13 smaller domains | ~10,700 | 100% | 1 | ~100% | 2010–2020 |

Three consequences.

**(a) The intraday gate must be applied per source, not to the file.** Judged as one corpus, FNSPID looks partly intraday. It is not: one block is intraday and every other block is date-only.

**(b) `lenta.ru` is a Russian-language general news site.** Its headlines are Cyrillic and its subject matter is not financial — *«Зеленый день» заменит в России «Черную пятницу»*, *«Кристен Стюарт заявила о готовности сделать предложение своей девушке»*. It is present in both audited files. Passing this text to an English financial sentiment model would produce scores, and those scores would be meaningless. **Any use of FNSPID must filter by source explicitly**; taking "all headlines" is not an option.

**(c) The only intraday block is the wrong content.** Reuters carries genuine intraday stamps — 1440 distinct minutes, 0.2% midnight — but the sampled titles are a **global general newswire**: *"Fashion comes first at Royal Ascot Ladies' Day"*, *"Austrian rail chief should become chancellor next week"*, *"REG-Baillie Gifford Japan - Net Asset Value(s)"*, *"Heian Ceremony <2344.OS>-2011/12 group forecast"*. It carries no ticker tags, and its hour-of-day distribution peaks at 08:00–13:00 UTC (03:00–08:00 ET) — **European trading hours, not US ones**. It is not a measurement of US equity news tone, and relevance is a selection criterion that must be applied before any return relationship is examined (P20).

### Benzinga's intraday share by year

The one block that is both US-equity and ticker-tagged is date-only for nearly all of its span:

| Year | 2009 | 2011 | 2013 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Midnight share | 100% | 100% | 100% | 98% | 97% | 97% | 98% | 97% | 75% |

Intraday stamps appear only as a minority tail, and only meaningfully in 2020 — well under one year of sessions against a target of ≥ 1,250.

### The second file is no better

`nasdaq_exteral_data.csv` (23.2 GB, 2007–2023): **98.6% midnight, 4 distinct minutes-of-day** across the probe, and it contains the same `lenta.ru` Russian content. It adds Nasdaq-sourced articles and a longer span, but not usable intraday timestamps.

## 6. Duplication and near-duplication

Within the ticker-tagged blocks, syndicated and templated headlines recur heavily — `"Stocks That Hit 52-Week Highs On Friday"`, `"71 Biggest Movers From Friday"`, and per-CEO templates such as `"CAPE BANCORP, INC. (CBNJ) CEO Michael D Devlin buys 1,081 Shares"` differing only in a share count.

A precise dedup rate is **not** quoted here. The sample is a cluster sample over tickers, so its duplication rate is not the corpus's: exact repeats of a market-wide roundup headline across many tickers are largely invisible in a sample that contains only some of those tickers. The rate is computed on the assembled corpus at the point the corpus is built, and reported then.

Structural note for the dedup rule: the same Benzinga roundup is emitted **once per tagged ticker**, so a market-level aggregate that does not deduplicate across tickers will weight one editorial act by the number of symbols it mentions.

## 7. Concentration and coverage — partially unresolved

- **Source concentration** is measured and is the §5 table. It is the finding that drives the universe decision.
- **Company concentration** cannot be estimated from this sample. Byte-offset slicing over a ticker-ordered file returns whole ticker blocks, so the per-ticker counts are an artifact of which offsets were drawn. Deferred to the point the selected corpus is assembled.
- **Headlines per trading day, market-wide** — same reason, deferred. The `n_t` distribution and the zero-news day count are properties of the assembled corpus.
- **Coverage stability across years** is visible per source in §5 and is sufficient to bound the candidate window; it is re-checked on the assembled corpus before the window is finally frozen.

## 8. B04 decisions: universe, window, and artifacts

### Decision 1 — same-day analysis (RQ2) is inadmissible

No sub-corpus is simultaneously intraday-timestamped and relevant to US equity news. The relevant blocks are date-only; the intraday block is a global general newswire without ticker tags.

**RQ2 is dropped.** The date-only fallback applies: a headline dated `d` is mapped to the **next trading session at or after `d`+1**, so that no headline can inform the return of a session that had already closed. `config.DATE_ONLY_FALLBACK = True`. This flag is currently **inert in the pipeline** — implementing it is B09, and until then no dependent scoring or panel build may run (P14).

This also demotes what Act 2 can claim: with date-only stamps there is no verified information boundary within a day, so even the next-day association rests on the assumption that a headline dated `d` was available before the close of the mapped session. That assumption is stated in the report, not buried.

### Decision 2 — universe

**Benzinga sub-corpus of `All_external.csv`**, selected on relevance and coverage before any return relationship was examined:

- US-equity news with 100% ticker tagging (the only blocks that have it, alongside Zacks/SeekingAlpha);
- English;
- the longest continuous span among the relevant blocks (2009–2020);
- one editorial source, so the source mix cannot drift underneath the aggregate — a real advantage over pooling.

Explicitly **excluded, with reasons**: `lenta.ru` (Russian-language, non-financial); `reuters.com` (global general newswire, no ticker tags, European-hour weighted); `bloomberg.com` (no ticker tags, date-only, ends 2013); Zacks/SeekingAlpha/the smaller domains (candidates for a pooled sensitivity check later, but they change the source mix over time).

Whether to add Zacks and SeekingAlpha as a pooled variant is a **later sensitivity decision**, and it is recorded now so that it cannot be made after seeing a coefficient.

### Decision 3 — window: **provisional 2010-01-01 to 2019-12-31**

Ten calendar years, ~2,500 trading sessions, comfortably above the ≥1,250 target. 2009 is excluded as a partial ramp-up year and 2020 as a partial final year with a mixed timestamp regime — its 25% intraday share makes it a different measurement from the rest.

**Provisional, and marked as such in `config.py`.** It is a bound derived from per-source spans in a cluster sample; the `n_t` distribution and zero-news day count that would confirm coverage stability can only be computed on the assembled corpus. The window is frozen at that point, before any sentiment–return coefficient is examined, and the freeze is logged.

### Decision 4 — artifacts pinned

| Artifact | Identifier | Status |
|---|---|---|
| News dataset | `Zihan1004/FNSPID` @ `bf9189c41527198897d1af3e17b1a0095279fc45` | **Pinned** — inspected |
| News file | `Stock_news/All_external.csv`, sha256 `dde5291…f111c470`, 5,731,397,037 bytes | **Pinned** — verified |
| FinBERT | `ProsusAI/finbert` @ `4556d13015211d73dccd3fdd39d39232506f3e43` (modified 2023-05-23) | **Pinned** — inspected |
| FinBERT labels | `id2label = {0: positive, 1: negative, 2: neutral}` | **Verified** — see note |
| Loughran–McDonald | — | **UNRESOLVED**: SRAF distributes it behind a non-stable link; must be fetched by hand, then recorded in `config.LM_DICT_VERSION` |
| Financial PhraseBank | — | Not required unless §2 of the [validation protocol](validation-protocol.md) falls back to it |

**FinBERT label-order note.** The published order is `0: positive, 1: negative, 2: neutral` — *not* the conventional negative/neutral/positive. Code that assumes an index order silently inverts the sign of every score. `src/scoring.py` resolves the positions from `model.config.id2label`, which is correct; B12 must keep that property when it exposes class probabilities.

**Model availability.** The FinBERT checkpoint post-dates the entire 2010–2019 window. This is legitimate **retrospective measurement** and is not, and will not be described as, a reconstruction of tools available to an investor at the time (P17).

## 9. Verified facts and open unknowns

**Verified**

1. Timezone is explicit in the data (`UTC` suffix), on 99.95% of rows.
2. `All_external.csv` is five-plus heterogeneous sub-corpora, separable by URL domain.
3. Reuters is the only intraday block: 1440 distinct minutes, 0.2% midnight.
4. Reuters is a global general newswire, no ticker tags, European-hour weighted.
5. Benzinga is 96–100% date-only from 2009 through 2019; 75% in 2020.
6. `nasdaq_exteral_data.csv` is 98.6% midnight and also contains `lenta.ru`.
7. `lenta.ru` is Russian-language, non-financial, present in both files.
8. The `Article` field contains embedded newlines; naive line-splitting corrupts rows.
9. Repo, file, and model identifiers are as pinned in §8.
10. FinBERT's label order is positive/negative/neutral.

**Unknown**

1. **What `Date` means** — publication, update, or collection. Undocumented upstream. Limited practical force given the date-only selection, decisive if Reuters were ever used.
2. Corpus-wide duplication rate — needs the assembled corpus.
3. Company concentration and market-wide headlines per session — cannot be estimated from a ticker-clustered sample.
4. Whether Benzinga coverage is stable enough across 2010–2019 to freeze the window — checked on the assembled corpus.
5. Whether a headline dated `d` was genuinely available before the mapped session's close — unverifiable with date-only stamps; carried as a stated assumption.
6. The Loughran–McDonald release to pin.

## 10. Consequences for the plan

| Consequence | Where it lands |
|---|---|
| RQ2 dropped; date-only mapping required and currently unimplemented | **B09**, now live rather than conditional |
| Source filtering is mandatory in the loader, not optional hygiene | B03 follow-up in `src/data.py` |
| CSV must be parsed with quoting; embedded newlines are real | `src/data.py` |
| Cross-ticker duplication of one editorial act | dedup rule, before aggregation |
| Actual-session-close work (B07) has no bearing on a date-only corpus | B05/B07 scope shrinks; keep the correction, drop its urgency |
| Window is provisional until the corpus is assembled | B04 revisit, logged when frozen |
| Assembling the corpus needs the full 5.7 GB file, filtered to Benzinga | its own instructed increment; not a side effect of a small fix |
