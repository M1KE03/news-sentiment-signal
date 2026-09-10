# How to label — rubric v3

Two files, in this order. **Calibration first, evaluation second.** They serve different purposes and the order matters: the calibration items fit the lexicon thresholds, and if a rubric problem surfaces it is far better to find it in the set that produces no reported number.

| Step | File | Items | Rough time |
|---|---|---:|---:|
| 1 | `worksheet_v2_calibration.csv` | 200 | ~70 min |
| 2 | `worksheet_v2_evaluation.csv` | 600 | ~200 min |

Both are already in the shuffled presentation order. **Do not sort, filter or reorder them.** Keep `headline_id` and `text` exactly as they are; fill only the four answer columns.

---

## The four columns

| Column | What to enter |
|---|---|
| `label` | exactly `positive`, `negative`, `neutral`, or `unusable` |
| `mixed` | `1` if the headline is genuinely balanced good-and-bad, else `0` |
| `hard` | `1` if the rubric did **not** decide it, else `0` |
| `notes` | short, optional — but useful when `hard` is `1` |

Spelling matters: `positive`, not `postive`. Ingestion rejects anything else.

## The question

> Read the headline **alone**. For the company or market it concerns, does this read as good news, bad news, or neither, for the value of the relevant equity?

Headline text only. No article bodies, no price charts, no knowledge of what happened next, and **do not open `interim/scores.parquet`** — model scores exist for every one of these items, and your provenance record will assert that you did not consult them.

## What changed in v2, and why

Your pilot flagged **34 of 60 hard**, including 19 of 20 neutrals. Six new rules cover the types your notes identified. **If one of rules 11–16 applies, it gives you an answer — set `hard` to `0`.**

**But do not try to drive the `hard` rate to zero.** Some of what you hit is not a gap in the rubric — it is genuine ambiguity in the headlines, worth measuring rather than suppressing. For comparison: Financial PhraseBank, the field's standard dataset and FinBERT's own training data, reached unanimous agreement on only **2,264 of 4,846 sentences (46.7%)** using *5 to 8* finance-literate annotators per sentence. More than half its items had at least one dissenter. Your 57% is the same phenomenon, visible because one annotator flagging it is more honest than majority voting hiding it.

So `hard = 1` is a real answer, not a failure. Act 1 now reports every metric twice — on the full evaluation set and on the `hard = 0` subset — so the flag does analytical work. Use it whenever the rubric genuinely does not decide, and `hard = 0` whenever it does.

| # | Type | Rule |
|---|---|---|
| 11 | Lists, screens, roundups | `neutral` unless the headline asserts a direction. *"Top 5 Small-Cap Stocks"* → `neutral`. *"3 Stocks Set to Spring"* → `positive`. Fund position disclosures (*"Among Shumway Buys"*) → `neutral`. |
| 12 | Legal actions | `negative` for the party **facing** it, `neutral` for the party **bringing** it — judge the headline's subject. Settlement or dismissal → `positive` for the party that faced it. **You were right to question this one.** I checked: event studies find defendants take material negative abnormal returns around filing, while **plaintiffs' returns are indistinguishable from zero** — the claim's value is already in the plaintiff's price. The rule matches the data. |
| 13 | Personnel changes | `neutral` unless the headline states an adverse **cause** — then label the cause. *"Appoints New CFO"* → `neutral`. *"CFO Resigns Amid Probe"* → `negative` (the probe is the news). **A bare forced departure is NOT negative:** the literature finds the market often reads an ousting as *positive* — removal of an underperformer. *"Board Ousts CEO"* with no cause given → `neutral` + `hard = 1`. |
| 14 | Capital actions | Effect on existing holders. Share offering → `negative`. Buyback or dividend rise → `positive`. Dividend cut → `negative`. Debt raise → `neutral` unless distress or a favourable rate is named. **Reliability differs:** buybacks and dividend changes are strongly signed, but offerings draw a *positive* reaction **30–40%** of the time — so an offering that names a favourable use of proceeds is `hard = 1`. |
| 15a | Product / operating approvals | Approval or clearance → `positive`. Rejection, or a demand for more data → `negative`. Filing or ongoing review → `neutral`. |
| 15b | **Mergers and acquisitions** | **Depends on which side the subject is.** Target (being acquired, receiving a bid) → `positive`; targets earn **+15% to +30%** on announcement. Acquirer (buying, bidding) → `neutral`; acquirer returns are mixed to slightly negative and not reliably signed. A collapsed deal reverses it: `negative` for the target, `neutral` for the acquirer. A regulatory clearance *of a deal* carries the deal's asymmetry, not a direction of its own. **I had this wrong in v2:** *"EU Allows Marriott's Acquisition of Starwood"* is `neutral` — Marriott is the acquirer. It would be `positive` for Starwood. |
| 16 | Promotional PR | Adjectives with no verifiable outcome (*"productive"*, *"strategic"*) → `neutral`. If a concrete outcome is named, take its direction. |

**Two v2 rules were wrong and are corrected in v3.** You asked whether the categories should be checked against what actually happens to prices. They should have been, and doing it caught rule 13 (forced departures) and rule 15 (M&A direction) pointing the wrong way. What the evidence is *not* used for: overriding your reading with a category's historical average. That would make the labels a function of returns, which is Act 2's outcome variable, and rule 3 already forbids the same move on a smaller scale. Full reasoning: [`validation-protocol.md`](../../docs/validation-protocol.md) §6b.

**Rule 5 also clarified:** reported results, not just guidance, are judged against the expectation named in the headline — or, if none is named, **against the prior period**. So *"Q4 EPS €(0.16) vs €(0.19) in Same Qtr. Last Year"* is `positive`: the loss narrowed. Only a result with no comparison at all is `neutral`.

## The v1 rules still in force

1. **The comparative is the signal.** *"Profit warning smaller than feared"* → `positive`.
2. **Direction attaches to the subject.** *"Costs fell"* → `positive`. *"Provisions rose"* → `negative`.
3. **Reported price moves are `neutral`.** *"Shares fall 4%"* describes the market's reaction, not company news. This one deliberately disagrees with all three scorers.
4. **Analyst actions take the analyst's direction.** Upgrade → `positive`, target cut → `negative`.
6. **Balanced headlines** → `neutral` **and** `mixed = 1`.
7. **Questions and speculation** → `neutral` unless the headline asserts the answer.
8. **Macro headlines take the direction for equities.** *"Jobless claims fall"* → `positive`.
9. **Do not guess.** If nothing decides it: `neutral` + `hard = 1`.
10. **One pass.** Do not revisit earlier labels after seeing later ones.

Full text: [`docs/validation-protocol.md`](../../docs/validation-protocol.md) §5.

## Saving

Save each file **as UTF-8 CSV**, keeping its name.

⚠️ Your pilot file came back in the Windows ANSI codepage, which turned a `€` into `Û`. It did no harm — the text column is checked against the frozen export — but in Excel use **Save As → CSV UTF-8 (Comma delimited)**, not plain "CSV". Also make sure no extra header row is added above `headline_id`.

## When you finish

Fill `provenance_v2.json`. Every `null` and `TO BE COMPLETED` field is a declaration only you can make, and the file is **rejected until all are replaced** — that refusal is deliberate, not a bug. The `_`-prefixed keys explain each field and are ignored by the reader.

- `annotator_role` — a role, not a name
- `label_source` — `"human"`
- `blind_to_model_outputs` — `true` if you did not open the score cache
- `independent_of_analyst` — `false` (you are running the analysis)
- `scores_existed` — `true` (the pass finished 2026-09-10, before labelling)
- `sessions` — one entry per sitting, with time zones
- `deviations` — three are pre-recorded for the v1→v2 change; add any rubric question that arises

## Stop points

After **calibration (200)**, stop and tell me. I check the `hard` rate and class balance before you spend three hours on the evaluation set. If v2 has fixed the problem, the hard rate should fall well below the pilot's 57%; if it has not, better to know at 200 than at 800.
