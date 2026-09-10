# What this study found

A plain-language account, with the mathematics behind each claim and what each number actually means.

Every figure here comes from a file in `report/tables/`. Nothing is rounded in a direction that flatters it.

---

## The two questions

The study asked two things and deliberately kept them apart:

1. **Can a finance-specific AI model read financial headlines better than simpler word-counting tools?**
2. **Does the overall mood of a day's financial news tell you anything about how the stock market moves the next day?**

They are separate because a good reader of language is not automatically a good predictor of prices, and the study was designed so that a "no" to either could be reported honestly.

**Data.** 869,183 Benzinga headlines from 2010–2019, covering 2,516 trading days. 800 of them were labelled by hand — good news, bad news, or neither — by a person who could not see any model's output. The market data is SPY, the main S&P 500 fund.

---

## Finding 1 — Yes, the AI reads headlines better. But for an unglamorous reason.

Three tools each classified 600 headlines. The hand labels were the answer key.

| Tool | Score (macro-F1) |
|---|---:|
| **FinBERT** — AI model trained on financial text | **0.594** |
| Loughran–McDonald — word list built for finance | 0.493 |
| VADER — general-purpose word list | 0.424 |

### What "macro-F1" means, and why it was chosen

For each of the three categories separately, we compute the **F1 score** — a balance between *"when it says positive, is it right?"* (precision) and *"of the truly positive ones, how many did it catch?"* (recall):

```
F1 = 2 × (precision × recall) / (precision + recall)
```

Then we **average the three F1 scores equally**. That average is macro-F1. 1.0 is perfect; about 0.33 is random guessing.

**Why the equal weighting matters — this is the whole reason the finding exists.** Half the headlines really are neutral. A tool that answers "neutral" to almost everything would score decently on plain accuracy simply by riding the majority. Macro-F1 refuses to let it: being poor at one category out of three drags the average down no matter how large that category is. Plain accuracy would have hidden what we found.

### The comparison

> **FinBERT − Loughran–McDonald = +0.102**
> **95% interval: [+0.050, +0.153]**

**What the interval means.** We tested 600 headlines, not every headline that exists. A different 600 would have given a slightly different number. The interval is the range of values consistent with what we saw.

**Its impact: the interval never touches zero.** Had it included zero, we could not have ruled out that the two tools are equally good. It doesn't, so the difference is real and not an accident of which headlines we happened to draw.

**How the interval was computed, and why not the obvious way.** We used a *paired bootstrap resampling article groups*. In plain terms: repeatedly rebuild the 600-headline sample by drawing **whole news stories** at random with replacement, 10,000 times, recomputing the gap each time, and take the middle 95% of the results.

The detail that matters is **whole stories, not individual headlines**. The same story often appears as several near-identical headlines. Treating those as independent evidence would make the sample look bigger than it is and the interval **falsely narrow** — we would claim more certainty than we have.

### Where the difference actually comes from

Here is the part that changes the interpretation. Broken down by category:

| Tool | Bad news | Neutral | **Good news** |
|---|---:|---:|---:|
| FinBERT | 0.573 | 0.664 | **0.546** |
| Loughran–McDonald | 0.557 | 0.658 | **0.263** |

On bad news and neutral news the two are **within 0.02 of each other** — effectively tied. **The entire gap is good news.**

The confusion matrix says it directly:

> Of 195 genuinely positive headlines, Loughran–McDonald called **159 of them "neutral"** — **81.5%**.

**Why.** The Loughran–McDonald list was built to scan annual reports for risk language. It contains **2,345 negative words and 347 positive ones** — a ratio of nearly 7 to 1. Its score is:

```
score = (positive words found − negative words found) / (total sentiment words found)
```

On a short headline this usually finds nothing at all. Measured on our data: the score takes only **three distinct values** (−1, 0, +1), and **78.3% of headlines score exactly 0** — no matching word anywhere in the text.

**The impact:** it isn't reading badly. It has almost no positive vocabulary to react to, so good news slides into "neutral" by default. On long documents, where positive words eventually appear, this would matter far less.

**So the finding is narrower than the headline number:** *a risk dictionary designed for annual reports performs poorly on short headlines, specifically on good news.* That is a narrower and more useful claim than "AI understands language better", which the raw 0.102 would have implied on its own.

### What weakens this finding

- **Plain accuracy tells a much weaker story.** The accuracy gap is +0.049, interval **[0.000, 0.097]** — the lower end sits exactly on zero. Accuracy is dominated by the neutral majority that both tools handle similarly. We report both and do not quote only the stronger one.
- **The answer key may favour FinBERT.** The labelling asked *"would this news move the price?"* — the same framing used to train FinBERT. The word lists measure something subtly different: whether the language itself is positive or negative. Some unknown part of +0.102 is that alignment rather than skill. It cannot be fixed by rewording the question, because the opposite framing would favour the word lists just as unfairly.
- **The tools do not have equal resolution.** On 198 test headlines, FinBERT produced 198 different scores, VADER 54, Loughran–McDonald **3**.
- **One labeller, who is also the analyst.** No second opinion, so no measurement of how often the answer key itself is wrong.

---

## Finding 2 — No, daily news mood tells you nothing about tomorrow's market

> **Effect: −1.74 basis points per standard deviation of news tone**
> **95% interval: [−4.90, +1.41]**

### What a basis point is

One **basis point** is one hundredth of one percent (0.01%). On a $100 investment, **1.74 basis points is 1.7 cents.**

**Why that is small.** SPY moves about **93 basis points** on a typical day — roughly 93 cents on $100. The effect we were looking for is about **one fifty-third of an ordinary day's movement**.

### What "per standard deviation of tone" means

Tone was rescaled so that 1 unit = one typical day's variation in news mood. **The impact:** the number answers a concrete question — *"if today's news is unusually positive, by about as much as news mood normally varies, what happens tomorrow?"* Answer: **−1.7 basis points, and we cannot distinguish that from nothing.**

### Reading the interval

The interval **includes zero**, so we cannot claim any association.

But it is also **narrow enough to sit inside ±5 basis points**, which lets us say something stronger than "we found nothing": *whatever effect exists, it is smaller than 5 basis points.*

**Why 5 basis points is the benchmark.** That is roughly the round-trip cost of trading a large fund — about 5 cents on $100. An effect smaller than your trading cost cannot be acted upon. **This is not a claim about profits**, just a sense of scale, and it was fixed in advance so it could not be moved to suit the result.

### What this claim rests on

This claim only *just* holds. The interval ends at **−4.9031**, and −5.0 would have broken it — a margin of **0.1 basis points**, which is the precision we report to. We ran the analysis five different ways (varying a statistical setting called the bandwidth) and the conclusion held every time, but **5 of 7 checks landed on that same boundary**.

Additionally, when the daily mood is computed as a **median** instead of a **mean**, the estimate falls to **−0.17 basis points** — essentially zero.

So we split the claim in two and report them at different strengths:

- **"No association detected" — solid.** True under every method tried.
- **"Small enough to rule out anything above 5 basis points" — fragile.** True by the stated rule, but by a hair.

Nothing else in Act 2 came close to significance either: 14 further tests across three tools and five time horizons, none rejecting.

---

## Why the two findings do not contradict each other

FinBERT reads headlines clearly better. Yet when we asked which tool better predicts the market, **we could not tell them apart at all**:

> FinBERT vs Loughran–McDonald on market prediction: **−2.43 basis points, interval [−6.88, +2.03]** — far too wide to separate them.

That looks like a contradiction. It isn't, and we worked out why **before running either test**.

### The mathematics of measurement error

Suppose there is some real "information content" in the news, call it `Z`, and it genuinely moves tomorrow's return:

```
tomorrow's return  =  θ × Z  +  random noise
```

No tool measures `Z` perfectly. Each measures it with error:

```
tool's score  =  Z  +  measurement error
```

Standard statistics then gives the result: when you use the imperfect score to estimate `θ`, you don't get `θ`. You get `θ` shrunk toward zero:

```
what you measure  =  θ  ×  Var(Z) / (Var(Z) + Var(error))
```

That fraction is called the **reliability**. **Its impact:** the noisier the tool, the more the apparent effect shrinks. This is why a better tool *should*, in principle, show a larger effect — the original motivation for connecting the two questions.

### Why it fails here — averaging

The market test does **not** use individual headlines. It uses the **average mood of about 337 headlines per day**.

Averaging destroys random error. If individual measurements each have error variance `Var(error)`, the average of `n` of them has:

```
error variance of the average  =  Var(error) / n
```

With `n = 337`, that is **337 times smaller**.

**The impact, made concrete.** Take two tools where one is **twice as noisy per headline** — a large, real difference. Work through the reliability formula with a daily signal variance of 0.01:

| | Per headline | After averaging 337 |
|---|---:|---:|
| Better tool's reliability | 0.038 | 0.931 |
| Worse tool's reliability | 0.020 | 0.871 |
| **Ratio between them** | **1.96×** | **1.07×** |

And because the analysis standardises the tone series, the shrinkage enters as a **square root**, which compresses it further:

```
√0.931 / √0.871  =  1.034
```

> **A twofold difference in per-headline quality becomes a 3.4% difference in the daily result.**

### What this means

**The study could never have detected the quality gap at the market level.** Not because the gap isn't real — Finding 1 shows it is — but because averaging 337 headlines a day removes most of the difference before the market test ever sees it.

So the link between the two questions is **not disproven. It is untestable at this level of aggregation.** That is a stronger and more useful statement than a bare null result, and we could only make it because the mathematics was derived *before* the numbers were computed.

Three further conditions would have to hold for the link to work at all, and none is established here:

- **Averaging only helps if the errors are independent.** If a tool systematically misreads one kind of headline, those mistakes cluster on the days such headlines appear. At a modest 10% correlation between errors, the effective divisor collapses from 337 to about **10**.
- **The scores are capped at −1 and +1.** Near a cap, the error cannot point outward, which breaks the classical assumption. Simulated, this **amplified** the coefficient rather than shrinking it — the opposite of the expected direction.
- **If news genuinely has no relation to next-day returns (`θ = 0`), no amount of measurement quality creates one.** Reliability multiplies `θ`. Multiply zero by anything and it stays zero.

---

## What the study can and cannot say

**Can say:**

- FinBERT classifies these headlines better than the Loughran–McDonald word list, by 0.102 macro-F1 [0.050, 0.153], and the difference survives on the subset the labeller found unambiguous.
- The difference is concentrated almost entirely in good news, and is explained by the word list's 7-to-1 negative-to-positive vocabulary and its three-valued behaviour on short text.
- Daily aggregate news tone shows no detectable association with the next day's SPY return, and any effect is probably smaller than 5 basis points — with that second half reported as fragile.
- The design compresses a twofold measurement-quality difference into roughly 3% of the market coefficient, so it cannot test whether better reading produces a better signal.

**Cannot say:**

- That FinBERT "understands" financial language better in general. It was tested on headlines, against an answer key phrased in the same terms it was trained on.
- That news sentiment is useless for trading. This tested one aggregation, one market-wide instrument, one horizon.
- That the word list is a poor tool. It was applied far outside the document length it was designed for.
- Anything about individual stocks. Attempting that on the three most-covered companies left 44–56% of days with no headlines at all, and every interval spanned zero.

---

## Where the numbers live

| Claim | File |
|---|---|
| Act 1 metrics | `report/tables/table1_act1_metrics.csv` |
| Act 1 comparisons | `report/tables/table1b_act1_contrasts.csv` |
| Thresholds and their determinacy | `report/tables/calibration_record.json` |
| Act 2 primary + secondary | `report/tables/table3_lag_family.csv`, `table4_effect_sizes.csv` |
| Precision recorded before the result was seen | `report/tables/advance_precision.json` |
| Robustness | `report/tables/table6*.csv` |
| Inputs, settings and code state | `report/tables/run_manifest.json` |

The full write-up is [`report/report.md`](report/report.md). The derivations and simulation are in [`docs/archive/mathematical-appendix.md`](docs/archive/mathematical-appendix.md), reproducible with `python attenuation_study.py`.
