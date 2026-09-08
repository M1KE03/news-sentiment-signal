# `data/raw/`

Nothing in here is tracked by git. Three things belong here, all fetched by
`download.py` or by hand:

| file | source | note |
|---|---|---|
| the news dump | D1, locked at Stage 0 — FNSPID (HuggingFace) or the Kaggle Benzinga headline set | large; redistribution-restricted, hence untracked |
| `LoughranMcDonald_MasterDictionary.csv` | the Loughran–McDonald SRAF site | pin the release in `config.LM_DICT_VERSION` |
| SPY / ^VIX cache | `yfinance`, via `src.data.load_market` | rebuilt on demand |

The Financial PhraseBank (Malo et al. 2014) is pulled straight from HuggingFace
by `src.validate.load_phrasebank`; record its citation and licence in the report.
