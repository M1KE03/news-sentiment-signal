# Loughran–McDonald dictionary acquisition

Acquired: 2026-09-10. This resolves the missing dictionary artifact for LM scoring; it does not complete scoring, calibration or empirical validation.

The [official Notre Dame SRAF page](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) links the **1993–2025 master dictionary**, updated March 2026, and states that academic research use is free. The page identifies positive category years as inclusion and negative years as removal. The existing scorer selects entries whose Positive/Negative value is greater than zero.

| Field | Verified value |
|---|---|
| Official filename | `Loughran-McDonald_MasterDictionary_1993-2025.csv` |
| Officially linked file | [Google Drive CSV](https://drive.google.com/file/d/1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF/view) |
| Download endpoint | `https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF` |
| Local file | `data/raw/LoughranMcDonald_MasterDictionary.csv` |
| Bytes | 9,093,460 |
| SHA-256 | `e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d` |
| Entries | 86,553 unique, nonempty words |
| Active positive / negative words | 347 / 2,345 |
| Configuration | `LM_DICT_VERSION`, `LM_DICT_SHA256`, `LM_DICT_PATH` |

Verification used the exact downloaded bytes, checked CSV columns and numeric sentiment categories, and instantiated the existing `LMScorer`. Synthetic phrases containing positive words, negative words and no sentiment words returned +1, −1 and 0. No sampled annotation headline was scored.

The dictionary contains the literal word `NULL`; validation therefore reads it with `keep_default_na=False`. Both its Positive and Negative values are zero, so the existing LM scorer's default null parsing does not change either scoring word list for this artifact.

The bytes were copied unchanged into the configured raw-data path. The raw CSV is gitignored; this document and the config retain acquisition evidence. `LM_DICT_SHA256` records the acquired artifact; scorer fingerprints continue to identify the actual loaded positive/negative word lists. The configured byte hash is not a new automatic file-integrity gate.

For reconstruction, download the officially linked CSV and compare its SHA-256 before using it. The Drive object may change; a hash mismatch is a different artifact and requires a recorded decision, not silently updating the pin. No claim is made that this March 2026 dictionary was available during the 2010–2019 study window.
