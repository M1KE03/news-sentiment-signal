"""R02: the acquisition path cannot destroy or misattribute the analysis input.

Audit A05: `assemble_corpus` defaulted to `HEADLINES_PARQUET`, downloaded
`/resolve/main/` rather than the pinned revision, and enforced neither the
recorded byte length nor the SHA-256. A single documented command could
therefore replace 869k deduplicated rows with 1.41M undeduplicated ones, under
the same filename, from an artifact nobody had checked.

Every test here is offline: the HTTP layer is replaced with an in-memory CSV.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

# download.py lives under data/raw/ and is not an importable package.
_spec = importlib.util.spec_from_file_location("dl", ROOT / "data" / "raw" / "download.py")
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


CSV = (
    "Date,Article_title,Stock_symbol,Url,Publisher,Author,Article,"
    "Lsa_summary,Luhn_summary,Textrank_summary,Lexrank_summary\n"
    "2015-03-02 00:00:00 UTC,Apple beats on earnings,AAPL,https://www.benzinga.com/1,B,,,,,,\n"
    "2015-03-02 00:00:00 UTC,Apple beats on earnings,MSFT,https://www.benzinga.com/2,B,,,,,,\n"
    "2015-03-03 00:00:00 UTC,A different headline entirely,GE,https://www.benzinga.com/3,B,,,,,,\n"
    "2015-03-04 00:00:00 UTC,Zacks upgrade for XYZ,XYZ,https://www.zacks.com/4,Z,,,,,,\n"
).encode("utf-8")

CSV_SHA = hashlib.sha256(CSV).hexdigest()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.raw = _Raw(payload)
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Raw:
    def __init__(self, payload: bytes):
        self._buf = payload
        self._pos = 0
        self.decode_content = False

    def read(self, n=-1):
        if n is None or n < 0:
            n = len(self._buf) - self._pos
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


@pytest.fixture
def offline(monkeypatch):
    """Serve CSV over the requests API, and pin config to match it."""
    monkeypatch.setattr(dl.requests, "get", lambda url, **kw: _FakeResponse(CSV))
    monkeypatch.setattr(dl, "remote_size", lambda url=None: len(CSV))
    monkeypatch.setattr(config, "NEWS_FILE_BYTES", len(CSV))
    monkeypatch.setattr(config, "NEWS_FILE_SHA256", CSV_SHA)
    monkeypatch.setattr(config, "SAMPLE_START", "2015-01-01")
    monkeypatch.setattr(config, "SAMPLE_END", "2015-12-31")
    return CSV


# ---------------------------------------------------------------- A05 guard


def test_assembly_refuses_to_write_the_analysis_input(offline, tmp_path, monkeypatch):
    """The headline defect: --assemble must not be able to clobber the clean corpus."""
    clean = tmp_path / "headlines.parquet"
    monkeypatch.setattr(config, "HEADLINES_PARQUET", clean)
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", tmp_path / "headlines_raw.parquet")

    pd.DataFrame({"headline_id": ["keep"], "text": ["precious"]}).to_parquet(clean, index=False)

    with pytest.raises(dl.AcquisitionError, match="deduplicated analysis input"):
        dl.assemble_corpus(out=clean)

    # Untouched.
    assert pd.read_parquet(clean)["headline_id"].tolist() == ["keep"]


def test_assembly_defaults_to_the_raw_destination(offline, tmp_path, monkeypatch):
    raw = tmp_path / "headlines_raw.parquet"
    monkeypatch.setattr(config, "HEADLINES_PARQUET", tmp_path / "headlines.parquet")
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)

    out = dl.assemble_corpus()
    assert out == raw
    assert not (tmp_path / "headlines.parquet").exists()


# ------------------------------------------------------- pinned identity


def test_url_uses_the_pinned_revision_not_a_branch():
    assert config.NEWS_HF_REVISION in dl.FNSPID_URL
    assert "/resolve/main/" not in dl.FNSPID_URL


def test_wrong_digest_refuses_to_publish(offline, tmp_path, monkeypatch):
    raw = tmp_path / "headlines_raw.parquet"
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)
    monkeypatch.setattr(config, "NEWS_FILE_SHA256", "0" * 64)

    with pytest.raises(dl.AcquisitionError, match="sha256"):
        dl.assemble_corpus()
    assert not raw.exists(), "nothing may be written when the artifact is unverified"
    assert not list(tmp_path.glob("*.tmp"))


def test_wrong_length_refuses_to_publish(offline, tmp_path, monkeypatch):
    raw = tmp_path / "headlines_raw.parquet"
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)
    monkeypatch.setattr(config, "NEWS_FILE_BYTES", len(CSV) + 1)

    with pytest.raises(dl.AcquisitionError, match="length"):
        dl.assemble_corpus()
    assert not raw.exists()


def test_manifest_records_the_verified_identity(offline, tmp_path, monkeypatch):
    raw = tmp_path / "headlines_raw.parquet"
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)

    dl.assemble_corpus()
    man = json.loads(dl.manifest_path(raw).read_text(encoding="utf-8"))
    assert man["sha256"] == CSV_SHA
    assert man["bytes_read"] == len(CSV)
    assert man["verified_against_pin"] is True
    assert man["revision"] == config.NEWS_HF_REVISION
    assert man["deduplicated"] is False
    assert man["rows_wrong_source"] == 1          # the zacks.com row


# ------------------------------------------------- assemble -> dedup -> census


def test_dedup_step_produces_the_analysis_input(offline, tmp_path, monkeypatch):
    raw = tmp_path / "headlines_raw.parquet"
    clean = tmp_path / "headlines.parquet"
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)
    monkeypatch.setattr(config, "HEADLINES_PARQUET", clean)

    dl.assemble_corpus()
    assert len(pd.read_parquet(raw)) == 3         # benzinga rows, incl. the repeat

    dl.deduplicate_corpus()
    out = pd.read_parquet(clean)
    # The same headline emitted under two tickers collapses to one.
    assert len(out) == 2
    man = json.loads(dl.manifest_path(clean).read_text(encoding="utf-8"))
    assert man["deduplicated"] is True
    assert man["rows_in"] == 3 and man["rows_out"] == 2
    assert man["source_manifest"]["sha256"] == CSV_SHA

    # R03b: the collapsed row keeps BOTH tickers it was filed under, instead of
    # discarding the loser's tag with its row.
    collapsed = out[out["n_cluster_rows"] == 2]
    assert len(collapsed) == 1
    assert sorted(collapsed["tickers"].iloc[0]) == ["AAPL", "MSFT"]

    # R03b: lineage lands BESIDE the corpus it describes, and accounts for every
    # raw row. Writing it to `config.DEDUP_LINEAGE_PARQUET` instead meant this
    # very test deposited a stray artifact in the real `data/interim`, next to
    # the frozen corpus, while publishing into tmp_path.
    lin_path = dl.lineage_path(clean)
    assert lin_path.parent == clean.parent
    assert not config.DEDUP_LINEAGE_PARQUET.exists(), (
        "lineage was written to the configured path instead of beside `out`"
    )
    lineage = pd.read_parquet(lin_path)
    assert len(lineage) == 3
    assert int(lineage["kept"].sum()) == 2
    assert man["lineage"] == lin_path.name


def test_dedup_refuses_an_unverified_raw_corpus(tmp_path, monkeypatch):
    """A clean artifact may not be derived from provenance nobody recorded."""
    raw = tmp_path / "headlines_raw.parquet"
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)
    monkeypatch.setattr(config, "HEADLINES_PARQUET", tmp_path / "headlines.parquet")
    pd.DataFrame({"headline_id": ["a"], "text": ["x"], "text_norm": ["x"],
                  "ts_utc": [pd.Timestamp("2015-01-01", tz="UTC")]}).to_parquet(raw, index=False)

    with pytest.raises(dl.AcquisitionError, match="no acquisition manifest"):
        dl.deduplicate_corpus()


def test_dedup_refuses_a_manifest_that_records_no_verification(tmp_path, monkeypatch):
    raw = tmp_path / "headlines_raw.parquet"
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", raw)
    monkeypatch.setattr(config, "HEADLINES_PARQUET", tmp_path / "headlines.parquet")
    pd.DataFrame({"headline_id": ["a"], "text": ["x"], "text_norm": ["x"],
                  "ts_utc": [pd.Timestamp("2015-01-01", tz="UTC")]}).to_parquet(raw, index=False)
    dl.manifest_path(raw).write_text(json.dumps({"verified_against_pin": False}), encoding="utf-8")

    with pytest.raises(dl.AcquisitionError, match="does not record verification"):
        dl.deduplicate_corpus()


def test_missing_raw_corpus_names_the_command_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HEADLINES_RAW_PARQUET", tmp_path / "absent.parquet")
    with pytest.raises(dl.AcquisitionError, match="run --assemble"):
        dl.deduplicate_corpus()
