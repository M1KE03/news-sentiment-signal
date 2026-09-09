"""R01d: one checkpoint generation, process interruption and writer exclusion."""

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src import scoring


class Stub:
    name = "lm"

    def __init__(self, version="v1", value=.5):
        self.fingerprint = {"scorer": "lm", "version": version}
        self.value = value
        self.calls = 0

    def score(self, texts):
        self.calls += len(texts)
        return np.full(len(texts), self.value, dtype="float32")


def heads(n=4):
    return pd.DataFrame({"headline_id": [f"h{i}" for i in range(n)],
                         "text": [f"story {i}" for i in range(n)]})


def seed(path):
    scoring.score_all(heads(), path, [Stub()], verbose=False)
    return path.read_bytes(), scoring.load_cache(path)[1]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["write", "flush", "validate", "replace"])
def test_failure_before_commit_preserves_previous_generation(tmp_path, monkeypatch, existing, stage):
    path = tmp_path / "scores.parquet"
    before = seed(path) if existing else None

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    target, attr = {"write": (scoring.pq, "write_table"),
                    "flush": (scoring.os, "fsync"),
                    "validate": (scoring, "_validate_candidate"),
                    "replace": (scoring.os, "replace")}[stage]
    monkeypatch.setattr(target, attr, fail)
    with pytest.raises(OSError, match="injected failure"):
        scoring.score_all(heads(), path, [Stub("v2", -.5)], verbose=False,
                          on_fingerprint_change="rescore")
    if before:
        assert path.read_bytes() == before[0]
        assert scoring.load_cache(path)[1] == before[1]
    else:
        assert not path.exists()
        assert scoring.load_cache(path)[0].empty
    assert not list(tmp_path.glob("*.tmp"))
    with scoring._writer_lock(path):  # exception released ownership
        pass


def test_failed_cleanup_does_not_hide_write_error(tmp_path, monkeypatch):
    path = tmp_path / "scores.parquet"
    before, _ = seed(path)

    def fail_write(*args, **kwargs):
        raise OSError("original write failure")

    def fail_cleanup(*args, **kwargs):
        raise PermissionError("cleanup failed")

    monkeypatch.setattr(scoring.pq, "write_table", fail_write)
    monkeypatch.setattr(scoring.os, "unlink", fail_cleanup)
    with pytest.raises(OSError, match="original write failure"):
        scoring.score_all(heads(), path, [Stub()], verbose=False)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation, message", [
    ("json", "JSON"), ("object", "object"), ("version", "version"),
    ("bool_version", "version"), ("generation", "generation_id"),
    ("rows", "n_rows"), ("negative_rows", "n_rows"),
    ("bool_rows", "n_rows"), ("missing", "legacy"),
])
def test_invalid_metadata_refuses_without_sidecar_fallback(tmp_path, mutation, message):
    path = tmp_path / "scores.parquet"
    _, original = seed(path)
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata)
    altered = dict(original)
    if mutation == "json":
        payload = b"{invalid"
    elif mutation == "object":
        payload = b"[]"
    else:
        changes = {"version": ("format_version", 99),
                   "bool_version": ("format_version", True),
                   "generation": ("generation_id", "no"),
                   "rows": ("n_rows", 999), "negative_rows": ("n_rows", -1),
                   "bool_rows": ("n_rows", True)}
        if mutation in changes:
            k, v = changes[mutation]
            altered[k] = v
        payload = json.dumps(altered).encode()
    metadata[scoring.CACHE_METADATA_KEY] = payload
    if mutation == "missing":
        del metadata[scoring.CACHE_METADATA_KEY]
    pq.write_table(table.replace_schema_metadata(metadata), path)
    scoring.meta_path(path).write_text(json.dumps(original))
    with pytest.raises(scoring.IncompatibleCache, match=message):
        scoring.load_cache(path)


def test_valid_checkpoint_ignores_malformed_legacy_sidecar(tmp_path):
    path = tmp_path / "scores.parquet"
    _, expected = seed(path)
    scoring.meta_path(path).write_text("{broken")
    assert scoring.load_cache(path)[1] == expected
    assert scoring.meta_path(path).read_text() == "{broken"
    assert pd.read_parquet(path).score_lm.tolist() == [.5] * 4


@pytest.mark.parametrize("parquet", [False, True])
def test_legacy_pair_is_never_migrated(tmp_path, parquet):
    path = tmp_path / "scores.parquet"
    if parquet:
        heads().to_parquet(path)
    scoring.meta_path(path).write_text("{broken")
    with pytest.raises(scoring.IncompatibleCache, match="legacy"):
        scoring.load_cache(path)


def test_corrupt_file_refuses_with_original_cause(tmp_path):
    path = tmp_path / "scores.parquet"
    path.write_bytes(b"not parquet")
    with pytest.raises(scoring.IncompatibleCache, match="unreadable") as exc:
        scoring.load_cache(path)
    assert isinstance(exc.value.__cause__, pa.ArrowInvalid)


def test_read_permissions_remain_io_errors(tmp_path, monkeypatch):
    path = tmp_path / "scores.parquet"

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError, match="denied"):
        scoring.load_cache(path)


def test_reader_uses_one_open_generation(tmp_path, monkeypatch):
    path = tmp_path / "scores.parquet"
    _, old_meta = seed(path)
    replacement = tmp_path / "replacement.parquet"
    scoring.score_all(heads(), replacement, [Stub("v2", -.5)], verbose=False)
    real_read = scoring.pq.read_table
    real_open = Path.open
    opens = []

    def track_open(self, *args, **kwargs):
        if self == path:
            opens.append(self)
        return real_open(self, *args, **kwargs)

    def replace_during_read(stream, **kwargs):
        assert hasattr(stream, "read")
        try:
            os.replace(replacement, path)
        except PermissionError:  # Windows may refuse while an old reader is open
            pass
        return real_read(stream, **kwargs)

    monkeypatch.setattr(Path, "open", track_open)
    monkeypatch.setattr(scoring.pq, "read_table", replace_during_read)
    frame, meta = scoring.load_cache(path)
    assert opens == [path]
    assert meta == old_meta
    assert frame.score_lm.tolist() == [.5] * 4


CHILD = r'''
import sys, os
import numpy as np
import pandas as pd
from src import scoring
path, mode = sys.argv[1:]
def wait():
    print('READY', flush=True)
    sys.stdin.read(1)
if mode == 'hold':
    with scoring._writer_lock(path): wait()
elif mode == 'contend':
    scoring.load_cache = lambda path: (_ for _ in ()).throw(AssertionError('loaded before lock'))
    try: scoring.score_all(pd.DataFrame(), path, scorers=[])
    except scoring.CacheLockedError: print('LOCKED', flush=True)
    else: raise AssertionError('writer admitted')
elif mode == 'acquire':
    with scoring._writer_lock(path): print('ACQUIRED', flush=True)
else:
    class Stub:
        name = 'lm'
        fingerprint = {'scorer':'lm', 'version': 'v1' if mode == 'after_text' else 'v2'}
        def score(self, texts): return np.full(len(texts), -.5, dtype='float32')
    original = scoring.os.replace
    def replace(src, dst):
        if mode == 'before': wait()
        original(src, dst)
        if mode.startswith('after'): wait()
    scoring.os.replace = replace
    heads = pd.DataFrame({'headline_id':['h0','h1'], 'text':['story 0','story 1']})
    if mode == 'after_text': heads.loc[0,'text'] = 'changed text'
    scoring.score_all(heads, path, [Stub()], verbose=False, checkpoint_every=1,
                      on_fingerprint_change='rescore')
'''


def child_command(path, mode):
    return [sys.executable, "-u", "-c", CHILD, str(path), mode]


@contextmanager
def paused_child(path, mode):
    root = Path(__file__).resolve().parents[1]
    with subprocess.Popen(child_command(path, mode), cwd=root, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
        messages = queue.Queue()
        threading.Thread(target=lambda: messages.put(child.stdout.readline()), daemon=True).start()
        try:
            assert messages.get(timeout=20).strip() == "READY", child.stderr.read()
            yield child
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)


def test_writer_contention_and_release_after_process_death(tmp_path):
    path = tmp_path / "scores.parquet"
    with paused_child(path, "hold") as holder:
        denied = subprocess.run(child_command(path, "contend"), capture_output=True,
                                text=True, timeout=20)
        assert denied.returncode == 0, denied.stderr
        assert denied.stdout.strip() == "LOCKED"
        assert not path.exists()
        holder.kill()
        holder.wait(timeout=10)
    assert path.with_suffix(".parquet.lock").exists()
    acquired = subprocess.run(child_command(path, "acquire"), capture_output=True,
                              text=True, timeout=20)
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "ACQUIRED"


@pytest.mark.parametrize("mode", ["before", "after", "after_text"])
def test_hard_stop_around_commit_and_resumption(tmp_path, mode):
    path = tmp_path / "scores.parquet"
    old_bytes, old_meta = seed(path)
    with paused_child(path, mode):
        pass  # kill without running the writer's finally blocks
    saved, meta = scoring.load_cache(path)
    if mode == "before":
        assert path.read_bytes() == old_bytes
        assert meta == old_meta
        assert list(tmp_path.glob("*.tmp"))  # valid orphan must not be promoted
    elif mode == "after":
        assert meta["generation_id"] != old_meta["generation_id"]
        assert saved.score_lm.notna().sum() == 1
    else:
        assert meta["generation_id"] != old_meta["generation_id"]
        assert saved.score_lm.tolist() == [-.5, .5, .5, .5]
    resumed_heads = heads()
    if mode == "after_text":
        resumed_heads.loc[0, "text"] = "changed text"
    scorer = Stub("v1" if mode == "after_text" else "v2", -.5)
    result = scoring.score_all(resumed_heads, path, [scorer], verbose=False,
                               on_fingerprint_change="rescore")
    assert scorer.calls == {"before": 4, "after": 3, "after_text": 0}[mode]
    expected = [-.5, .5, .5, .5] if mode == "after_text" else [-.5] * 4
    assert result.score_lm.tolist() == expected


def test_runner_refuses_invalid_checkpoint_before_aggregation(tmp_path, monkeypatch):
    import run_all

    path = tmp_path / "scores.parquet"
    path.write_bytes(b"not parquet")
    headline_path = tmp_path / "headlines.parquet"
    heads().to_parquet(headline_path)
    monkeypatch.setattr(config, "SCORES_PARQUET", path)
    monkeypatch.setattr(config, "HEADLINES_PARQUET", headline_path)
    monkeypatch.setattr(run_all.align, "aggregate_daily",
                        lambda *args: pytest.fail("aggregation must not be reached"))
    with pytest.raises(scoring.IncompatibleCache):
        run_all.build_panel()


def test_validated_read_loads_no_models(tmp_path, monkeypatch):
    path = tmp_path / "scores.parquet"
    seed(path)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(scoring, "_configured_fingerprints",
                        lambda: pytest.fail("read must not inspect model/lexicon artifacts"))
    assert len(scoring.load_cache(path)[0]) == 4
