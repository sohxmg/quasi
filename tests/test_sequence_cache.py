"""§14 B15: the parquet is read in a CHILD process and training loads numpy arrays.

These tests write a real parquet and run the real subprocess, because the whole point of
the module is what happens across a process boundary -- a fixture that stubbed the child
would pass while the shipped path was broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from feynman_prm.data.math_shepherd import read_sequences_parquet, write_sequences_parquet
from feynman_prm.data.sequence_cache import (
    ENV_DISABLE,
    build_cache,
    cache_is_fresh,
    cache_path_for,
    load_sequence_columns,
)

pd = pytest.importorskip("pandas")


def make_rows(n: int = 5, with_prefix_hash: bool = True) -> list[dict]:
    rows = []
    for i in range(n):
        n_steps = 1 + i % 3
        row = {
            "qid": f"q{i % 2}" + "a" * 36,
            "split": "train" if i % 2 == 0 else "val",
            "correct": bool(i % 2),
            "z": -1 if i % 2 else i % 4,
            "recovery": bool(i % 3 == 0),
            "n_steps": n_steps,
            "length": 3 + 3 * n_steps,
            "input_ids": np.arange(3 + 3 * n_steps, dtype=np.int32),
            "state_pos": np.arange(n_steps + 1, dtype=np.int32) * 3,
            "span_start": np.arange(n_steps, dtype=np.int32),
            "span_end": np.arange(n_steps, dtype=np.int32) + 2,
        }
        if with_prefix_hash:
            row["prefix_hash"] = np.asarray(
                [(-1) ** j * (10**18 + i * 7 + j) for j in range(n_steps + 1)], dtype=np.int64
            )
        rows.append(row)
    return rows


def write(tmp_path, rows) -> "object":
    path = tmp_path / "sequences.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def same_rows(a, b) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.qid == y.qid and x.correct == y.correct and x.z == y.z
        assert x.recovery == y.recovery
        for field in ("input_ids", "state_pos", "span_start", "span_end"):
            assert np.array_equal(getattr(x, field), getattr(y, field))
            assert getattr(x, field).dtype == np.int64
        if x.prefix_hash is None or y.prefix_hash is None:
            assert x.prefix_hash is None and y.prefix_hash is None
        else:
            assert np.array_equal(x.prefix_hash, y.prefix_hash)
            assert x.prefix_hash.dtype == np.int64


def test_cached_read_matches_the_in_process_read(tmp_path, monkeypatch):
    """The contract. Whatever the cache does, it must hand back the rows the pandas read
    handed back before 2026-08-16 -- same order, same values, same int64 dtypes."""
    path = write(tmp_path, make_rows())

    monkeypatch.setenv(ENV_DISABLE, "0")
    direct = read_sequences_parquet(path)
    monkeypatch.delenv(ENV_DISABLE)
    cached = read_sequences_parquet(path)   # builds the cache in a subprocess

    same_rows(direct, cached)
    assert cache_path_for(path).exists()


def test_the_split_filter_survives_the_cache(tmp_path):
    path = write(tmp_path, make_rows(n=6))
    train = read_sequences_parquet(path, split="train")
    val = read_sequences_parquet(path, split="val")
    assert len(train) == 3 and len(val) == 3
    assert len(read_sequences_parquet(path)) == 6


def test_a_parquet_with_no_prefix_hash_gives_None_not_a_crash(tmp_path):
    """§7.5.13: a pre-2026-08-15 parquet must still train, with CF attaching nothing. The
    cache has to carry the column's ABSENCE, not fill it with zeros -- `cf_attach` never
    matches on 0, but a zeros row would look like a row that carries hashes."""
    path = write(tmp_path, make_rows(with_prefix_hash=False))
    rows = read_sequences_parquet(path)
    assert all(r.prefix_hash is None for r in rows)


def test_a_stale_cache_is_rebuilt_and_never_silently_read(tmp_path):
    """§8.2's tell: re-running prepare_data.py must not leave the previous selection on
    disk in a form the next launch reads as current."""
    path = write(tmp_path, make_rows(n=4))
    assert len(read_sequences_parquet(path)) == 4
    cache = cache_path_for(path)
    assert cache_is_fresh(cache, path)

    write(tmp_path, make_rows(n=6))          # same path, different contents
    assert not cache_is_fresh(cache, path)
    assert len(read_sequences_parquet(path)) == 6


def test_a_truncated_cache_is_rebuilt(tmp_path):
    path = write(tmp_path, make_rows())
    read_sequences_parquet(path)
    cache = cache_path_for(path)
    cache.write_bytes(cache.read_bytes()[:64])
    assert not cache_is_fresh(cache, path)
    assert len(read_sequences_parquet(path)) == 5


def test_write_sequences_parquet_leaves_a_fresh_cache(tmp_path):
    """prepare_data.py writes both, so the first launch after it does no conversion at all."""
    rows = make_rows()
    path = tmp_path / "sequences.parquet"
    write_sequences_parquet(rows, path)
    assert cache_is_fresh(cache_path_for(path), path)
    same_rows(read_sequences_parquet(path), read_sequences_parquet(path))
    assert len(read_sequences_parquet(path, split="train")) == 3


def test_the_child_never_imports_torch(tmp_path):
    """The whole fix is that the process holding the pyarrow buffers has no torch in it.
    Assert it on the module the child actually runs, not on the intention."""
    import subprocess
    import sys

    path = write(tmp_path, make_rows())
    probe = (
        "import sys, feynman_prm.data.sequence_cache as m;"
        f"m.build_cache({str(path)!r});"
        "print('torch' in sys.modules, 'wandb' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=str(cache_path_for(path).parents[0]),
        env={**__import__("os").environ,
             "PYTHONPATH": str(__import__("pathlib").Path(__file__).resolve().parents[1])},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"


def test_build_cache_is_atomic_no_tmp_left_behind(tmp_path):
    path = write(tmp_path, make_rows())
    build_cache(path)
    assert not [p for p in tmp_path.iterdir() if ".tmp" in p.name]


def test_question_ids_reads_either_split_without_building_rows(tmp_path):
    """train.py's §8.2 leakage check needs the VAL qids, and paying a full row build for two
    string columns would put ~20 s of `SequenceRow` construction on every launch."""
    from feynman_prm.data.sequence_cache import question_ids

    path = write(tmp_path, make_rows(n=6))
    train = question_ids(path, split="train")
    val = question_ids(path, split="val")
    assert train and val and not (train & val)
    assert question_ids(path) == train | val
    assert train == {r.qid for r in read_sequences_parquet(path, split="train")}


def test_a_missing_required_column_is_an_error_not_a_default(tmp_path):
    """A short `correct` array would cache happily and blow up later in indexing code, with
    nothing pointing back at the parquet."""
    from feynman_prm.data.sequence_cache import columns_from_lists

    rows = make_rows()
    with pytest.raises(KeyError, match="correct"):
        columns_from_lists({k: [r[k] for r in rows] for k in rows[0] if k != "correct"})


def test_asking_for_a_split_a_file_does_not_carry_raises(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "split"} for r in make_rows()]
    path = write(tmp_path, rows)
    assert len(read_sequences_parquet(path)) == 5
    with pytest.raises(KeyError, match="split"):
        read_sequences_parquet(path, split="train")


def test_load_sequence_columns_says_which_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        load_sequence_columns(tmp_path / "nope.parquet")
