"""THE parquet read, moved OUT of any process that has torch in it. 2026-08-16.

`bash scripts/train.sh` aborted on kratos with

    munmap_chunk(): invalid pointer
    Fatal Python error: Aborted
    Current thread ...:
      File "feynman_prm/train.py", line 175 in main          <- rows = read_sequences_parquet(...)
      File "feynman_prm/train.py", line 482 in <module>

glibc raises that when `free()` is handed a pointer it did not allocate: native heap
corruption, nothing to do with any Python object.

**Read the traceback carefully, because it says more than "line 175".** There is NO frame
below `main`. A crash *inside* `pd.read_parquet` would have shown `math_shepherd.py` and a
stack of pandas/pyarrow frames; a crash inside an import would have shown
`<frozen importlib._bootstrap>`. A bare `main` frame at the assignment statement means the
call had already RETURNED and the abort landed while the statement's temporaries -- the
DataFrame and the pyarrow buffers behind it -- were being freed. The read works; the
teardown is what corrupts the heap.

**The import-order fix that preceded this one did not work, and is deleted.**
`feynman_prm/__init__.py` imported pyarrow before torch on the theory that whichever
`dlopen`s first wins symbol resolution for both. The extension-module list in the crash
above shows `pyarrow.lib` loading before `torch._C`, i.e. that import ran and the abort
happened anyway. A guard that does not guard is worse than no guard (§14, B11/B12), so it
went rather than being left in place looking like a fix.

**What this module does instead: the training process never links the two libraries in the
same address space at all.** `sequences.parquet` is converted ONCE, in a fresh interpreter
that has pandas/pyarrow and no torch and no wandb, into a plain `.npz` of flat numpy arrays;
every later launch loads that with numpy alone. The conversion process is exactly the "bare
`python -c`" that was measured to read the same parquet without crashing.

Three properties worth stating, because they are the reasons this is a fix and not a
workaround:

  * **It is agnostic about which pair of libraries is at fault.** pyarrow/torch,
    pyarrow/numpy ABI, two libstdc++ copies from mixing conda-forge and pip wheels -- the
    training process holds none of them.
  * **It fails LOUDLY.** If the child aborts too, the parent raises with the child's stderr
    and its signal, which is a diagnosis (pandas/pyarrow is broken in this env, independent
    of torch -- reinstall pyarrow from ONE channel) rather than a bare SIGABRT.
  * **It is cached and keyed on the parquet's own size and mtime**, so re-running
    `prepare_data.py` -- or rsyncing a new parquet -- rebuilds it, and a stale cache cannot
    silently train the previous selection (§8.2's tell).

`FEYNMAN_SEQUENCE_CACHE=0` restores the old in-process read for every caller, which is the
one-line way to check whether this module is what is keeping a run alive.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Bump when the on-disk layout changes; a cache written by an older version is rebuilt.
CACHE_VERSION = 1

ENV_DISABLE = "FEYNMAN_SEQUENCE_CACHE"

# Per-row variable-length columns, stored flattened as `<name>__values` + `<name>__offsets`.
LIST_COLUMNS = ("input_ids", "state_pos", "span_start", "span_end")
# Absent from a `sequences.parquet` written before 2026-08-15 (§7.5.13). Its absence is
# carried through the cache so `read_sequences_parquet` still hands `prefix_hash=None` to
# `SequenceRow` and `cf_attach` still degrades to "nothing attaches" instead of crashing.
OPTIONAL_LIST_COLUMNS = ("prefix_hash",)

_STR_COLUMNS = ("qid", "split")
_BOOL_COLUMNS = ("correct", "recovery")
_INT_COLUMNS = ("z",)


def cache_path_for(parquet: str | Path) -> Path:
    """`.../sequences.parquet` -> `.../sequences.cache.npz`."""
    parquet = Path(parquet)
    return parquet.with_suffix(".cache.npz")


# ---------------------------------------------------------------------------------------
# building the column dict (this half runs in the CHILD, or under FEYNMAN_SEQUENCE_CACHE=0)
# ---------------------------------------------------------------------------------------


def _flatten(sequences, dtype) -> tuple[np.ndarray, np.ndarray]:
    """Ragged list of 1-D arrays -> (concatenated values, (n+1,) offsets)."""
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if len(sequences):
        values = np.concatenate([np.asarray(s, dtype=dtype) for s in sequences])
    else:
        values = np.zeros(0, dtype=dtype)
    return values, offsets


def columns_from_lists(columns: dict[str, list]) -> dict[str, np.ndarray]:
    """The one place the on-disk layout is defined. `columns` is column name -> list.

    Every column except `split` and `prefix_hash` is mandatory and its absence is an error,
    not a default: a row set missing `correct` or `z` would otherwise cache as a shorter
    array and fail much later, in indexing code, with nothing pointing at the parquet.
    """
    n_rows = max((len(v) for v in columns.values()), default=0)
    missing = [
        name for name in ("qid",) + _BOOL_COLUMNS + _INT_COLUMNS + LIST_COLUMNS
        if name not in columns
    ]
    if n_rows and missing:
        raise KeyError(f"sequence rows are missing required column(s): {', '.join(missing)}")

    out: dict[str, np.ndarray] = {}
    # `qid` is always written (it is what `read_sequences_parquet` counts rows with); `split`
    # only when the source carried it, so asking for a split on a file without one raises
    # there rather than quietly matching nothing.
    out["qid"] = np.asarray([str(v) for v in columns.get("qid", [])], dtype=np.str_)
    if "split" in columns:
        out["split"] = np.asarray([str(v) for v in columns["split"]], dtype=np.str_)
    for name in _BOOL_COLUMNS:
        out[name] = np.asarray(columns.get(name, []), dtype=bool)
    for name in _INT_COLUMNS:
        out[name] = np.asarray(columns.get(name, []), dtype=np.int64)
    for name in LIST_COLUMNS:
        # int32 on disk: ids and positions are bounded by the vocab (~151k) and `max_len`.
        # The cast back to int64 happens per row in `read_sequences_parquet`, which is the
        # copy `np.asarray(record.input_ids, dtype=np.int64)` already made.
        values, offsets = _flatten(columns.get(name, []), np.int32)
        out[f"{name}__values"], out[f"{name}__offsets"] = values, offsets
    for name in OPTIONAL_LIST_COLUMNS:
        if name not in columns:
            continue
        # int64 and NOT int32: a prefix hash is a signed 64-bit blake2b digest (prefix_hash.py).
        values, offsets = _flatten(columns[name], np.int64)
        out[f"{name}__values"], out[f"{name}__offsets"] = values, offsets
    return out


def columns_from_rows(rows) -> dict[str, np.ndarray]:
    """From `prepare_data.py`'s row dicts, so writing the cache costs no parquet read-back."""
    rows = list(rows)
    wanted = _STR_COLUMNS + _BOOL_COLUMNS + _INT_COLUMNS + LIST_COLUMNS + OPTIONAL_LIST_COLUMNS
    present = {name for name in wanted if rows and name in rows[0]}
    return columns_from_lists({name: [r[name] for r in rows] for name in present})


def columns_from_parquet(path: str | Path) -> dict[str, np.ndarray]:
    """The in-process read. **pandas and pyarrow are imported HERE and nowhere above**, so
    the parent only ever executes this under `FEYNMAN_SEQUENCE_CACHE=0`."""
    import pandas as pd

    frame = pd.read_parquet(path)
    wanted = _STR_COLUMNS + _BOOL_COLUMNS + _INT_COLUMNS + LIST_COLUMNS + OPTIONAL_LIST_COLUMNS
    columns = {name: frame[name].tolist() for name in wanted if name in frame.columns}
    out = columns_from_lists(columns)
    # `frame` and everything pyarrow owns behind it dies at the end of THIS function. In the
    # child that is the whole point; in the parent (escape hatch) it is the free() that
    # aborted the 2026-08-16 run.
    del frame
    return out


# ---------------------------------------------------------------------------------------
# the cache file
# ---------------------------------------------------------------------------------------


def _source_key(parquet: Path) -> np.ndarray:
    stat = parquet.stat()
    return np.asarray([CACHE_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)


def save_columns(columns: dict[str, np.ndarray], out: str | Path, parquet: str | Path) -> Path:
    """Atomic: write a sibling temp file and `os.replace`, so a killed build never leaves a
    half-written cache that the next launch would read as valid."""
    out, parquet = Path(out), Path(parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(columns)
    payload["meta"] = _source_key(parquet)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            np.savez(handle, **payload)   # a file OBJECT: np.savez appends ".npz" to a name
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out


def load_columns(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:   # allow_pickle=False: it is all arrays
        return {name: handle[name] for name in handle.files}


def cache_is_fresh(cache: Path, parquet: Path) -> bool:
    """Keyed on the parquet's size and mtime, never on "the file exists". Re-running
    `prepare_data.py` changes both (§8.2: a stale artifact silently training the previous
    selection is exactly the failure that hid the 23,000-question run)."""
    if not cache.exists():
        return False
    try:
        with np.load(cache, allow_pickle=False) as handle:
            meta = handle["meta"]
    except Exception:
        return False    # truncated, or written by a version that had no `meta`
    return bool(np.array_equal(np.asarray(meta, dtype=np.int64), _source_key(parquet)))


def build_cache(parquet: str | Path, out: str | Path | None = None) -> Path:
    """Read the parquet and write the cache. **This is the function the child runs.**"""
    parquet = Path(parquet)
    out = Path(out) if out is not None else cache_path_for(parquet)
    return save_columns(columns_from_parquet(parquet), out, parquet)


def _build_in_subprocess(parquet: Path, cache: Path) -> None:
    command = [
        sys.executable, "-m", "feynman_prm.data.sequence_cache",
        "--parquet", str(parquet), "--out", str(cache),
    ]
    # The child must import `feynman_prm` the same way the parent did, whatever the cwd is.
    package_parent = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in (package_parent, env.get("PYTHONPATH", "")) if p]
    )
    env.pop(ENV_DISABLE, None)   # the child IS the in-process read; never recurse into a build
    print(f"[sequence-cache] converting {parquet} -> {cache} (once; a fresh interpreter, so "
          f"pyarrow and torch are never loaded together)", flush=True)
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        for line in proc.stdout.strip().splitlines():
            print(f"[sequence-cache] {line}", flush=True)
        return

    signal = f"signal {-proc.returncode}" if proc.returncode < 0 else f"exit {proc.returncode}"
    tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
    raise RuntimeError(
        f"the parquet -> cache conversion died ({signal}).\n"
        f"{tail}\n\n"
        "That child had NO torch and NO wandb loaded, so if it aborted natively "
        "(munmap_chunk / free(): invalid pointer / Segmentation fault) then pandas+pyarrow "
        "are broken in this environment on their own -- most often numpy/pyarrow built "
        "against different ABIs, or conda-forge and pip wheels of numpy/pyarrow/torch "
        "mixed in one env. Reinstall pyarrow and numpy from ONE channel and re-run "
        f"`python -m feynman_prm.data.sequence_cache --parquet {parquet}`.\n"
        f"To bypass this module entirely (and get the original crash back), set "
        f"{ENV_DISABLE}=0."
    )


def question_ids(parquet: str | Path, split: str | None = None) -> set[str]:
    """The qids ON DISK for a split, without building a single `SequenceRow`.

    `np.load` reads an `.npz`'s directory and decompresses only the members asked for, so
    this touches two small string arrays and never the ~500 MB of token ids. That is what
    makes it cheap enough for `train.py` to ask about the val split at launch (§8.2's
    leakage check needs BOTH sides, not just the one it is training on).
    """
    parquet = Path(parquet)
    if os.environ.get(ENV_DISABLE) == "0":
        columns = columns_from_parquet(parquet)
    else:
        cache = cache_path_for(parquet)
        if not cache_is_fresh(cache, parquet):
            _build_in_subprocess(parquet, cache)
        with np.load(cache, allow_pickle=False) as handle:
            columns = {name: handle[name] for name in ("qid", "split") if name in handle.files}

    qids = columns["qid"]
    if split is None:
        return {str(q) for q in qids}
    if "split" not in columns:
        raise KeyError(f"{parquet} carries no `split` column; cannot select split={split!r}")
    return {str(q) for q in qids[columns["split"] == split]}


def load_sequence_columns(parquet: str | Path, rebuild: bool = False) -> dict[str, np.ndarray]:
    """The public entry: flat columns for `sequences.parquet`, without importing pyarrow."""
    parquet = Path(parquet)
    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} does not exist -- run scripts/prepare_data.py first (§18.1)"
        )
    if os.environ.get(ENV_DISABLE) == "0":
        return columns_from_parquet(parquet)   # the escape hatch, and the pre-2026-08-16 path

    cache = cache_path_for(parquet)
    if rebuild or not cache_is_fresh(cache, parquet):
        _build_in_subprocess(parquet, cache)
    return load_columns(cache)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    out = build_cache(args.parquet, args.out)
    columns = load_columns(out)
    has_hash = "prefix_hash__offsets" in columns
    note = "yes" if has_hash else "NO -- pre-2026-08-15 parquet, CF attaches nothing (§7.5.13)"
    print(f"{len(columns['qid'])} rows -> {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"prefix_hash={note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
