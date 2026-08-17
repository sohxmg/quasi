"""Feynman-PRM: step-level quasimetric process reward model."""

# A native abort now prints the Python frame it happened in. The 2026-08-16 crash below gave
# `munmap_chunk(): invalid pointer` and NOTHING else -- no traceback, no line number -- so the
# only way to locate it was to reason backwards from which log line had not printed yet. This
# is ~0 cost, it is on for every entry point, and it turns the next such abort into a stack.
import faulthandler

faulthandler.enable()

# ---------------------------------------------------------------------------------------
# THE pyarrow-BEFORE-torch IMPORT THAT USED TO SIT HERE IS DELETED. IT DID NOT WORK.
# 2026-08-16, the same day it was added.
#
# It ordered the two `dlopen`s on the theory that whichever loads first wins symbol
# resolution for both. The next run had `pyarrow.lib` ahead of `torch._C` in the crash
# dump's extension-module list -- i.e. the import ran -- and `munmap_chunk(): invalid
# pointer` fired anyway. Its own escape clause said what to do in that case, and this is it:
# **delete the block rather than leave a guard that does not guard** (§14, B11/B12).
#
# It also had the location wrong, and the faulthandler above is what showed it. There is no
# frame BELOW `main` in the dump -- no pandas, no pyarrow, no `importlib` -- so the read had
# already returned and the abort landed on the free() of the DataFrame and the pyarrow
# buffers behind it, as the statement's temporaries were released. The read works; the
# teardown corrupts the heap.
#
# The fix is `feynman_prm/data/sequence_cache.py`: the parquet is converted ONCE in a child
# interpreter that has no torch in it, and every reader loads flat numpy arrays. Read that
# module's docstring before touching anything in this area. `FEYNMAN_SEQUENCE_CACHE=0`
# restores the in-process read, which is how to check whether this is still needed.
# ---------------------------------------------------------------------------------------

__version__ = "0.1.0"
