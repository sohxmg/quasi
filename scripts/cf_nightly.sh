#!/usr/bin/env bash
#
# One NIGHT of counterfactual generation (§7.5.6). Run it again tomorrow; run it again the
# night after. It is resumable, idempotent, and it stops on the clock rather than on a signal.
#
#   bash scripts/cf_nightly.sh                 # 7.5 hours, the defaults below
#   HOURS=8 bash scripts/cf_nightly.sh         # a longer window
#   bash scripts/cf_nightly.sh --dry-run       # print the command and the state, run nothing
#
# WHAT MAKES IT SAFE TO RUN TWICE, OR TO KILL:
#   * every response is appended and FLUSHED to <out>.responses.jsonl the moment it lands;
#   * --resume skips ids that already HOLD a response, and retries the ones that failed
#     (`_already_done` keys on `response is not null`, not on the id -- that distinction is
#     the whole flag);
#   * --stop-after stops SUBMITTING at the deadline and lets in-flight requests land, so no
#     request the endpoint has already worked on is thrown away.
# Therefore: re-running after a crash costs at most the requests that were in flight, and
# re-running after a clean finish costs one validation pass and no API calls at all.
#
# DO NOT wrap this in `timeout(1)`. SIGTERM kills the process between the write and the
# flush; --stop-after is the graceful equivalent and is why it exists.

set -euo pipefail
cd "$(dirname "$0")/.."

# WHICH INTERPRETER. `python3` on this box is AMBIGUOUS: a login shell puts /usr/local/bin
# ahead of /opt/anaconda3/bin, so `python3` is 3.12.5 WITHOUT the deps while `pip` is
# anaconda's WITH them -- `pip install pyyaml` then reports success and the run still dies on
# `No module named yaml`. Resolve it once, here, rather than per-shell:
#   PYTHON=... overrides;
#   otherwise prefer an interpreter that can actually import yaml;
#   otherwise fall back to python3 and let it fail loudly with the real reason.
PYTHON=${PYTHON:-}
if [[ -z "$PYTHON" ]]; then
  for cand in python3 /opt/anaconda3/bin/python3; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import yaml' >/dev/null 2>&1; then
      PYTHON=$cand; break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "!! no python3 on PATH can import yaml." >&2
  echo "   Install it for the interpreter the script will use, e.g.:" >&2
  echo "     /opt/anaconda3/bin/python3 -m pip install pyyaml" >&2
  echo "   or point at one explicitly:  PYTHON=/opt/anaconda3/bin/python3 bash $0" >&2
  exit 1
fi

# THE FRONT 27,000 OF THE BHARATCODE SLICE (2026-08-14), not all 47,000 of it. The back 20,000
# -- cf027000..cf046999 -- were handed to Gemini, which is generating them now
# (scripts/cf_nightly_gemini.sh); `cf_items_70k_bc.jsonl` is the undivided slice and is still
# what `split_cf_items.py --check` validates, so it must NOT be pointed at here.
#
# **THE TAIL WENT AND THE FRONT STAYED, AND THAT ORDER IS THE WHOLE POINT.** `generate
# --resume` walks the file IN ORDER, so this run consumes forward from cf002211 -- it would
# need ~34 nights at the measured 4.53 ok/min to reach cf027000, and it stops there anyway.
# The two campaigns therefore cannot meet even if both run every night for months. Handing over
# the FRONT would have put Gemini's next request and this script's next request on the same
# anchor tonight. Same argument `split_cf_items.py` makes for why the paid slice came off the
# back, applied one level down.
#
# The other 3,000 of the 70k file are `cf_items_70k_oai.jsonl` and belong to the paid
# gpt-5-nano run (scripts/cf_nightly_openai.sh). Two endpoints generating one anchor is money
# spent twice for one training row, and the slices are what makes that impossible rather than
# unlikely. `split_cf_items.py --check` re-proves the three-way split below, every night, and
# the HANDOVER guard after it re-proves this second cut against what Gemini actually holds.
# **A SECOND 20,000 WENT TO OPENAI ON 2026-08-21** (cf007000..cf026999), because the paid
# slice was exhausted and this run has not been started since 2026-08-11. Off the BACK again,
# for the same reason: this run stopped at cf002200 and `--resume` walks forward, so at 4.53
# ok/min it would need ~18 nights to reach cf007000. What is left here is cf000000..cf006999 --
# 4,799 anchors it has never touched, still in front of it, and still its own.
# `cf_items_70k_bc_keep.jsonl` (cf000000..cf026999) is kept only so the old runs stay readable.
ITEMS=${ITEMS:-data/cf/cf_items_70k_bc_keep2.jsonl}
OUT=${OUT:-data/cf/cf70k.jsonl}
HOURS=${HOURS:-7.5}
CONCURRENCY=${CONCURRENCY:-6}
LOGDIR=${LOGDIR:-data/cf/logs}

# --no-thinking IS the default here, chosen by the human on 2026-08-10, and it is a schedule
# decision taken with the quality cost known. MEASURED (§7.5.6): thinking mode is ~12.6x more
# completion tokens, which at 70k anchors is ~130 nights against ~13 -- the campaign does not
# exist in thinking mode. Its quality cost was real and is now largely repaired: the
# `result`-is-DERIVED prompt fix removed the `result == anchor_result` defect that lost 2 of 6
# items, and `dedup_key` removed the validator bug underneath it.
# THINKING=1 opts back in, per night. Read §7.5.6's `cf_fix*` table before you do.
THINK_FLAG=${THINKING:+}
[[ -z "${THINKING:-}" ]] && THINK_FLAG=--no-thinking

if [[ ! -f "$ITEMS" ]]; then
  echo "!! $ITEMS does not exist. Build it ONCE, before the first night:" >&2
  echo "     ${PYTHON} scripts/generate_counterfactuals.py sample \\" >&2
  echo "         --max-rows 70000 --per-question 4 --limit 70000 \\" >&2
  echo "         --out data/cf/cf_items_70k.jsonl" >&2
  echo "     ${PYTHON} scripts/split_cf_items.py" >&2
  echo "   The item file must NEVER be regenerated mid-campaign: --resume matches on" >&2
  echo "   custom_id, and re-sampling renumbers every anchor, so cf004211 would silently" >&2
  echo "   become a different step of a different question and the resume would be a lie." >&2
  exit 1
fi

# The disjointness of the two endpoints' anchor sets is the one property that cannot be
# recovered after the fact -- a duplicated anchor is money already spent -- so it is re-proved
# every night rather than assumed from the day it was set up.
"$PYTHON" scripts/split_cf_items.py --check >/dev/null || {
  echo "!! the bharatcode and openai anchor slices overlap or are missing." >&2
  echo "   Run: ${PYTHON} scripts/split_cf_items.py --check" >&2
  exit 1
}

# **20,000 OF THE BHARATCODE SLICE WENT TO GEMINI ON 2026-08-14** and this guard is what makes
# the boundary a checked fact rather than a remembered one. `--resume` matches custom_ids
# inside ONE response file, so neither campaign can see the other's: if the two item files ever
# overlap, both endpoints pay for the same anchor and `L_CF` keys on the anchor, so the second
# copy is a duplicate training row rather than a second opinion.
#
# **IT MUST PASS TODAY, and that is the difference from the all-or-nothing handover this file
# carried for one day.** ITEMS is cf000000..cf026999 and gemini holds cf027000..cf046999, so
# the sets are disjoint by arithmetic and this run is EXPECTED TO PROCEED. A refusal means the
# cut moved -- somebody re-pointed ITEMS at the undivided `cf_items_70k_bc.jsonl`, or gemini's
# file was re-cut wider -- and it is the one condition under which running this costs money.
#
# It checks the real sets rather than the mere existence of the handover file, so a future
# re-cut moves the boundary here automatically. The 2,201 anchors bharatcode already generated
# are NOT the hazard -- `--resume` skips those -- so it compares only what tonight would
# actually request.
#
# **THERE ARE TWO HANDOVERS TO RE-PROVE SINCE 2026-08-21**, not one: gemini holds
# cf027000..cf046999 and openai now holds cf007000..cf026999 as well. Both are checked, each
# against the anchors THIS run would actually request tonight, and the loop means a third
# handover is one filename away rather than a rewrite.
HANDOVER=${HANDOVER:-data/cf/cf_items_gm.jsonl data/cf/cf_items_oai.jsonl}
for PEER in $HANDOVER; do
  [[ -f "$PEER" ]] || continue
  "$PYTHON" - "$ITEMS" "${OUT%.jsonl}.responses.jsonl" "$PEER" <<'PY' || exit 1
import json, sys
items, responses, peer = sys.argv[1:4]
def ids(path, nonnull=False):
    out = set()
    for line in open(path):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if nonnull and row.get("response") is None:
            continue
        if row.get("custom_id"):
            out.add(row["custom_id"])
    return out
try:
    done = ids(responses, nonnull=True)
except FileNotFoundError:
    done = set()
clash = sorted((ids(items) - done) & ids(peer))
if clash:
    print(f"!! REFUSING: {len(clash):,} of the anchors tonight would request are in "
          f"{peer},\n   which another endpoint is generating RIGHT NOW "
          f"(e.g. {', '.join(clash[:5])})."
          f"\n   Every one of them would be paid for twice for one training row.\n"
          f"   THE CUT MOVED. Since 2026-08-21 this run owns cf000000..cf006999, openai owns\n"
          f"   cf007000..cf026999 plus cf067000..cf069999, and gemini owns "
          f"cf027000..cf066999 --\n   which do not overlap, so this guard passing is the "
          f"normal case and a refusal means one\n   side was re-pointed. Check ITEMS is "
          f"cf_items_70k_bc_keep2.jsonl and not the wider\n   cf_items_70k_bc_keep.jsonl or "
          f"the undivided cf_items_70k_bc.jsonl, then re-cut the\n   peer's file if it really "
          f"did widen:\n"
          f"     python3 scripts/cf_exclude_generated.py \\\n"
          f"         --slice data/cf/cf_items_gm_src.jsonl --out data/cf/cf_items_gm.jsonl \\\n"
          f"         --self data/cf/cf70k_gm.responses.jsonl\n"
          f"     python3 scripts/cf_exclude_generated.py \\\n"
          f"         --slice data/cf/cf_items_oai_src.jsonl --out data/cf/cf_items_oai.jsonl \\\n"
          f"         --self data/cf/cf70k_oai.responses.jsonl",
          file=sys.stderr)
    raise SystemExit(1)
PY
done

mkdir -p "$LOGDIR" "$(dirname "$OUT")"
TOTAL=$(wc -l < "$ITEMS" | tr -d ' ')
RESPONSES="${OUT%.jsonl}.responses.jsonl"
DONE=0
[[ -f "$RESPONSES" ]] && DONE=$(grep -c '"response": *{' "$RESPONSES" || true)

echo "=== counterfactual night $(date '+%Y-%m-%d %H:%M') ==="
echo "  items      $ITEMS  ($TOTAL anchors)"
echo "  already    $DONE responses on disk  ->  $((TOTAL - DONE)) to go"
echo "  tonight    the next $((TOTAL - DONE)) in file order, whichever fit in ${HOURS}h"
echo "  window     ${HOURS}h at concurrency $CONCURRENCY  ${THINK_FLAG:-(thinking ON)}"
echo "  out        $OUT"
# WHAT DECIDES TONIGHT'S WORK, in one line, because it is the question everyone asks:
# `_already_done` reads <out>.responses.jsonl and keeps the custom_ids whose row has a
# NON-NULL response; `generate --resume` then requests every item in $ITEMS not in that set,
# in file order. So a FAILED item (429, timeout, degenerate) is not "done" and comes back
# round tonight, while a succeeded one never costs a second request.

CMD=("$PYTHON" scripts/generate_counterfactuals.py generate
     --items "$ITEMS" --out "$OUT"
     --resume --stop-after "$HOURS" --concurrency "$CONCURRENCY")
[[ -n "$THINK_FLAG" ]] && CMD+=("$THINK_FLAG")

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '  would run  '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

LOG="$LOGDIR/$(date '+%Y%m%d-%H%M').log"
echo "  log        $LOG"
echo

# `tee` so the night is watchable live in tmux AND readable in the morning. The exit status
# taken is the generator's, not tee's -- PIPESTATUS, because `set -e` reads the last command.
set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

LEFT=$(grep -o 'SESSION .* still to do' "$LOG" | tail -1 || true)
NOW=0
[[ -f "$RESPONSES" ]] && NOW=$(grep -c '"response": *{' "$RESPONSES" || true)

echo
echo "=== night finished $(date '+%Y-%m-%d %H:%M')  exit $STATUS ==="
[[ -n "$LEFT" ]] && echo "  $LEFT"
echo "  responses on disk  $DONE -> $NOW  (+$((NOW - DONE)) tonight, $((TOTAL - NOW)) left)"
echo "  tomorrow: run this exact command again. It picks up from $NOW."

# 130 is Ctrl-C and it is a NORMAL way to end a session, not a failure -- responses are
# appended and flushed as they land, so an interrupt costs at most the requests in flight.
# Reporting it as an error would train the reader to ignore the error line that matters.
if [[ "$STATUS" -eq 130 ]]; then
  echo "  (interrupted by hand -- nothing lost beyond the requests that were in flight)"
  exit 0
fi
if [[ "$STATUS" -ne 0 ]]; then
  echo "  NON-ZERO EXIT. Nothing is lost -- every response that landed is in $RESPONSES." >&2
  echo "  Read the tail of $LOG, then just run this script again." >&2
fi
exit "$STATUS"
