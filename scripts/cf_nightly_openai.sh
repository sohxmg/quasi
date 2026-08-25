#!/usr/bin/env bash
#
# One NIGHT of counterfactual generation against OpenAI's gpt-5.6-luna, run ALONGSIDE
# scripts/cf_nightly.sh (bharatcode) on a DISJOINT set of anchors.
#
#   bash scripts/cf_nightly_openai.sh                 # the 2,000,000-token grant, then stop
#   BUDGET=500000 bash scripts/cf_nightly_openai.sh   # a smaller slice of it
#   MODEL=gpt-5-nano-2025-08-07 bash scripts/...      # the old, weak, cheap endpoint
#   bash scripts/cf_nightly_openai.sh --dry-run       # print the command and the state
#
# WHY LUNA AND NOT gpt-5-nano (2026-08-10, human's call, and the measurements agree). nano at
# `low` yielded 3 of 20 items (§7.5.7) and its dominant failure was positives that CHANGED THE
# RESULT -- 29 discards, of which only 2 are symbolically equal to the anchor, so the rewrites
# were genuinely wrong and the lever was model capability, not the validator. nano at `medium`
# yielded 0 of 2: it spent the whole token cap on reasoning and returned empty content. The
# same model family reached through the Codex CLI (gpt-5.6-luna) validated 476 of 502 anchors.
# MEASURED on this endpoint 2026-08-10, 2 anchors at `low`: 2/2 ok, full 3 positives and 6
# negatives each, ZERO rewrites dropped by validation, parity held (AUC 0.556).
#
# WHICH ANCHORS, SINCE 2026-08-21. The original slice (cf067000..cf069999) IS EXHAUSTED: the
# night of 2026-08-21 requested nothing, because all 2,980 eligible anchors of it already had
# responses. So **20,000 OF WHAT WAS LEFT OF THE BHARATCODE SLICE WERE HANDED OVER**
# (cf007000..cf026999), which bharatcode had not touched and, not having run since 2026-08-11,
# was not going to. It reads `cf_items_oai.jsonl`: the two blocks concatenated, minus every
# anchor that already HOLDS a non-null response anywhere else in data/cf. MEASURED 2026-08-21:
# 23,000 - 20 pilot anchors = **22,980 in the file**, of which 2,980 are already done, so
# **20,000 to go** -- ~40 nights at the ~507/night the 2M grant buys.
#
# **THE TAIL WAS HANDED OVER, NOT THE FRONT.** bharatcode keeps cf000000..cf006999: it stopped
# at cf002200 and `--resume` walks FORWARD, so at its measured 4.53 ok/min it would need ~18
# nights just to reach the new boundary, and its items file ends there anyway. Handing over the
# front would have put both endpoints on the same anchor tonight.
#
# WHY IT CANNOT COLLIDE WITH THE BHARATCODE OR GEMINI RUNS. Three checks, all before any money
# is spent. `split_cf_items.py --check` re-proves the three IMMUTABLE campaign slices share no
# custom_id -- those did not move and must not; only which endpoint reads which block changed.
# `cf_exclude_generated.py --check` holds out every anchor already generated anywhere else,
# which is what covers a handover and the PILOTS (`cf_oai_pilot_low` ran 20 anchors from the
# oai slice into its own `--out`, invisible to `--resume`, and would have been paid for twice).
# And the peer guard re-proves, TONIGHT, that nothing this run would request is in gemini's or
# bharatcode's items file. Nothing else is shared -- separate `--out`, so separate
# `.responses.jsonl`, separate `--resume` state, separate report.
#
# WHY IT STOPS ON TOKENS AND NOT ON THE CLOCK. bharatcode is free and rate-limited, so its
# constraint is hours. This one is paid and fast, so its constraint is the grant:
# `--token-budget` stops SUBMITTING at 2M tokens, drains what is in flight (already paid for),
# writes the report, and `--resume` continues from exactly there when there is more budget.
# `--stop-after` is still set as a backstop in case the endpoint hangs rather than answers.
#
# COST SHAPE, so the budget is readable. MEASURED on luna at `low`, 2026-08-10:
#   prompt      2,958 /item   (1,164 of it already served from cache on request TWO)
#   completion    934 /item   (456 of which are reasoning tokens)
#   TOTAL       3,892 /item   ->  the 2,000,000-token grant buys ~514 anchors, ~4,600 rewrites
# The prompt is the LARGER half of the bill on this workload, which is why the byte-identical
# SYSTEM PROMPT matters: OpenAI's automatic prefix cache keys on it above 1,024 tokens, and
# `--prompt-cache-key` pins requests to the same cache. Cached prompt tokens bill at a
# fraction of the input rate but are counted against the budget ANYWAY -- cheaper is not free,
# and a budget that ignored them would overshoot by whatever the hit rate turned out to be.
# Read `served from cache` in the run's closing summary for the discount actually obtained.
#
# NOTE ON DOLLARS vs TOKENS. `--token-budget` stops at 2,000,000 TOKENS. luna is a frontier
# model and its per-token price is far above nano's, so if the grant is denominated in dollars
# rather than tokens, this stops in the right place only by coincidence. Check the bill after
# night one before assuming the second night is affordable.

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

# NOT `cf_items_oai_src.jsonl` -- that is the raw union of the two blocks, and 20 of its
# anchors were already generated by the nano pilot. This is the same file minus those, built
# once by `cf_exclude_generated.py` and copied LINE FOR LINE so every custom_id is unchanged.
# (It supersedes `cf_items_70k_oai_luna.jsonl`, the 2,980-anchor file this run read from
# 2026-08-10 to 2026-08-21, when the original slice was exhausted; that file is a strict
# subset of this one and is kept only so the old runs stay readable.)
ITEMS=${ITEMS:-data/cf/cf_items_oai.jsonl}
# The union the items file is checked against, below. Two blocks, concatenated line for line:
# the original oai slice and the back 20,000 of what was left of the bharatcode slice.
SRC_SLICE=${SRC_SLICE:-data/cf/cf_items_oai_src.jsonl}
OUT=${OUT:-data/cf/cf70k_oai.jsonl}
MODEL=${MODEL:-gpt-5.6-luna}
BASE_URL=${BASE_URL:-https://api.openai.com/v1}
BUDGET=${BUDGET:-2000000}
# `low`, not `minimal` and not `medium`. MEASURED on the vLLM endpoint (§7.5.6): with the chain
# of thought off the model stops doing the per-candidate arithmetic and reports the ANCHOR'S
# OWN result for every negative, losing whole items to the `result != anchor_result` check --
# `minimal` is the same trade on this family. Upward, `medium` on nano returned empty content
# for 2 of 2. `low` on luna is MEASURED at 456 reasoning tokens/item and 2/2 clean, so it is
# both the cheapest setting that does the algebra and, on this model, sufficient.
EFFORT=${EFFORT:-low}
CONCURRENCY=${CONCURRENCY:-8}
# max_completion_tokens, which on a reasoning model is SHARED with the chain of thought.
# MEASURED on luna at `low`: 934 completion tokens/item, 456 of them reasoning -- so 16000 is
# ~17x the observed need. It is SLACK, not a target: the cap exists so one runaway anchor
# cannot eat the grant, and it is priced only when used. Do NOT lower it to "save budget" --
# a truncated item costs every token it spent and yields nothing at all.
MAX_TOKENS=${MAX_TOKENS:-16000}
HOURS=${HOURS:-7.5}
LOGDIR=${LOGDIR:-data/cf/logs}

if [[ -f .env ]]; then
  # Only the key, and only if it is not already exported. `.env` holds five other secrets.
  set -a; . ./.env; set +a
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "!! OPENAI_API_KEY is unset and .env did not provide it." >&2
  exit 1
fi

if [[ ! -f "$ITEMS" ]]; then
  echo "!! $ITEMS does not exist. Build it ONCE, after a handover:" >&2
  echo "     ${PYTHON} scripts/split_cf_items.py --check  # the three campaign slices" >&2
  echo "     ${PYTHON} scripts/cf_exclude_generated.py \\" >&2
  echo "         --slice $SRC_SLICE --out $ITEMS \\" >&2
  echo "         --self ${OUT%.jsonl}.responses.jsonl" >&2
  exit 1
fi

# Re-proved every night rather than assumed from the day it was set up, because a duplicated
# anchor is money already spent and cannot be recovered after the fact.
"$PYTHON" scripts/split_cf_items.py --check >/dev/null || {
  echo "!! the bharatcode, codex and openai anchor slices overlap or are missing." >&2
  echo "   Run: ${PYTHON} scripts/split_cf_items.py --check" >&2
  exit 1
}

# The slice check covers the three CAMPAIGNS. This covers what bharatcode already generated
# inside the block this run inherited, and the pilots, which wrote to their own `--out` and are
# therefore invisible to `--resume`.
"$PYTHON" scripts/cf_exclude_generated.py --check \
    --slice "$SRC_SLICE" \
    --out "$ITEMS" \
    --self "${OUT%.jsonl}.responses.jsonl" >/dev/null || {
  echo "!! the openai items file contains anchors generated elsewhere -- most likely" >&2
  echo "   bharatcode was restarted past cf006999. Re-cut it:" >&2
  echo "     ${PYTHON} scripts/cf_exclude_generated.py \\" >&2
  echo "         --slice $SRC_SLICE --out $ITEMS \\" >&2
  echo "         --self ${OUT%.jsonl}.responses.jsonl" >&2
  exit 1
}

# **20,000 OF WHAT WAS LEFT OF THE BHARATCODE SLICE CAME HERE ON 2026-08-21** (cf007000..
# cf026999), because the original oai slice was exhausted -- 2,980 of 2,980 -- and bharatcode
# has not been run since 2026-08-11. This guard is the one `cf_nightly.sh` already carries,
# from the other side: `--resume` matches custom_ids inside ONE response file, so neither
# campaign can see the other's, and if two item files overlap both endpoints pay for the same
# anchor for one training row. bharatcode now owns cf000000..cf006999 and gemini
# cf027000..cf066999, so the sets are disjoint BY ARITHMETIC and this run is EXPECTED TO
# PROCEED. A refusal means somebody re-pointed one side at an undivided file.
for PEER in ${PEERS:-data/cf/cf_items_gm.jsonl data/cf/cf_items_70k_bc_keep2.jsonl}; do
  [[ -f "$PEER" ]] || continue
  "$PYTHON" - "$ITEMS" "${OUT%.jsonl}.responses.jsonl" "$PEER" <<'GUARD' || exit 1
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
    print(f"!! REFUSING: {len(clash):,} of the anchors tonight would request are also in "
          f"{peer}\n   (e.g. {', '.join(clash[:5])}). Every one would be PAID FOR TWICE "
          f"for one training row.\n   THE CUT MOVED. Since 2026-08-21 this run owns "
          f"cf007000..cf026999 plus cf067000..cf069999,\n   bharatcode owns "
          f"cf000000..cf006999 and gemini cf027000..cf066999.", file=sys.stderr)
    raise SystemExit(1)
GUARD
done

mkdir -p "$LOGDIR" "$(dirname "$OUT")"
TOTAL=$(wc -l < "$ITEMS" | tr -d ' ')
RESPONSES="${OUT%.jsonl}.responses.jsonl"
DONE=0
[[ -f "$RESPONSES" ]] && DONE=$(grep -c '"response": *{' "$RESPONSES" || true)

echo "=== openai counterfactual night $(date '+%Y-%m-%d %H:%M') ==="
echo "  items      $ITEMS  ($TOTAL anchors: the exhausted tail of the 70k"
echo "             plus the 20,000 handed over from bharatcode on 2026-08-21,"
echo "             minus every anchor already generated elsewhere)"
echo "  already    $DONE responses on disk  ->  $((TOTAL - DONE)) to go"
echo "  model      $MODEL  reasoning_effort=$EFFORT  at $BASE_URL"
echo "  budget     $BUDGET total tokens (prompt + completion), concurrency $CONCURRENCY"
echo "  out        $OUT"

CMD=("$PYTHON" scripts/generate_counterfactuals.py generate
     --items "$ITEMS" --out "$OUT"
     --base-url "$BASE_URL" --api-key-env OPENAI_API_KEY
     --model "$MODEL"
     --reasoning-effort "$EFFORT"
     --max-tokens "$MAX_TOKENS"
     --completion-token-param max_completion_tokens
     --temperature -1
     --prompt-cache-key feynman-prm-cf-v1
     --token-budget "$BUDGET" --token-budget-scope total
     --concurrency "$CONCURRENCY"
     --resume --stop-after "$HOURS")
# NOTE there is no --no-thinking here and there must not be: it sends `chat_template_kwargs`,
# a vLLM extension that OpenAI rejects with a 400. `--reasoning-effort` is this family's
# equivalent knob.

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '  would run  '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

LOG="$LOGDIR/openai-$(date '+%Y%m%d-%H%M').log"
echo "  log        $LOG"
echo

set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

NOW=0
[[ -f "$RESPONSES" ]] && NOW=$(grep -c '"response": *{' "$RESPONSES" || true)

echo
echo "=== openai night finished $(date '+%Y-%m-%d %H:%M')  exit $STATUS ==="
grep -E '^\[generate\] (BUDGET|per item)' "$LOG" | sed 's/^/  /' || true
echo "  responses on disk  $DONE -> $NOW  (+$((NOW - DONE)) tonight, $((TOTAL - NOW)) left)"
echo "  tomorrow: run this exact command again. It picks up from $NOW."

if [[ "$STATUS" -eq 130 ]]; then
  echo "  (interrupted by hand -- nothing lost beyond the requests that were in flight)"
  exit 0
fi
if [[ "$STATUS" -ne 0 ]]; then
  echo "  NON-ZERO EXIT. Nothing is lost -- every response that landed is in $RESPONSES." >&2
  echo "  Read the tail of $LOG, then just run this script again." >&2
fi
exit "$STATUS"
