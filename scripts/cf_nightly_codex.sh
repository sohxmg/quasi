#!/usr/bin/env bash
#
# One NIGHT of counterfactual generation through the CODEX CLI, run alongside
# scripts/cf_nightly.sh (bharatcode) and scripts/cf_nightly_openai.sh on DISJOINT anchors.
#
#   bash scripts/cf_nightly_codex.sh                  # 7.5 h, then stop submitting
#   CONCURRENCY=16 bash scripts/cf_nightly_codex.sh   # back off if the plan starts shedding
#   bash scripts/cf_nightly_codex.sh --dry-run        # print the command and the state
#
# WHY IT CANNOT COLLIDE. It reads `cf_items_70k_cx.jsonl` (cf047000..cf066999); bharatcode
# reads cf000000..cf046999 and OpenAI reads cf067000..cf069999. `split_cf_items.py --check`
# re-proves all three are pairwise disjoint before any request is made, every night. Separate
# `--out`, so separate `.responses.jsonl`, separate `--resume` state, separate report.
#
# WHY IT STOPS ON THE CLOCK. There is no token meter to stop on -- the Codex CLI bills against
# a subscription, not a grant -- so `--stop-after` is the only bound, exactly as for bharatcode.
#
# WHAT WAS MEASURED BEFORE THIS SCRIPT WAS WRITTEN (2026-08-10, live, §7.5.8). Concurrency
# 1/2/4/8/16/24 -> 1.70/2.83/4.19/6.92/14.3/18.5 ok/min, ZERO failures at every level and no
# rate-limit wall found. Validation yield 23/24 on a mixed draw and 13/16 on a latex-only one,
# against gpt-5-nano's 3/20 (§7.5.7).
#
# **THE ONE THING THOSE NUMBERS DO NOT COVER, AND IT IS THIS SCRIPT'S WHOLE RISK.** Every level
# above was a burst of 52-78 SECONDS. This script runs for 7.5 HOURS, which is ~350x the
# requests, and subscription plans meter on rolling windows a one-minute burst cannot touch.
# So the sweep establishes that nothing breaks at 24 concurrent streams; it establishes NOTHING
# about the tenth hour. `--concurrency 24` is therefore a hypothesis and night 1 is its test:
# read `failed` in the closing summary. If it is non-zero and clustered late, the plan started
# shedding and CONCURRENCY should come down -- that is a scheduling fact, not a fault, and
# `--resume` means it costs nothing but time.

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

ITEMS=${ITEMS:-data/cf/cf_items_70k_cx.jsonl}
OUT=${OUT:-data/cf/cf70k_cx.jsonl}
# Empty = whatever ~/.codex/config.toml selects (gpt-5.6-luna as of 2026-08-10). Left empty on
# purpose: pinning a model here would silently override the interactive `/model` choice, and
# the config file is the one place a human already expects to set it.
MODEL=${MODEL:-}
CONCURRENCY=${CONCURRENCY:-24}
# Per-anchor wall clock. MEASURED at ~39 s serial and ~3 s/item amortised at concurrency 24, so
# 900 s is ~23x the observed worst case. It is a HANG guard, not a budget: a Codex session that
# has stopped making progress should be abandoned and retried, and `--resume` will pick the
# anchor back up on the next night.
TIMEOUT=${TIMEOUT:-900}
HOURS=${HOURS:-7.5}
LOGDIR=${LOGDIR:-data/cf/logs}

command -v "${CODEX_BIN:-codex}" >/dev/null 2>&1 || {
  echo "!! ${CODEX_BIN:-codex} is not on PATH." >&2
  echo "   It lives at ~/.local/bin/codex; add that to PATH or set CODEX_BIN." >&2
  exit 1
}

if [[ ! -f "$ITEMS" ]]; then
  echo "!! $ITEMS does not exist. Build the slices ONCE, before the first night:" >&2
  echo "     ${PYTHON} scripts/split_cf_items.py" >&2
  exit 1
fi

"$PYTHON" scripts/split_cf_items.py --check >/dev/null || {
  echo "!! the three anchor slices overlap or are missing." >&2
  echo "   Run: ${PYTHON} scripts/split_cf_items.py --check" >&2
  exit 1
}

mkdir -p "$LOGDIR" "$(dirname "$OUT")"
TOTAL=$(wc -l < "$ITEMS" | tr -d ' ')
RESPONSES="$OUT.responses.jsonl"
DONE=0
[[ -f "$RESPONSES" ]] && DONE=$(grep -c '"response": *{' "$RESPONSES" || true)

echo "=== codex counterfactual night $(date '+%Y-%m-%d %H:%M') ==="
echo "  items      $ITEMS  ($TOTAL anchors, cf047000..cf066999 -- disjoint from bc and oai)"
echo "  already    $DONE responses on disk  ->  $((TOTAL - DONE)) to go"
echo "  model      ${MODEL:-<from ~/.codex/config.toml>}  via codex exec"
echo "  bound      ${HOURS}h wall clock, concurrency $CONCURRENCY"
echo "  out        $OUT"

CMD=("$PYTHON" scripts/generate_via_codex.py
     --items "$ITEMS" --out "$OUT"
     --concurrency "$CONCURRENCY"
     --timeout "$TIMEOUT"
     --resume --stop-after "$HOURS")
[[ -n "$MODEL" ]] && CMD+=(--model "$MODEL")

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '  would run  '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

LOG="$LOGDIR/codex-$(date '+%Y%m%d-%H%M').log"
echo "  log        $LOG"
echo

set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

NOW=0
[[ -f "$RESPONSES" ]] && NOW=$(grep -c '"response": *{' "$RESPONSES" || true)

echo
echo "=== codex night finished $(date '+%Y-%m-%d %H:%M')  exit $STATUS ==="
echo "  responses on disk  $DONE -> $NOW  (+$((NOW - DONE)) tonight, $((TOTAL - NOW)) left)"
echo
echo "  VALIDATE IT (free, no quota, no requests):"
echo "    ${PYTHON} scripts/generate_counterfactuals.py generate \\"
echo "      --items $ITEMS --out $OUT --replay $RESPONSES"
echo "  then read 'outcomes:' in ${OUT%.jsonl}.report.md -- not the AUC alone (§7.5.6)."
echo "  tomorrow: run this exact command again. It picks up from $NOW."

if [[ "$STATUS" -eq 130 ]]; then
  echo "  (interrupted by hand -- nothing lost beyond the anchors that were in flight)"
  exit 0
fi
if [[ "$STATUS" -ne 0 ]]; then
  echo "  NON-ZERO EXIT. Nothing is lost -- every response that landed is in $RESPONSES." >&2
  echo "  Read the tail of $LOG, then just run this script again." >&2
fi
exit "$STATUS"
