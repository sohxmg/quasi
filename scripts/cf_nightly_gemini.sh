#!/usr/bin/env bash
#
# One DAY of counterfactual generation against Gemini 3.5 Flash-Lite, on the anchors the
# Codex CLI was working through before it went down PLUS the back 20,000 of the bharatcode
# slice, handed over on 2026-08-14 because bharatcode is barely being run.
#
#   bash scripts/cf_nightly_gemini.sh                  # the whole day's quota, then stop
#   REQUESTS=40 bash scripts/cf_nightly_gemini.sh      # a smaller slice of it (a pilot)
#   EFFORT=low bash scripts/cf_nightly_gemini.sh       # 3x faster per request, MEASURED WORSE
#   PER_REQUEST=1 bash scripts/cf_nightly_gemini.sh    # one anchor per request, the old shape
#   bash scripts/cf_nightly_gemini.sh --dry-run        # print the command and the state
#
# WHICH ANCHORS, AND WHY THEY CANNOT COLLIDE WITH ANYTHING ELSE. It reads `cf_items_gm.jsonl`:
# the CODEX slice (cf047000..cf066999) and the BACK 20,000 OF THE BHARATCODE SLICE
# (cf027000..cf046999) concatenated, minus every anchor that already HOLDS a non-null response
# anywhere in data/cf. bharatcode keeps cf000000..cf026999 and OpenAI keeps cf067000..cf069999,
# so the arithmetic keeps all three campaigns apart; the exclusion is what keeps this run off
# the 1,440 anchors Codex paid for before it stopped. MEASURED 2026-08-14:
# 40,000 - 1,440 = **38,560 anchors in the file**, of which gemini has already done 13,243, so
# **25,317 to go**, ~3 days at ~9,600/day.
#
# **THE TAIL WAS HANDED OVER, NOT THE FRONT, AND NOT THE WHOLE REMAINDER.** bharatcode had
# generated 2,201 and had 44,799 left; 20,000 of those moved and 24,799 stayed. Taking them off
# the BACK is what lets bharatcode keep running: it resumes at cf002211 and `--resume` walks
# forward, so at its measured 4.53 ok/min it would need ~34 nights to reach cf027000 -- and it
# stops there anyway, because its items file ends at cf026999. Handing over the front would
# have put both endpoints on the same anchor tonight.
#
# **NOTHING WAS RE-SAMPLED, and the campaign is still 70,000 anchors.** Every id here is one
# `sample` assigned in the original file and means exactly what it meant then. A second
# `sample` run would NOT be a way to add anchors: ids are positional, so a new file's cf012345
# is a different step of a different question, which is the trap `--resume` and
# `split_cf_items.py` are both built around.
#
# THE BLOCKS ARE IN CAMPAIGN ORDER, ON PURPOSE. `generate --resume` walks in file order, so
# gemini finishes the codex slice it is 13,243 anchors into (5,317 left) before it starts on
# the bharatcode tail at cf027000. Nothing breaks if that order changes -- `--resume` keys on
# custom_id, not position -- but a half-finished slice is harder to reason about than a
# finished one.
#
# Both checks re-run below before any request is made, every day, because a duplicated anchor
# is quota already spent and cannot be recovered after the fact -- `L_CF` keys on the anchor,
# so a second rewrite set for one is a duplicate training row, not a second opinion.
#
# **BHARATCODE STILL RUNS AND IS SAFE TO RUN; CODEX IS NOT.** bharatcode resumes at cf002211,
# which is in its OWN items file (`cf_items_70k_bc_keep.jsonl`) and not in this one, so the two
# nightly scripts can both go out tonight -- `cf_nightly.sh` re-proves that against this file
# before it requests anything. Codex is the opposite case: it would resume at cf048444, which
# IS in this file, and `--resume` matches custom_ids inside ONE response file so neither side
# would see the other. If Codex comes back, re-cut the handover with `cf_exclude_generated.py`
# first, or give it a slice of its own.
#
# WHY IT STOPS ON REQUESTS AND NOT ON THE CLOCK OR ON TOKENS. This endpoint's quota is metered
# in REQUESTS, and the day's budget is gone long before the night is. The limits below are PER
# KEY -- read off the account's own chart on 2026-08-10 -- and this run holds TEN keys:
#
#   | meter                        | limit/key | what this run does per key     | headroom |
#   |------------------------------|-----------|--------------------------------|----------|
#   | requests per day (RPD)       | 500       | --request-budget 480           | 4%       |
#   | requests per minute (RPM)    | 15        | --rpm 12                       | 20%      |
#   | INPUT tokens per minute (TPM)| 250,000   | 12/min x ~4,300 = ~52,000      | 79%      |
#
# So RPD is the binding constraint, RPM is a pacing constraint, and TPM is not a constraint at
# all -- which is also why `--max-tokens` is set generously below: **output tokens are not
# metered on this endpoint**, only input ones are. `--stop-after` is a hang backstop, nothing
# more.
#
# THE TWO MULTIPLIERS, AND THE DIFFERENT THINGS THEY DO (both added 2026-08-11, and the key
# count went 3 -> 10 on 2026-08-12 when the human added GEMINI_API_KEY_4..10):
#
#   TEN KEYS           a requests-per-day quota belongs to a KEY, so ten keys is ten quotas.
#                      `--api-key-env` takes a comma-separated list of variable NAMES and
#                      rotates round-robin, each key with its OWN --rpm spacing and its OWN
#                      --request-budget. 10 x 480 = 4,800 requests, and the aggregate pace is
#                      10 x 12 = 120/min, so the day still takes ~40 minutes -- the day's
#                      length is set by one key's budget divided by one key's rate, and adding
#                      keys widens it rather than lengthening it.
#   TWO ANCHORS EACH   `--anchors-per-request 2` puts two independent anchors in one request,
#                      under a system prompt (SYSTEM_PROMPT_GROUPED) that spends a screen
#                      saying they are unrelated and must not be mixed. The ~2,300-token
#                      system prefix is SHARED between them, so the second anchor costs only
#                      its own ~950-token user message -- and on a request-metered endpoint it
#                      costs no quota at all.
#
#   4,800 requests x 2 anchors = **~9,600 anchors a day**, against 480 before the keys and the
#   batching. 18,548 to go is therefore **~2 days** rather than ~42. The responses are SPLIT
#   BACK to one row per anchor as they land, so `--resume`, `--replay`,
#   `cf_exclude_generated.py` and the report never see a batched row.
#
#   **AT THIS SIZE THE ITEMS FILE RUNS OUT BEFORE THE QUOTA DOES.** 18,548 anchors is 9,274
#   requests, i.e. under two full days, so day 2 will exit on "nothing left to do" with most of
#   the budget unspent. That is the campaign finishing, not a fault -- but it means `REQUESTS`
#   and `per key:` will both read low on the last day and neither is evidence of a problem.
#
# **THE TEN KEYS MUST BE ON DIFFERENT PROJECTS OR THIS BUYS NOTHING**, and the hazard grows
# with the count rather than staying fixed. A quota is per key only when the keys are minted in
# different Google Cloud projects; keys sharing a project share one quota, and rotating between
# them spends it ten times over while the budget counter still reads healthy -- 429s with a
# clean dashboard, which is the failure shape §14 is a catalogue of. Two variables holding the
# SAME key are refused outright by `read_keys` (all ten values were checked distinct on
# 2026-08-12). Two different keys on one project are NOT detectable from here: the tell is the
# `per key:` line in the closing summary going lopsided, and 429s in the log. **With ten keys,
# read that line as ten numbers that should all be within a few of each other** -- a subset
# sharing one project shows up as that subset stalling early while the rest run on.
#
# WHAT BATCHING COSTS, AND WHAT IT COST ONCE ALREADY. A truncated or degenerate response now
# loses TWO anchors instead of one, and an anchor whose block the model simply omits is written
# as `pair_block_missing` with `response: null` -- unspent, and back in tomorrow's queue.
#
# It also cost a real regression on first contact, MEASURED 2026-08-11 and worth knowing because
# the mechanism is not obvious: batched at `low`, six anchors yielded 3/6 against 5/6 for the
# SAME six unbatched, and the whole gap was **8 discarded positives with
# `result_differs_from_anchor`**. Not cross-contamination -- the pairing was correct. The model
# was REWORDING the `result` string on claim-asserting steps ("second digit after the decimal
# point" -> "second number past the decimal"), which `results_equal` compares as strings and
# duly discards. The base prompt already forbids it ("write them all in the same format"); the
# longer batched context was losing it. `_GROUP_ADDENDUM` now restates it in the batched
# context, and that discard count went 8 -> 0. **Watch `outcomes:`, and `PER_REQUEST=1` is the
# one-word revert to the shape every measurement before 2026-08-11 was taken on.**
#
# `--request-budget` counts HTTP REQUESTS, not items or anchors: retries, resamples after a
# degenerate answer, and the one `response_format` probe all count, because the quota counts
# them. Read `REQUESTS ... answered` and `per key:` in the closing summary for what the
# overhead actually was.
#
# **THE QUOTA RESETS AT MIDNIGHT PACIFIC, NOT AT MIDNIGHT LOCAL.** Running this twice in one
# Pacific day does not buy 2,880 requests; it buys 1,440 and then a wall of 429s that the
# retry ladder will spend the rest of the budget on. One run per Pacific day.
#
# WHAT IS MEASURED AND WHAT IS NOT. MEASURED 2026-08-10, 12 requests at ONE anchor each: 11/12
# kept, token-id AUC 0.626, 65/65 known error kinds; `response_format: json_schema` with
# `strict: true` accepted and returned verbatim, `reasoning_effort` accepted, `max_tokens` is
# the right field. MEASURED 2026-08-11, five 3-request arms over the SAME six tail anchors
# (cf066994..cf066999), which is what the EFFORT table below and the prompt fix rest on -- and
# it is the arm-to-arm comparison that is worth reading, since every arm drew the same anchors.
# The three-key rotation is measured live: `per key: 1 / 1 / 1` on every one of those runs.
#
# **NOT MEASURED: the yield of either shape at any scale.** n = 6, one contiguous latex-heavy
# block, so 6/6 is not 100% any more than the earlier 3/6 was 50%. Every AUC here is on the
# regex fallback tokenizer (`transformers` is not installed on this box). Two things that could
# still go either way at 2,880 anchors a day: whether `high` holds its yield off this block, and
# whether the three keys are really three quotas. **Run REQUESTS=40 first, read `outcomes:` and
# `per key:` -- not the AUC alone -- before spending a full day on it.**
#
# WHY `high` HERE AND `low` EVERYWHERE ELSE. §7.5.7 chose `low` on the OpenAI family for COST:
# thinking tokens are billed there, and `medium` on gpt-5-nano spent the entire cap thinking and
# returned empty content on 2 of 2. Neither half of that argument survives on this endpoint. The
# meter is REQUESTS; output tokens are not metered at all; and the pace is set by the per-key
# `--rpm` spacing, so a slower request does not make a shorter day. That leaves only what effort
# buys, and what it buys is the one defect `low` is known for -- negatives that report the
# anchor's own result because the per-candidate arithmetic was never done. MEASURED at the
# EFFORT table below: 4/6 -> 5/6 -> 6/6 across low/medium/high on the same six anchors.

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

# NOT `cf_items_gm_src.jsonl` -- that is the raw union of the two blocks, and 1,440 of its
# anchors were already generated by Codex. This is the same file minus those, built once by
# `cf_exclude_generated.py` and copied LINE FOR LINE so every custom_id is unchanged. (It
# supersedes `cf_items_70k_cx_gm.jsonl`, the codex-only handover this run read from 2026-08-10
# to 2026-08-14; that file is a strict subset of this one and is kept only so the old runs stay
# readable.)
ITEMS=${ITEMS:-data/cf/cf_items_gm.jsonl}
# The union the items file is checked against, below. Two blocks, concatenated line for line:
# the codex slice and the back 20,000 of the bharatcode slice.
SRC_SLICE=${SRC_SLICE:-data/cf/cf_items_gm_src.jsonl}
OUT=${OUT:-data/cf/cf70k_gm.jsonl}
MODEL=${MODEL:-gemini-3.5-flash-lite}
BASE_URL=${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
# The environment VARIABLE NAMES, never the keys -- a key on the command line is in `ps` and in
# shell history for the whole run. Comma-separated, and each one carries its own quota.
KEY_ENVS=${KEY_ENVS:-GEMINI_API_KEY,GEMINI_API_KEY_2,GEMINI_API_KEY_3,GEMINI_API_KEY_4,GEMINI_API_KEY_5,GEMINI_API_KEY_6,GEMINI_API_KEY_7,GEMINI_API_KEY_8,GEMINI_API_KEY_9,GEMINI_API_KEY_10}
# 480 of the 500 RPD, PER KEY. The 20-request margin is insurance, not caution: overrunning
# RPD does not stop the run, it turns every remaining request into a 429 that the retry ladder
# spends three more requests on, so the overshoot costs ~4x what it saves.
REQUESTS=${REQUESTS:-480}
# 12 of the 15 RPM, PER KEY, enforced as a 5-second minimum spacing on that key and shared
# across the worker threads. Three keys therefore go out at ~36/min in aggregate.
RPM=${RPM:-12}
# Two anchors per request, which is what turns 1,440 requests into ~2,880 anchors. Set
# PER_REQUEST=1 to go back to the shape every measurement before 2026-08-11 was taken on.
PER_REQUEST=${PER_REQUEST:-2}
# DERIVED FROM THE KEY COUNT, and it has to be: `--rpm` sets the pace and this only sets how
# many may be in the air while it does, so the number that matters is rate x latency. The
# aggregate pace is N_KEYS x RPM, and a two-anchor request at `high` takes ~10.5 s (MEASURED
# 2026-08-11), so 10 keys at 12/min is 2/s x 10.5 s = ~21 requests that must be in flight just
# to keep the spacing gates saturated. **A fixed 12 would have throttled the day to ~57% of the
# quota and looked like the endpoint was slow.** The 20 s latency budget below is deliberately
# above the measured 10.5 s -- a thread that has nothing to wait for costs nothing, because the
# spacing gate and not this is what stops bursts, while a thread that is missing costs quota.
# It lands at 40 for ten keys and reproduces the old 12 at three exactly.
CONCURRENCY=${CONCURRENCY:-0}
# **`high`, and it is FREE HERE.** MEASURED 2026-08-11 on six anchors, batched two per request,
# every arm on the SAME six so the draw cannot explain the difference:
#
#   | reasoning_effort | kept | discarded rewrites          | completion tok/item | s/request |
#   |------------------|------|-----------------------------|---------------------|-----------|
#   | low              | 4/6  | 5 negative/result_equals_anchor | 1,137           | 3.5       |
#   | medium           | 5/6  | 3 positive + 2 negative     | 1,303               | 3.5       |
#   | **high**         | **6/6** | **0**                    | **1,314**           | 10.5      |
#
# The whole §7.5.7 argument for `low` was COST, and there is no cost here: the meter is
# requests, output tokens are unmetered, and 1,314 against 1,137 is noise on an input-TPM limit
# at ~21%. The 3x wall clock is not a cost either -- the pace is set by the 5-second per-key
# `--rpm` spacing, not by latency, so the day is ~40 minutes at every effort. What `high` buys
# is exactly the defect `low` was known to have: negatives that report the ANCHOR'S OWN result
# because the model never did the per-candidate arithmetic. n = 6, ONE contiguous latex-heavy
# block -- a pilot, not a rate.
EFFORT=${EFFORT:-high}
# Output tokens are NOT metered on this endpoint (the TPM limit is on INPUT tokens), so this
# is pure slack and costs nothing when unused. Do NOT lower it to "save quota" -- a truncated
# item spends a whole request and now yields nothing for TWO anchors, and a request is the
# thing that is actually scarce here.
MAX_TOKENS=${MAX_TOKENS:-24000}
# A HANG guard, not a budget. The request budget is what ends a normal day, in ~40 minutes.
HOURS=${HOURS:-3}
LOGDIR=${LOGDIR:-data/cf/logs}

if [[ -f .env ]]; then
  # Only what is not already exported. `.env` holds five other secrets.
  set -a; . ./.env; set +a
fi
# Checked HERE as well as in `read_keys`, so a missing key fails before the slice checks spend
# a minute reading 20,000 anchors off disk. All-or-nothing on purpose: quietly rotating between
# two of three keys spends the survivors' quota 1.5x faster than --request-budget expects.
MISSING=""
IFS=',' read -r -a _KEY_NAMES <<< "$KEY_ENVS"
for name in "${_KEY_NAMES[@]}"; do
  [[ -n "${!name:-}" ]] || MISSING="$MISSING $name"
done
if [[ -n "$MISSING" ]]; then
  echo "!! unset or empty in the environment and in .env:$MISSING" >&2
  echo "   Add them to .env, or name only the keys you have:" >&2
  echo "     KEY_ENVS=GEMINI_API_KEY bash $0" >&2
  echo "   (each key carries its own daily quota, so dropping one costs a third of the day)" >&2
  exit 1
fi
N_KEYS=${#_KEY_NAMES[@]}

# rate x latency, per the CONCURRENCY comment above. 20 s of latency budget against a pace of
# N_KEYS x RPM per minute, floored at 4 so a one-key run still has a little parallelism.
if [[ "$CONCURRENCY" -eq 0 ]]; then
  CONCURRENCY=$(( (N_KEYS * RPM * 20 + 59) / 60 ))
  [[ "$CONCURRENCY" -lt 4 ]] && CONCURRENCY=4
fi

if [[ ! -f "$ITEMS" ]]; then
  echo "!! $ITEMS does not exist. Build it ONCE, before the first day:" >&2
  echo "     ${PYTHON} scripts/split_cf_items.py --check  # the three campaign slices" >&2
  echo "     cat data/cf/cf_items_70k_cx.jsonl data/cf/cf_items_70k_bc_gm.jsonl \\" >&2
  echo "         > $SRC_SLICE" >&2
  echo "     ${PYTHON} scripts/cf_exclude_generated.py \\" >&2
  echo "         --slice $SRC_SLICE \\" >&2
  echo "         --out $ITEMS \\" >&2
  echo "         --self ${OUT%.jsonl}.responses.jsonl" >&2
  exit 1
fi
if [[ ! -f "$SRC_SLICE" ]]; then
  echo "!! $SRC_SLICE does not exist -- it is what the items file is checked against." >&2
  echo "     cat data/cf/cf_items_70k_cx.jsonl data/cf/cf_items_70k_bc_gm.jsonl \\" >&2
  echo "         > $SRC_SLICE" >&2
  exit 1
fi

# Re-proved every day rather than assumed from the day it was set up.
"$PYTHON" scripts/split_cf_items.py --check >/dev/null || {
  echo "!! the bharatcode, codex and openai anchor slices overlap or are missing." >&2
  echo "   Run: ${PYTHON} scripts/split_cf_items.py --check" >&2
  exit 1
}

# The slice check covers the three CAMPAIGNS. This covers what Codex and bharatcode already
# generated inside the two slices this run inherited, and any pilot that wrote to its own
# --out and is therefore invisible to --resume.
"$PYTHON" scripts/cf_exclude_generated.py --check \
    --slice "$SRC_SLICE" \
    --out "$ITEMS" \
    --self "${OUT%.jsonl}.responses.jsonl" >/dev/null || {
  echo "!! the gemini items file contains anchors generated elsewhere -- most likely codex" >&2
  echo "   came back, or bharatcode was pointed past cf026999. Re-cut it:" >&2
  echo "     ${PYTHON} scripts/cf_exclude_generated.py \\" >&2
  echo "         --slice $SRC_SLICE --out $ITEMS \\" >&2
  echo "         --self ${OUT%.jsonl}.responses.jsonl" >&2
  exit 1
}

mkdir -p "$LOGDIR" "$(dirname "$OUT")"
TOTAL=$(wc -l < "$ITEMS" | tr -d ' ')
RESPONSES="${OUT%.jsonl}.responses.jsonl"
DONE=0
[[ -f "$RESPONSES" ]] && DONE=$(grep -c '"response": *{' "$RESPONSES" || true)

DAY_REQUESTS=$(( REQUESTS * N_KEYS ))
DAY_ANCHORS=$(( DAY_REQUESTS * PER_REQUEST ))
LEFT=$(( TOTAL - DONE ))

echo "=== gemini counterfactual day $(date '+%Y-%m-%d %H:%M') ==="
echo "  items      $ITEMS  ($TOTAL anchors: the CODEX slice minus what codex did,"
echo "             plus the back 20,000 of the BHARATCODE slice -- disjoint from"
echo "             bharatcode's own cf000000..cf026999 and from openai)"
echo "  already    $DONE responses on disk  ->  $LEFT to go"
echo "  model      $MODEL  reasoning_effort=$EFFORT  at $BASE_URL"
echo "  keys       $N_KEYS, rotated round-robin, EACH WITH ITS OWN QUOTA"
echo "             $KEY_ENVS"
echo "             -- if any of them share a Google project they share one quota and this buys 429s"
echo "  budget     $REQUESTS requests/key x $N_KEYS = $DAY_REQUESTS requests, paced at $RPM/min/key"
echo "  pace       $(( N_KEYS * RPM ))/min aggregate, $CONCURRENCY in flight (rate x a 20s latency budget)"
echo "  batching   $PER_REQUEST anchors per request  ->  up to $DAY_ANCHORS anchors today"
echo "  expect     ~$(( REQUESTS / RPM )) minutes, NOT ${HOURS}h -- the request budget is what ends the day"
echo "             ~$(( (LEFT + DAY_ANCHORS - 1) / DAY_ANCHORS )) more days at that rate"
echo "  out        $OUT"

CMD=("$PYTHON" scripts/generate_counterfactuals.py generate
     --items "$ITEMS" --out "$OUT"
     --base-url "$BASE_URL" --api-key-env "$KEY_ENVS"
     --model "$MODEL"
     --reasoning-effort "$EFFORT"
     --max-tokens "$MAX_TOKENS"
     --anchors-per-request "$PER_REQUEST"
     --rpm "$RPM"
     --request-budget "$REQUESTS"
     --concurrency "$CONCURRENCY"
     --resume --stop-after "$HOURS")
# NOTE there is no --no-thinking and there must not be: it sends `chat_template_kwargs`, a
# vLLM extension. `--reasoning-effort` is this family's equivalent knob and is MEASURED
# accepted here. There is no --prompt-cache-key either -- Gemini caches the shared prefix
# implicitly, the field is an OpenAI one, and input tokens are at ~21% of their limit anyway.

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '  would run  '; printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

LOG="$LOGDIR/gemini-$(date '+%Y%m%d-%H%M').log"
echo "  log        $LOG"
echo

set +e
"${CMD[@]}" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

NOW=0
[[ -f "$RESPONSES" ]] && NOW=$(grep -c '"response": *{' "$RESPONSES" || true)

echo
echo "=== gemini day finished $(date '+%Y-%m-%d %H:%M')  exit $STATUS ==="
grep -E '^\[generate\] (REQUESTS|per key|per item)' "$LOG" | sed 's/^/  /' || true
echo "  responses on disk  $DONE -> $NOW  (+$((NOW - DONE)) today, $((TOTAL - NOW)) left)"
echo
echo "  READ 'outcomes:' in ${OUT%.jsonl}.report.md -- not the AUC alone (§7.5.6). Three things"
echo "  to look for, in this order:"
echo "    * 'per key' above: $N_KEYS numbers that should all be within a few of each other."
echo "      A SUBSET stalling early while the rest run on -> those keys share one project's"
echo "      quota, so they are one quota being spent several times. Drop them from KEY_ENVS."
echo "      (A low but EVEN row is just the items file running out, which is the campaign"
echo "      finishing -- $DAY_ANCHORS anchors/day against $LEFT left at the start of today.)"
echo "    * pair_block_missing in outcomes -> the model answered one anchor and dropped the"
echo "      other. Those anchors are unspent and come back tomorrow; if it is common,"
echo "      PER_REQUEST=1."
echo "    * negatives reporting the anchor's own result, and positives whose result 'differs'"
echo "      only symbolically (§16.30) -- the two failures that matter on cheap thinking."
echo "  tomorrow, AFTER the quota resets at midnight Pacific: run this exact command again."

# 130 is Ctrl-C and it is a NORMAL way to end a session -- responses are appended and flushed
# as they land, so an interrupt costs at most the requests in flight.
if [[ "$STATUS" -eq 130 ]]; then
  echo "  (interrupted by hand -- nothing lost beyond the requests that were in flight)"
  exit 0
fi
if [[ "$STATUS" -ne 0 ]]; then
  echo "  NON-ZERO EXIT. Nothing is lost -- every response that landed is in $RESPONSES." >&2
  echo "  Read the tail of $LOG, then just run this script again." >&2
fi
exit "$STATUS"
