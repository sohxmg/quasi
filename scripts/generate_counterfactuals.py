#!/usr/bin/env python
"""Generate the ④ L_CF dataset (§7.5) with an LLM.

    sample    Math-Shepherd train questions -> cf_items.jsonl   (offline, no API)

    generate  cf_items.jsonl -> counterfactuals.jsonl           LIVE, one call per item,
              against any OpenAI-compatible /chat/completions endpoint (BharatCode; see
              `secret.txt`). This is the path that runs today.

    submit    cf_items.jsonl -> one Anthropic Batch API job -> cf_batch.json
    collect   cf_batch.json  -> counterfactuals.jsonl          the batch equivalent

Both writing paths end in `finish()`, which writes through
`feynman_prm.data.counterfactual.write_jsonl` -- so the on-disk format is the loader's
format by construction, not by agreement -- plus a `.report.md` a human can read and a
`.rejected.jsonl` / `.raw.jsonl` pair for anything thrown away.

**One example is an EQUIVALENCE CLASS, not a pair** (multi-positive L_CF, 2026-08-08). The
prompt asks for 3 positives and 6 negatives; `validate` keeps >=1 and >=3 (LOWERED from
2 and 5 on 2026-08-15 -- see `_add_count_args`). Positives are dropped individually rather
than killing the item.

**The generation rule that makes or breaks this dataset.** `phi_i = phi(h_{i-1}, act_emb_i)`
and `act_emb` is a MEAN OF INPUT EMBEDDINGS of the step (§6.4). So `d(phi(anchor), phi(v))`
is driven hard by how many tokens `v` shares with the anchor, and ANY correlation between
wording and correctness is a shortcut the loss takes instead of learning meaning. **Both
directions are fatal.** A positive that is a near-copy of the anchor is picked by token
overlap; a positive that is the only heavily-reworded candidate is picked by the same
statistic with a minus sign, and the model then learns to prefer whatever is worded most
differently. The prompt therefore asks for PARITY -- positives and negatives spread over the
same range of wording distance -- so that overlap predicts nothing and the arithmetic is the
only thing left to read.

The report measures whether that held, as the **AUC of "does lexical overlap with the anchor
rank the positives above the negatives"**, scored on TOKENIZER IDS -- the things `act_emb`
actually averages, operators included. **Chance is 0.5 and a deviation in either direction is
the failure.** (Both halves of that were defects until 2026-08-08: the old check counted
one-sided hits and measured them on `normalise_step`, which deletes `* + - / =`.)

Selection draws ONLY from the train split, through the same `split_questions` call the
training data uses, at the same seed -- the val questions §9.2 calibrates tau on never
appear here.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import operator
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

from feynman_prm.config import load_config
from feynman_prm.data.branch_points import normalise_step
from feynman_prm.data.counterfactual import CounterfactualExample, write_jsonl
from feynman_prm.data.math_shepherd import build_questions, iter_hf_rows, split_questions

# Claude API list price, $ per 1M tokens. The Batch API bills at 50% of these.
PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
BATCH_DISCOUNT = 0.5
MAX_REQUESTS_PER_BATCH = 20_000     # API ceiling is 100k; smaller jobs fail smaller

# `error_kind` is NOT constrained. The model coins its own name when the natural error for a
# step is not on this list, which is intended -- these are the VOCABULARY `collect` scores its
# histogram against, split known vs coined, not a schema enum. Extended 2026-08-08 alongside
# the multi-positive L_CF.
LOCAL_ERROR_KINDS = [
    "operator_flipped",
    "operand_changed",
    "operands_swapped",
    "term_dropped",
    "off_by_one",
    "precedence_or_grouping",
    "wrong_quantity_from_context",
]
REASONING_ERROR_KINDS = [
    "wrong_method",
    "misread_problem",
    "invalid_manipulation",
    "unjustified_leap",
    "ignores_constraint",
    "off_target",
]
ERROR_KINDS = LOCAL_ERROR_KINDS + REASONING_ERROR_KINDS

# Token-Jaccard against the anchor at or above which a negative counts as a LOCAL EDIT for
# reporting. 0.70 is a threshold on a MEASURED distribution, not a guess: across cf_v3 and
# cf_nothink the per-item overlaps fall into two clumps -- one-token edits at 0.71-1.00 and
# rewordings at 0.12-0.59 -- with nothing between 0.59 and 0.71. Diagnostic only; nothing is
# accepted or dropped on it, because a threshold that gates would turn a measurement of the
# gap into a target the generator optimises against.
LOCAL_EDIT_OVERLAP = 0.70

_DEDUP_WHITESPACE = re.compile(r"\s+")


def dedup_key(step: str) -> str:
    """Identity key for deciding whether two rewrites are THE SAME REWRITE. Case and
    whitespace are folded; **the mathematics is left completely alone.**

    This exists because `normalise_step` was doing this job and cannot. It applies
    `[^\\w\\s]`, which deletes `+ - * / = ^ $ \\ { }`, and strips `<<expr=value>>` outright --
    so it erases exactly the characters a LOCAL EDIT consists of. MEASURED 2026-08-09 on
    `cf_fix_nothink` item cf000004, where three mathematically distinct negatives

        $x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}$      (kept)
        $x=\\frac{-b\\pm\\sqrt{b^2+4ac}}{2a}$      (dropped as a duplicate)
        $x=\\frac{ b\\pm\\sqrt{b^2-4ac}}{2a}$      (dropped as a duplicate)

    all normalise to `... x frac b pm sqrt b 2 4ac 2a`. The item then failed with
    `too_few_usable_negatives` -- a label blaming the model for output the validator had
    destroyed, which is the §14 failure: a guard that names the wrong diagnosis.

    The bias had a direction. A reasoning negative changes WORDS, which survive
    normalisation; a local edit changes OPERATORS, which do not. So the deduplicator was
    silently deleting local edits and sparing rewordings -- an anti-local-edit filter nobody
    wrote, working against the very quota `local_negative_quota` exists to enforce. It cost
    1 negative per run in thinking mode too (`cf_v3`, `cf_nothink`), so this is not a
    `--no-thinking` problem.

    Whitespace is removed rather than collapsed so that `$x = 5$` and `$x=5$`, and
    `\\frac{-b \\pm ...}` and `\\frac{-b\\pm ...}`, still read as one rewrite. `normalise_step`
    keeps its job of measuring lexical OVERLAP, which is what it was built for.

    Only TRAILING `.!?` are stripped, never interior ones: `eggs.` and `eggs!` are the same
    rewrite, but `.` is also a decimal point, and folding it everywhere would make `3.5` and
    `35` the same number -- reintroducing the bug this function exists to remove, one
    character class over.
    """
    return _DEDUP_WHITESPACE.sub("", step.lower()).rstrip(".!?")


# --------------------------------------------------------------------------- the prompt

SYSTEM_PROMPT = """\
You build training data for model that scores maths solutions one step at a
time. It must represent a step by what it DOES -- the quantities it uses, the operation, the
value it lands on -- and ignore how the step is worded.

Given a problem, its chain of steps, and one target step, rewrite that step many times.
`positive_rewrites` keep the mathematics and change the wording. `negative_rewrites` change the
mathematics or the reasoning.

PARITY -- the rule that decides whether the example is worth anything.
Wording must not predict correctness. Delete every number and symbol, read only the prose, and
you should not be able to tell the positives from the negatives. Write each positive as a
genuine reformulation -- re-describe a quantity in words, reorder clauses, swap prose for
equation, reach the same number by another route, change voice, name what a pronoun stood for
-- and make the positives differ from EACH OTHER too, not three variations on one sentence.
Then spread the negatives over the same range: one a small local edit to an existing sentence,
one a full reformulation sharing almost no wording with the original, the rest in between.
Neither group may be the odd one out.

NAMES ARE FIXED, and this overrides every reformulation instruction above. Any symbol,
variable, label or name introduced by the problem or by an earlier step belongs to the whole
chain, not to this step. Use it exactly as it stands: never rename it, never re-letter it,
never redefine it, never switch its units. If step [0] set $x$ to be the number of apples,
every rewrite of step [3] still calls it $x$ -- calling it $a$ makes the step unreadable in
place and breaks every step that follows. The same holds for function names, point and set
labels, and any quantity the problem itself named. You may introduce a new name ONLY if the
target step is where it first appears and no later step refers to it. Reformulate by every
other means and leave the vocabulary of the chain alone.

INTERNAL CONSISTENCY. A negative is wrong relative to the PROBLEM and the earlier steps, never
relative to itself: its prose, its expression and its <<...>> annotation must agree with one
another. "half of April's, so 48*(1/3)=<<48/3=16>>16" is caught with no context and is useless;
"a third of April's, so 48*(1/3)=<<48/3=16>>16" is coherent, and only the problem shows it
wrong. The best negatives cannot be caught from the step alone.

ERROR KINDS. Label each negative with a short snake_case name for the break it makes. These are
common ones, not a closed list -- if the natural error for this step is not among them, coin
your own name and use that:
  local edits   operator_flipped, operand_changed, operands_swapped, term_dropped, off_by_one,
                precedence_or_grouping, wrong_quantity_from_context (a real number lifted from
                the wrong earlier step -- the most valuable local kind)
  reasoning     wrong_method, misread_problem, invalid_manipulation, unjustified_leap,
                ignores_constraint, off_target (true, but not what this step was for)
Prefer the error this step actually invites over anything on the list. AT LEAST ONE negative
must be a reasoning failure rather than an arithmetic edit, whenever the step admits one --
mistakes a real solver makes are worth far more than typos.

CONSTRAINTS
- Each rewrite is a fluent, self-contained replacement, correct to read straight after the
  preceding steps and straight before the ones that follow.
- The `text` field holds ONLY the replacement step, exactly as it would appear in the chain.
  Never prefix it with the error kind, a label, an index or a marker: `error_kind` is its own
  field, and a kind name inside the text is a word that appears in every negative and no
  positive, which is precisely the shortcut described above.
- Recompute every <<expr=value>> annotation to match that rewrite's own arithmetic.
- Match the original's formatting: LaTeX stays LaTeX, prose stays prose.
- No hedging, no markers, no commentary; never mention that this is a rewrite.
- Do not touch the other steps.
- Positives must differ from one another in wording; negatives must differ from one another in
  what they compute or claim.

RESULTS. Report anchor_result, a result for every positive and a result for every negative: a
bare number where there is one, else the shortest canonical form. Checked automatically -- a
positive whose result differs from the anchor is discarded, as is a negative whose result
matches it. Write them all in the same format ("41" and "$41$" read as different).

A result is DERIVED, never copied. Every rewrite's result is the value that rewrite's own
mathematics produces -- carry its arithmetic through to the end and report what you actually
get. If you changed a multiplication to an addition, the result is the sum; if you changed an
operand, the result is what the new operand yields. The anchor's result is the answer to a
different question and is never the answer to this one. Before writing a negative's result,
ask what value that specific rewrite reaches: if it is the anchor's, the rewrite does not
break anything and must be replaced, not relabelled.

Every step has a result, so "no result", "none" and "n/a" are never valid. A step that
asserts a relationship instead of computing a number reports THE CLAIM as its result
("factors of $2000^2$", "x is even", "converges"), written so that a rewrite making a
different claim necessarily writes a different string. If a step introduces a formula, the
result is the formula. If it establishes a bound, the result is the bound. Report the
smallest thing that changes when the mathematics changes -- that is what makes the automatic
check able to tell your negatives from your anchor at all.

UNSUITABLE. Set unsuitable=true with a one-clause reason if the step has no mathematics to
preserve or break, or you cannot make at least two coherent positives and five coherent
negatives at parity. A refused item costs nothing; a shortcut-laden one damages the model.

EXAMPLE 1  (two positives and three negatives shown; produce the number asked for)
Problem: A bakery made 60 muffins Monday, 15 fewer Tuesday, and twice Tuesday's number on
Wednesday. How many on Wednesday?
Steps: [0] On Tuesday the bakery made 60 - 15 = <<60-15=45>>45 muffins.
       [1] On Wednesday it made 45 * 2 = <<45*2=90>>90 muffins.        <- target, result 90

anchor_result: 90
positive_rewrites:
  - text:   Wednesday's output was double Tuesday's, so the bakery turned out 2 * 45 = <<2*45=90>>90 muffins.
    result: 90
  - text:   Tuesday's 45 muffins were matched again the next day, putting Wednesday's total at 45 + 45 = <<45+45=90>>90.
    result: 90
negative_rewrites:
  - text:   Wednesday's output was double Monday's, so the bakery turned out 2 * 60 = <<2*60=120>>120 muffins.
    result: 120
    error_kind: wrong_quantity_from_context
  - text:   On Wednesday it made 45 + 2 = <<45+2=47>>47 muffins.
    result: 47
    error_kind: operator_flipped
  - text:   The Monday-to-Tuesday fall repeated itself, leaving the bakery 60 - 15 - 15 = <<60-15-15=30>>30 muffins to show for Wednesday.
    result: 30
    error_kind: misread_problem

Note that no `text` above begins with its error_kind, or with any label at all -- each one is
just the step. The two positives reach 90 by different routes and read nothing like each other.
The negatives run from a one-token edit of the original to a sentence sharing almost nothing
with it, so how a rewrite reads says nothing about whether it is correct. The first negative is
the shape to aim for: flawless arithmetic that only the problem statement refutes.

EXAMPLE 2  (algebra -- note that $x$ is never renamed in any rewrite)
Problem: Solve $x^2 = 5x$.
Steps: [0] Rewrite the equation as $x^2 - 5x = 0$.
       [1] Factoring gives $x(x-5) = 0$, so $x = 0$ or $x = 5$.   <- target, result 0 or 5

anchor_result: 0 or 5
positive_rewrites:
  - text:   Pulling out the shared factor leaves $x(x-5)=0$, and a product vanishes only when one of its factors does, so $x=0$ or $x=5$.
    result: 0 or 5
negative_rewrites:
  - text:   Dividing both sides through by $x$ leaves $x = 5$.
    result: 5
    error_kind: invalid_manipulation
  - text:   Pulling out the shared factor leaves $x(x+5)=0$, so $x=0$ or $x=-5$.
    result: 0 or -5
    error_kind: operand_changed

Rewriting this step as "letting $a$ denote the unknown, $a(a-5)=0$" would be worthless however
good the algebra is: step [0] already named it $x$, and the rewrite has to drop into that
chain unchanged.

ONE LAST CHECK BEFORE YOU ANSWER. Compare your negatives against your positives on wording
alone. If the negatives are the ones that kept the original's symbols and expressions while the
positives are the ones that reworded everything, you have made wording predict correctness --
the failure this whole task is built to avoid. Rewrite until at least one positive keeps the
original's notation closely and at least one negative departs from it entirely."""


# --------------------------------------------------- several anchors in one request (§7.5.9)

# The labels a batched request gives its anchors. Short, unambiguous, and nothing a rewrite
# would ever contain, so a label leaking into an answer is visible rather than plausible.
GROUP_LABELS = "ABCDEFGH"

# **APPENDED to SYSTEM_PROMPT, never a rewrite of it**, for two reasons that both matter.
# (a) The single-anchor prompt is what every measurement in §7.5.6-§7.5.9 was taken against;
#     a second copy would drift from it silently and every one of those numbers would quietly
#     stop applying. There is one statement of the task and this adds to it.
# (b) SYSTEM_PROMPT stays an exact PREFIX of this, so the endpoint's prefix cache still hits on
#     the ~2,300 tokens that dominate the input meter (§7.5.6).
_GROUP_ADDENDUM = """\

=====================================================================================
SEVERAL ANCHORS IN ONE REQUEST. EVERY INSTRUCTION ABOVE APPLIES TO EACH ONE SEPARATELY.

This request carries more than one INDEPENDENT task, labelled ANCHOR A, ANCHOR B and so on.
They are different target steps, in general from different problems by different authors, and
they are batched into one request to save request quota and for NO other reason. There is no
relationship between them, no comparison to draw, no shared context, and nothing whatsoever to
carry from one to the next.

Answer them as if they had arrived as separate requests:

- Read one anchor's problem, its chain of steps and its target step, and answer for that
  anchor alone. Then start again from nothing for the next one.
- NEVER let a quantity, a number, a variable, a name, a result or an error kind from one
  anchor appear in another anchor's rewrites. A number that occurs in A's problem and not in
  B's has no business anywhere in B's answer. A rewrite that reaches across is not a small
  mistake -- it is a wrong answer wearing the shape of a right one, and it is worse than
  refusing the item.
- Each anchor has its OWN anchor_result, derived from ITS OWN target step's mathematics. B's
  result is never A's, even in the common case where the two happen to be the same number.
- THE RESULTS RULE ABOVE IS PER ANCHOR, AND IT IS THE ONE MOST EASILY LOST WHEN TWO ANCHORS
  SHARE A REQUEST. Inside one anchor, anchor_result and every rewrite's result are written in
  ONE format. A rewrite whose mathematics is UNCHANGED repeats the anchor's result string
  character for character -- same words, same spacing, same LaTeX -- and does not reword it,
  retype it or tidy it. This bites hardest on a step that asserts a claim rather than computing
  a number: "second digit after the decimal point" and "second number past the decimal" are the
  same claim and DIFFERENT STRINGS, and the check that discards positives cannot tell them
  apart. Reword the STEP as much as you like; never reword its result.
- The counts are PER ANCHOR and are never shared or split between them. Each anchor needs the
  full number of positive_rewrites and the full number of negative_rewrites asked for, and its
  own local-edit quota within its own negatives.
- Judge `unsuitable` separately. One anchor being unsuitable says nothing at all about the
  other, and you must still answer the other in full.
- NAMES ARE FIXED continues to mean "fixed within THAT anchor's own chain". A's $x$ and B's $x$
  are unrelated symbols; do not reconcile them, and do not rename either to tell them apart.
- The final parity check is run separately for each anchor, against that anchor's own
  positives and negatives.

OUTPUT SHAPE. Return ONE json object whose only key is `anchors`: a list holding one object per
anchor, in the order the anchors were given.

  {"anchors": [{"anchor_id": "A", ...the fields described above, for anchor A...},
               {"anchor_id": "B", ...the fields described above, for anchor B...}]}

Each element carries exactly the fields specified earlier -- anchor_id, unsuitable,
unsuitable_reason, anchor_result, positive_rewrites, negative_rewrites -- and nothing else. Set
anchor_id to the label that anchor was given. Emit an element for EVERY anchor, including any
you mark unsuitable. Never merge two anchors' rewrites into one list, never answer for one
anchor and omit the other, and never emit more elements than there were anchors."""

SYSTEM_PROMPT_GROUPED = SYSTEM_PROMPT + "\n" + _GROUP_ADDENDUM


def response_schema() -> dict:
    """Structured-output schema. Array length is NOT expressible (`minItems` is unsupported),
    so the counts are asked for in prose and enforced in `validate`.

    `error_kind` is a PLAIN STRING, deliberately: pinning it to `ERROR_KINDS` would forbid
    the model from naming the error a step actually invites, which the system prompt asks it
    to do. `collect` scores coinage rather than preventing it.
    """
    rewrite_array = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "result": {"type": "string"},
            },
            "required": ["text", "result"],
            "additionalProperties": False,
        },
    }
    negative_array = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "result": {"type": "string"},
                "error_kind": {"type": "string"},
            },
            "required": ["text", "result", "error_kind"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": {
            "unsuitable": {"type": "boolean"},
            "unsuitable_reason": {"type": "string"},
            "anchor_result": {"type": "string"},
            "positive_rewrites": rewrite_array,
            "negative_rewrites": negative_array,
        },
        "required": [
            "unsuitable",
            "unsuitable_reason",
            "anchor_result",
            "positive_rewrites",
            "negative_rewrites",
        ],
        "additionalProperties": False,
    }


def grouped_response_schema() -> dict:
    """`response_schema()` wrapped in an `anchors` array, with an `anchor_id` on each element.

    Derived from the single-anchor schema rather than restated, so the two cannot drift: a new
    field added to one is in the other by construction. **`anchor_id` is REQUIRED and goes
    FIRST** -- a structured-output model fills the fields in schema order, so asking for the
    label before the content makes it name the anchor it is about to answer for rather than
    label the answer it has already written.

    Array length is not expressible here either (§ `response_schema`), so "one element per
    anchor, in order" is asked for in prose and enforced in `split_group_response`.
    """
    block = response_schema()
    return {
        "type": "object",
        "properties": {
            "anchors": {
                "type": "array",
                "items": {
                    **block,
                    "properties": {"anchor_id": {"type": "string"}, **block["properties"]},
                    "required": ["anchor_id", *block["required"]],
                },
            },
        },
        "required": ["anchors"],
        "additionalProperties": False,
    }


def group_user_message(group: Sequence[dict], n_positives: int, n_negatives: int) -> str:
    """The per-anchor message, repeated under a label, plus a closing separation reminder.

    It calls `user_message` per anchor rather than restating it, for the reason
    `_GROUP_ADDENDUM` is an addendum: one statement of the task, so a change to the
    single-anchor instructions cannot silently leave the batched ones behind.

    **A group of ONE still goes through here when batching is on.** The negotiated
    `response_format` is fixed for the whole session by the probe, so the odd anchor at the end
    of the file has to ask for the same shape as every other request -- a single-anchor prompt
    against a grouped schema is a 400, or worse a parse that finds no `anchors` key.
    """
    labels = GROUP_LABELS[: len(group)]
    blocks = "\n\n".join(
        f"===================== ANCHOR {label} =====================\n"
        f"{user_message(item, n_positives, n_negatives)}"
        for label, item in zip(labels, group)
    )
    listed = ", ".join(labels[:-1]) + f" and {labels[-1]}" if len(labels) > 1 else labels
    return (
        f"{blocks}\n\n"
        f"===================== ANSWER =====================\n"
        f"Answer for {'both' if len(group) == 2 else 'all'} {len(group)} anchors in ONE json "
        f"object: {{\"anchors\": [...]}}, holding exactly {len(group)} elements, one per "
        f"anchor, in the order {listed}, each tagged with its own anchor_id. "
        f"The anchors are unrelated problems that happen to share a request. Give each one the "
        f"full {n_positives} positive_rewrites and {n_negatives} negative_rewrites, each with "
        f"its own anchor_result derived from its own target step, and never let a number, a "
        f"variable, a name or a result from one anchor appear in another's rewrites."
    )


def local_negative_quota(n_negatives: int) -> int:
    """How many negatives MUST be a LOCAL EDIT -- the original sentence with a digit or an
    operator changed. **2 of 6, set by the human on 2026-08-08** (§7.5.2).

    **Two-sided since 2026-08-09.** The prompt used to say "EXACTLY n may be", which reads as
    a ceiling with no floor, and the generator obeyed the ceiling only: `cf_nothink` came back
    at 22% local against the 33% asked for, with two items having NO negative above 0.70
    overlap at all. A missing local edit is not a neutral omission -- it removes the one
    negative that sits on top of the positives in wording, so what is left is the reworded
    ones and "most reworded = wrong" starts predicting again. The floor and the ceiling defend
    against opposite failures and the prompt now states both.

    Why there is a quota at all: a local edit keeps every other token, so it is the negative
    that scores HIGHEST on overlap with the anchor, while a positive is a genuine reword and
    scores lowest. Six of six local edits is what produced the 0.45 / 0.58 split and AUC 0.218
    (§7.5.6) -- the dataset was solvable by "pick the most-reworded candidate" with no
    mathematics at all. Capping them at a third holds the overlap distributions on top of each
    other, and the remaining two thirds are the errors a PRM actually has to catch: a wrong
    method, a misread problem, an unjustified leap. Those are harder to write and worth more.
    """
    return max(1, round(n_negatives / 3))


def user_message(item: dict, n_positives: int, n_negatives: int) -> str:
    numbered = "\n".join(
        f"{'>>' if i == item['step_index'] else '  '} [{i}] {step}"
        for i, step in enumerate(item["steps"])
    )
    n_local = local_negative_quota(n_negatives)
    return (
        f"Problem:\n{item['question']}\n\n"
        f"Steps (the target is marked >>):\n{numbered}\n\n"
        f"Target step [{item['step_index']}]:\n{item['steps'][item['step_index']]}\n\n"
        f"Write {n_positives} positive_rewrites and {n_negatives} negative_rewrites, each "
        f"negative with a distinct error_kind. EXACTLY {n_local} of the {n_negatives} "
        f"negatives MUST be a local edit -- not at most {n_local}, not zero, exactly "
        f"{n_local}. A local edit reproduces the target sentence word for word and changes "
        f"ONE number, operator or operand, leaving every other word, symbol and piece of "
        f"punctuation exactly where it was; read beside the original it should look almost "
        f"identical, and the overlap in wording should be near-total. Do not reword it, do "
        f"not shorten it, do not improve it. Those {n_local} must break DIFFERENT things "
        f"from each other -- {n_local} edits of the same quantity in the same place are one "
        f"negative written {n_local} times, however they are labelled. "
        f"The remaining {n_negatives - n_local} are FORBIDDEN to be local edits: none of "
        f"them may reuse the target sentence's phrasing or sentence shape, and if one of "
        f"them could be produced by changing a few characters of the original, it is wrong "
        f"and must be rewritten from scratch. "
        f"The other {n_negatives - n_local} must be REASONING failures: a "
        f"wrong method, a misread of the problem, a quantity taken from the wrong place, an "
        f"unjustified leap, a constraint ignored. Note that the reasoning failures include but are not limited to these failure types. Write those the way a student who "
        f"misunderstood would write them -- in their own words, not as an edit of the original sentence. Think like a teacher, what kind of mistakes would a normal student make here?. "
        f"Make the positives differ from each other in wording, and vary the "
        f"negatives -- local ones close to an existing sentence, reasoning ones sharing almost nothing with the "
        f"original. Never rename a variable or label introduced by the problem or an earlier "
        f"step. Report anchor_result and a result for every rewrite."
    )


# ------------------------------------------------------------------- sample (no API calls)

# A step ASSERTS something if it states a relation, or if it is a finished sentence. Either
# is enough on its own: `100-30=<<100-30=70>>70 are in Grade 5` ends on a digit and asserts a
# relation; `must be factors of $2000^2$.` has no relation symbol and is a finished claim.
_ASSERTS_RELATION = re.compile(r"[=<>]|\\le|\\ge|\\neq|\\approx|\\equiv")
_ENDS_A_SENTENCE = re.compile(r"[.!?:]$|\]$|\$$")


def step_asserts_something(step: str) -> bool:
    """Whether a step makes a claim a counterfactual could contradict.

    **A step that announces an intention and stops has nothing to break.** MEASURED
    2026-08-09 on `cf_items` cf000002, whose target step is the fragment

        Step 1: We multiply $-5x^3 - 5x^2 - 7x + 1$ by $-x^2 - 6x + 1$ to get

    -- the product itself lives in the NEXT step. It passed the digit-and-length filter, cost
    four full generations, and failed three of them: with nothing in the step to make wrong,
    the model returned the anchor VERBATIM as a negative (twice in the last run) and reached
    down to the final step for an `anchor_result` of `36`, which every negative then copied.
    The prompt already tells the model to refuse these (`unsuitable=true` when the step has no
    mathematics to preserve or break) and it did not, in 3 of 4 runs. Cheaper to never send it.

    Measured on 14,725 eligible steps from 2,824 real trajectories: this excludes **1.51%**
    (223), and the excluded set is `By Cauchy-Schwarz,` / `Expanding, we get` / `which
    simplifies to` / `From the first equation, we can write` -- dangling connectives, plus
    some `Sum of the roots: 12` label-value lines that have no mathematics to break either.

    **The errors are deliberately asymmetric.** A few complete-but-unpunctuated assertions are
    caught with them. That is the cheap direction: a false exclusion costs one candidate anchor
    out of a pool of tens of thousands, while a false inclusion costs a whole generation and
    yields an item that cannot work. Nothing downstream re-checks this, so it errs out.
    """
    text = step.strip()
    return bool(_ASSERTS_RELATION.search(text)) or bool(_ENDS_A_SENTENCE.search(text))


def eligible_step_indices(steps: tuple[str, ...], include_final: bool) -> list[int]:
    """A step is worth rewriting only if it computes something. Steps with no digit have no
    arithmetic to preserve or break, and the final step is usually a bare answer restatement
    whose only counterfactual is 'change the number' -- lexically trivial on both sides.
    Steps that assert nothing are excluded too -- see `step_asserts_something`."""
    last = len(steps) if include_final else len(steps) - 1
    return [
        i for i in range(max(last, 0))
        if len(steps[i].strip()) >= 20
        and any(ch.isdigit() for ch in steps[i])
        and step_asserts_something(steps[i])
    ]


# Relative weight at the two ENDS of the eligible list against its middle. 1/6 gives a ~5:1
# tilt at 6 eligible steps -- the ends stay reachable, and rare.
EDGE_STEP_WEIGHT = 1.0 / 6.0


def middle_step_weights(n_choices: int):
    """Sampling weight over the ELIGIBLE steps: a tent peaked on the middle of the solution.

    **Set by the human on 2026-08-10, and BOTH ends are the point.** A solution's opening
    steps are setup -- *"assume x is the number of apples"*, *"we are given that..."* -- and
    its closing steps are restatement -- *"therefore the answer is 12"*, *"hence x = 4"*.
    Both pass the digit-and-length-and-asserts filter, and neither carries arithmetic worth
    breaking: the front has nothing derived yet, the back only repeats what the step before
    already derived. **The work is in the middle.**

    Note `eligible_step_indices` already drops the final step outright (`include_final`), and
    that is NOT enough -- a "therefore" conclusion routinely occupies the last two or three
    steps, so the last eligible one is still often a restatement.

    **A weight, not a cutoff, and that is the whole design.** At 6 eligible steps the two
    middle steps take 27.8% each and the two ends 5.6% each, so both ends still appear and
    appear *rarely*. A hard cutoff would be one more filter with a threshold nobody has
    measured, and §7.5.6 records what those cost; a weight cannot fail the way a threshold
    can, because every step stays reachable and the tilt degrades smoothly.

    It self-limits where it does not matter: picking 4 of 5 eligible steps takes 4 of them
    whatever the weights say. It bites only on solutions with many eligible steps, which is
    exactly where there are setup and restatement steps to skip.
    """
    import numpy as np

    if n_choices < 3:
        # Two steps have no middle to peak on, and one has nothing to choose. Uniform, rather
        # than a formula that would divide by a zero half-width.
        weights = np.ones(n_choices, dtype=float)
    else:
        centre = (n_choices - 1) / 2
        distance = np.abs(np.arange(n_choices) - centre) / centre   # 0 mid, 1 at both ends
        weights = 1.0 - (1.0 - EDGE_STEP_WEIGHT) * distance
    return weights / weights.sum()


def _tap_first_rows(rows, max_rows: int, sink: set[str]):
    """Record the qids of the first `max_rows` rows while the stream passes through.

    The restriction has to be applied AFTER `split_questions`, never before it: the val
    holdout is a permutation of the trainable pool, so shrinking that pool would reshuffle it
    and questions §9.2 calibrates tau on could land in the CF training data. One pass, one
    set, and the split stays byte-for-byte the run's own.
    """
    from feynman_prm.data.math_shepherd import question_id

    for i, row in enumerate(rows):
        if max_rows and i < max_rows:
            sink.add(question_id(row["prompt"]))
        yield row


def cmd_sample(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.set)
    import numpy as np

    first_rows: set[str] = set()
    questions, _ = build_questions(_tap_first_rows(iter_hf_rows(), args.max_rows, first_rows))
    train, _val = split_questions(
        questions, cfg.data.n_val_questions, cfg.data.n_questions, cfg.run.seed
    )
    print(f"[sample] train pool {len(train)} questions (val held out first, seed {cfg.run.seed})")
    if args.max_rows:
        train = [q for q in train if q.qid in first_rows]
        print(f"[sample] restricted to the first {args.max_rows:,} train rows: "
              f"{len(first_rows):,} distinct questions there, {len(train)} of them in the "
              f"train pool")

    rng = np.random.default_rng(args.seed)
    items, skipped = [], 0
    for question in train:
        # A meaning-PRESERVING rewrite of a step that is already wrong is a muddled label, so
        # rewrite steps from fully-correct trajectories only.
        correct = [t for t in question.trajectories if t.correct]
        if not correct:
            skipped += 1
            continue
        traj = correct[int(rng.integers(len(correct)))]
        choices = eligible_step_indices(traj.steps, args.include_final_step)
        if not choices:
            skipped += 1
            continue
        picked = rng.choice(
            choices,
            size=min(args.per_question, len(choices)),
            replace=False,
            p=middle_step_weights(len(choices)),
        )
        for step_index in sorted(int(s) for s in picked):
            items.append(
                {
                    "custom_id": f"cf{len(items):06d}",
                    "qid": question.qid,
                    "question": question.prompt,
                    "steps": list(traj.steps),
                    "step_index": step_index,
                }
            )
        if args.limit and len(items) >= args.limit:
            break

    items = items[: args.limit] if args.limit else items
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")
    print(f"[sample] wrote {len(items)} items to {out} ({skipped} questions had no usable step)")
    return 0


# ------------------------------------------------------------------------------- submit

def read_items(path: str | Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_params(item: dict, args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        # The system prompt is identical across every request in the job, so one breakpoint
        # here makes all but the first read from cache.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "output_config": {
            "effort": args.effort,
            "format": {"type": "json_schema", "schema": response_schema()},
        },
        "messages": [
            {"role": "user", "content": user_message(item, args.positives, args.negatives)}
        ],
    }


def estimate_cost(client, items: list[dict], args: argparse.Namespace) -> None:
    """Input side is MEASURED with count_tokens on a sample; output side is an ASSUMPTION
    (`--assumed-output-tokens`) because nothing has been generated yet. Both are labelled."""
    sample = items[: min(3, len(items))]
    counted = []
    for item in sample:
        params = build_params(item, args)
        counted.append(
            client.messages.count_tokens(
                model=args.model,
                system=params["system"],
                messages=params["messages"],
            ).input_tokens
        )
    mean_in = sum(counted) / len(counted)
    price_in, price_out = PRICES.get(args.model, (None, None))
    print(f"[submit] measured input {mean_in:,.0f} tokens/request (n={len(sample)}, count_tokens)")
    if price_in is None:
        print(f"[submit] no cached price for {args.model}; skipping the estimate")
        return
    n = len(items)
    cost_in = n * mean_in / 1e6 * price_in * BATCH_DISCOUNT
    cost_out = n * args.assumed_output_tokens / 1e6 * price_out * BATCH_DISCOUNT
    print(
        f"[submit] ESTIMATE for {n:,} requests at batch pricing: "
        f"${cost_in:,.2f} in + ${cost_out:,.2f} out = ${cost_in + cost_out:,.2f}\n"
        f"         (output assumes {args.assumed_output_tokens} tokens/request and ignores "
        f"prompt-cache savings on the shared system prompt)"
    )


def cmd_submit(args: argparse.Namespace) -> int:
    import anthropic

    items = read_items(args.items)
    if not items:
        print("[submit] no items", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    estimate_cost(client, items, args)
    if not args.yes:
        print("[submit] dry run. Re-run with --yes to actually submit.")
        return 0

    batch_ids = []
    for start in range(0, len(items), MAX_REQUESTS_PER_BATCH):
        chunk = items[start : start + MAX_REQUESTS_PER_BATCH]
        batch = client.messages.batches.create(
            requests=[
                {"custom_id": item["custom_id"], "params": build_params(item, args)}
                for item in chunk
            ]
        )
        batch_ids.append(batch.id)
        print(f"[submit] batch {batch.id} -- {len(chunk)} requests, {batch.processing_status}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "batch_ids": batch_ids,
                "items": args.items,
                "model": args.model,
                "effort": args.effort,
                "positives": args.positives,
                "negatives": args.negatives,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[submit] wrote {out}; run `collect` when the batches end")
    return 0


# ------------------------------------------------------------------------------ validate

_CALC = re.compile(r"<<([^>]*)>>")
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _eval_arith(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression, or return None. `ast.literal_eval` cannot do
    operators and `eval` would run whatever the model wrote, so walk the tree by hand."""

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("unsupported")

    try:
        return walk(ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return None


def calculator_annotations_consistent(text: str) -> bool:
    """`<<3*4=12>>` must actually evaluate to 12. Unparseable annotations pass -- this is a
    check for contradictions, not a parser for every form Math-Shepherd contains."""
    for body in _CALC.findall(text):
        if "=" not in body:
            continue
        expr, _, claimed = body.rpartition("=")
        got, want = _eval_arith(expr), _eval_arith(claimed)
        if got is None or want is None:
            continue
        if abs(got - want) > 1e-6:
            return False
    return True


def scalar_text(value) -> str:
    """Coerce a JSON scalar to the string the checks assume.

    `response_schema()` declares `result` as a string and a strict json_schema endpoint
    honours it, so every run before 2026-08-09 saw strings. An endpoint that does NOT support
    json_schema -- Groq's llama-3.1-8b-instant is one, it falls back to `json_object` -- is
    free to answer `"result": 36`, and `_as_number` then raised
    `AttributeError: 'int' object has no attribute 'strip'` and took the whole run down at
    validation, AFTER the tokens were spent. Numbers and booleans become their JSON text;
    `None` becomes "". A guard that only works when the schema is enforced is not a guard.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    return str(value)


def _as_number(value) -> float | None:
    cleaned = re.sub(r"[,$%\s]", "", scalar_text(value).strip().rstrip("."))
    try:
        return float(cleaned)
    except ValueError:
        return None


def results_equal(a, b) -> bool:
    a, b = scalar_text(a), scalar_text(b)
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-9 * max(1.0, abs(na))
    return (a or "").strip().lower() == (b or "").strip().lower()


def jaccard(a: str, b: str) -> float:
    """Bag-of-words overlap on normalised tokens -- the same normalisation §4.4 used, which
    drops calculator annotations and punctuation.

    **This is NOT the view `act_emb` has.** `normalise_step` deletes `* + - / =` before the
    overlap is measured (`data/branch_points.py:36-41`), while `act_emb` is a mean over QWEN
    TOKENIZER tokens, which include every one of them. Kept as the human-readable number and
    as the gate on `--max-positive-overlap`; `token_jaccard` is what the shortcut statistic
    is scored on.
    """
    sa, sb = set(normalise_step(a).split()), set(normalise_step(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_jaccard(tokenizer, a: str, b: str) -> float:
    """Bag-of-TOKEN-IDS overlap, on the same tokenizer whose input embeddings `act_emb`
    averages (§6.4). Operators and digits survive here, which is the whole point."""
    sa = set(tokenizer(a, add_special_tokens=False)["input_ids"])
    sb = set(tokenizer(b, add_special_tokens=False)["input_ids"])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# A leading `snake_case_label` followed by whitespace, or `[label]` / `label:` -- the shapes a
# model reaches for when it copies a field into the text it belongs beside.
_LEADING_LABEL = re.compile(
    r"^\s*(?:\[\s*)?(?:error_kind\s*[:=]\s*)?"
    r"(?P<label>[a-z][a-z0-9]*(?:_[a-z0-9]+)+)"
    r"(?:\s*\])?\s*[:.\-]?\s+"
)


def strip_leading_label(text: str, error_kind: str = "") -> tuple[str, bool]:
    """Remove a `wrong_quantity_from_context   The bakery...` prefix, if the model wrote one.

    **MEASURED on the first live run (2026-08-08): every negative came back with its
    `error_kind` prefixed to its `text`**, because the system prompt's worked examples showed
    the kind in a column beside the sentence and the model reproduced the layout. The prompt
    now uses explicit field names and forbids the prefix; this is the belt to that braces,
    because the failure it causes is the worst one available here -- a token that appears in
    every negative and no positive is a PERFECT lexical shortcut straight into `act_emb`
    (§6.4), and `L_CF` would be solved on it without ever reading the arithmetic.

    Conservative by construction: it fires only on a `snake_case` token, which mathematical
    prose does not begin with, and only when the remainder is non-empty.
    """
    match = _LEADING_LABEL.match(text)
    if not match:
        return text, False
    label = match.group("label")
    # Either it IS the declared kind, or it is snake_case with an underscore -- both of which
    # a step of a maths solution does not open with.
    if error_kind and label != error_kind and "_" not in label:
        return text, False
    remainder = text[match.end():].strip()
    return (remainder, True) if remainder else (text, False)


@dataclass
class Validated:
    example: CounterfactualExample | None
    reason: str
    positive_overlaps: tuple[float, ...] = ()
    negative_overlaps: tuple[float, ...] = ()
    error_kinds: tuple[str, ...] = ()
    dropped_positives: int = 0
    dropped_negatives: int = 0
    stripped_labels: int = 0
    # Every rewrite this validator threw away, with the check that threw it and the text as
    # the model wrote it. **Kept since 2026-08-09 at the human's request: "save all the bad
    # negatives too, we will decide to drop or use them later."**
    #
    # It is not sentiment. `dropped_negatives` was a COUNT, and a count cannot tell an anchor
    # copy from a rounding rejection from a genuine near-duplicate -- three drops with three
    # different remedies, one of which (`<<100/30=3.33>>` against a true 3.3333...) is
    # arguably not a defect in the rewrite at all. Every diagnosis in §7.5.6 that led anywhere
    # came from reading the dropped text, and until now that meant re-deriving it by hand from
    # `.responses.jsonl`. This writes it down instead.
    #
    # It is a SIDECAR (`<out>.discarded.jsonl`), never the dataset: `write_jsonl` still emits
    # only what passed, so nothing here can reach `L_CF` by accident. Promoting any of it is a
    # deliberate act with its own script.
    discards: tuple[dict, ...] = ()
    # Which anchor this verdict belongs to. Empty on the hand-built test fixtures, which is
    # why it is last and defaulted -- but on a 70k run a reject row that does not name its
    # item cannot be looked up in `.responses.jsonl`, and that is the only reason to keep it.
    custom_id: str = ""


def validate(item: dict, payload: dict, args: argparse.Namespace) -> Validated:
    """Every positive gets the checks the single one used to get, plus mutual distinctness.

    A positive that fails is DROPPED, not fatal -- the example survives on the rest, and it
    is rejected only if fewer than `--min-positives` (or `--min-negatives`) remain. That is
    the difference from the single-positive version, where one bad rewrite killed the item.
    """
    anchor = item["steps"][item["step_index"]]
    anchor_key = dedup_key(anchor)
    anchor_result = scalar_text(payload.get("anchor_result", ""))
    cid = item.get("custom_id", "")
    if payload.get("unsuitable"):
        return Validated(None, "model_marked_unsuitable", custom_id=cid)

    discards: list[dict] = []

    def discard(kind: str, text: str, reason: str, raw: dict | object, label: str = "") -> None:
        """Record a thrown-away rewrite. `reason` names THE CHECK, not the outcome -- an
        `anchor_copy` and a `duplicate_of_a_kept_rewrite` are both "seen before" to the code
        and want completely different fixes from a reader."""
        discards.append(
            {
                "custom_id": cid,
                "kind": kind,
                "reason": reason,
                "text": text,
                "error_kind": label,
                "result": scalar_text(raw.get("result", "")) if isinstance(raw, dict) else "",
                "anchor_result": anchor_result,
            }
        )

    stripped = 0
    raw_positives = payload.get("positive_rewrites") or []
    positives, pos_overlaps, seen = [], [], {anchor_key}
    for positive in raw_positives:
        text = scalar_text(positive.get("text")).strip() if isinstance(positive, dict) else ""
        text, was_stripped = strip_leading_label(text)
        stripped += was_stripped
        if not text:
            discard("positive", text, "empty", positive)
            continue
        key = dedup_key(text)
        # A copy of the anchor, or of a positive already kept: a duplicate adds a zero-distance
        # pair to the class and nothing else.
        if key in seen:
            discard("positive", text,
                    "anchor_copy" if key == anchor_key else "duplicate", positive)
            continue
        if not results_equal(positive.get("result", ""), anchor_result):
            discard("positive", text, "result_differs_from_anchor", positive)
            continue
        if not calculator_annotations_consistent(text):
            discard("positive", text, "calculator_annotation_disagrees", positive)
            continue
        overlap = jaccard(anchor, text)
        if overlap >= args.max_positive_overlap:
            discard("positive", text, "too_close_to_anchor", positive)
            continue
        seen.add(key)
        positives.append(text)
        pos_overlaps.append(overlap)

    if len(positives) < args.min_positives:
        return Validated(
            None,
            "too_few_usable_positives",
            dropped_positives=len(raw_positives) - len(positives),
            discards=tuple(discards),
            custom_id=cid,
        )

    raw_negatives = payload.get("negative_rewrites") or []
    negatives, neg_overlaps, kinds = [], [], []
    for negative in raw_negatives:
        text = scalar_text(negative.get("text")).strip() if isinstance(negative, dict) else ""
        kind = scalar_text(negative.get("error_kind")).strip() if isinstance(negative, dict) else ""
        text, was_stripped = strip_leading_label(text, kind)
        stripped += was_stripped
        key = dedup_key(text)
        if not text:
            discard("negative", text, "empty", negative, kind)
            continue
        if key in seen:
            discard("negative", text,
                    "anchor_copy" if key == anchor_key else "duplicate", negative, kind)
            continue
        if results_equal(negative.get("result", ""), anchor_result):
            discard("negative", text, "result_equals_anchor", negative, kind)
            continue
        if not calculator_annotations_consistent(text):
            discard("negative", text, "calculator_annotation_disagrees", negative, kind)
            continue
        seen.add(key)
        negatives.append(text)
        neg_overlaps.append(jaccard(anchor, text))
        kinds.append(kind or "unlabelled")

    if len(negatives) < args.min_negatives:
        return Validated(
            None,
            "too_few_usable_negatives",
            dropped_positives=len(raw_positives) - len(positives),
            dropped_negatives=len(raw_negatives) - len(negatives),
            discards=tuple(discards),
            custom_id=cid,
        )

    return Validated(
        CounterfactualExample(
            question=item["question"],
            steps=tuple(item["steps"]),
            step_index=item["step_index"],
            positive_rewrites=tuple(positives),
            negative_rewrites=tuple(negatives),
        ),
        "ok",
        tuple(pos_overlaps),
        tuple(neg_overlaps),
        tuple(kinds),
        len(raw_positives) - len(positives),
        len(raw_negatives) - len(negatives),
        stripped,
        tuple(discards),
        cid,
    )


# --------------------------------------------------------------------- reporting the batch
#
# Three fixes to what this used to print, all 2026-08-08.
#
# (a) THE SHORTCUT CHECK IS TWO-SIDED. It used to count `positive_overlap > max(negative
#     overlaps)` and warn only when that ran high. **Near-zero is equally a failure**: it
#     means the positives are reliably the lexical OUTLIERS and L_CF is solved by "pick the
#     most-reworded candidate", with a minus sign instead of a plus. The statistic is now the
#     AUC of "does lexical overlap rank the positives above the negatives", chance 0.5, and
#     a deviation in EITHER direction is flagged.
# (b) IT IS SCORED ON TOKENIZER IDS. `jaccard` runs on `normalise_step`, which strips
#     punctuation (`data/branch_points.py:36-41`), so `* + - / =` are deleted before the
#     overlap is measured -- while `act_emb` is a mean over Qwen tokens, which include all of
#     them. The normalised number is kept alongside as the human-readable one.
# (c) ERROR KINDS ARE REPORTED, NOT GATED. Known vocabulary vs coined, and the
#     local-vs-reasoning split. **Gating on a rate nobody has measured is how a paid batch
#     gets thrown away** -- the first run is the measurement.


class _FallbackTokenizer:
    """NOT the Qwen tokenizer, and labelled as such everywhere it is used.

    A regex split that KEEPS operators and splits digits individually, which is much closer to
    what `act_emb` averages than `normalise_step` is -- the latter deletes every operator. Used
    only when `transformers` is unavailable; install it and pass `--tokenizer` for the real
    measurement.
    """

    name = "fallback-regex (NOT Qwen -- install transformers for the real statistic)"
    _TOKEN = re.compile(r"\d|[A-Za-z]+|[^\sA-Za-z0-9]")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self._TOKEN.findall(text or "")}


def load_act_emb_tokenizer(name: str | None):
    """Return (tokenizer, description). Never raises: a missing tokenizer degrades the
    statistic, it does not lose the dataset."""
    if not name:
        return _FallbackTokenizer(), _FallbackTokenizer.name
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name), name
    except Exception as exc:  # noqa: BLE001 -- any import/download/config failure degrades
        print(
            f"[report] could not load tokenizer {name!r} ({type(exc).__name__}: {exc}); "
            f"falling back to a regex split",
            file=sys.stderr,
        )
        return _FallbackTokenizer(), _FallbackTokenizer.name


def rank_auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """P(overlap of a random positive > overlap of a random negative), ties at 0.5.

    Chance is 0.5 at ANY P and N, which is why this replaced a hit-rate whose chance level
    moved with the counts.
    """
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(positives) * len(negatives))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class DatasetReport:
    n: int
    text: str
    token_auc: float | None
    normalised_auc: float | None


def build_report(
    verdicts: Sequence[Validated],
    reasons: Counter,
    tokenizer,
    tokenizer_name: str,
    args: argparse.Namespace,
) -> DatasetReport:
    kept = [v for v in verdicts if v.example is not None]
    n = len(kept)
    lines = [f"[report] outcomes: {dict(reasons.most_common())}"]
    if n == 0:
        return DatasetReport(0, "\n".join(lines), None, None)

    token_aucs, norm_aucs = [], []
    tok_pos, tok_neg, norm_pos, norm_neg = [], [], [], []
    # MEASURED local edits, per item. The `local` count further down classifies by error_kind
    # NAME, which is what the model called the break -- an `operator_flipped` label on a full
    # reword counts there and a genuine one-token edit labelled `misread_problem` does not. So
    # that number cannot answer "are the local edits actually present?". This one can: it is
    # the overlap the shortcut statistic is scored on, thresholded. Reported per item because
    # the quota is per item -- a mean would let one 6/6 item hide five 0/6 items.
    items_meeting_local_quota, local_by_overlap = 0, []
    for verdict in kept:
        ex = verdict.example
        anchor = ex.steps[ex.step_index]
        p_tok = [token_jaccard(tokenizer, anchor, t) for t in ex.positive_rewrites]
        n_tok = [token_jaccard(tokenizer, anchor, t) for t in ex.negative_rewrites]
        n_high = sum(1 for t in n_tok if t >= LOCAL_EDIT_OVERLAP)
        local_by_overlap.append(n_high)
        items_meeting_local_quota += n_high >= local_negative_quota(args.negatives)
        tok_pos.extend(p_tok)
        tok_neg.extend(n_tok)
        norm_pos.extend(verdict.positive_overlaps)
        norm_neg.extend(verdict.negative_overlaps)
        auc = rank_auc(p_tok, n_tok)
        if auc is not None:
            token_aucs.append(auc)
        auc_n = rank_auc(verdict.positive_overlaps, verdict.negative_overlaps)
        if auc_n is not None:
            norm_aucs.append(auc_n)

    token_auc = _mean(token_aucs) if token_aucs else None
    norm_auc = _mean(norm_aucs) if norm_aucs else None

    mean_pos = _mean([len(v.example.positive_rewrites) for v in kept])
    mean_neg = _mean([len(v.example.negative_rewrites) for v in kept])

    kinds = Counter(k for v in kept for k in v.error_kinds)
    known = Counter({k: c for k, c in kinds.items() if k in ERROR_KINDS})
    coined = Counter({k: c for k, c in kinds.items() if k not in ERROR_KINDS})
    total_kinds = sum(kinds.values()) or 1
    local = sum(c for k, c in kinds.items() if k in LOCAL_ERROR_KINDS)
    reasoning = sum(c for k, c in kinds.items() if k in REASONING_ERROR_KINDS)
    with_reasoning = sum(
        1 for v in kept if any(k in REASONING_ERROR_KINDS for k in v.error_kinds)
    )

    lines += [
        "",
        f"[report] kept {n} examples -- {mean_pos:.2f} positives and {mean_neg:.2f} negatives "
        f"each (requested {args.positives}/{args.negatives}, floors "
        f"{args.min_positives}/{args.min_negatives})",
        f"[report] dropped by validation: {sum(v.dropped_positives for v in kept)} positives, "
        f"{sum(v.dropped_negatives for v in kept)} negatives, from kept examples",
    ]
    stripped = sum(v.stripped_labels for v in verdicts)
    if stripped:
        lines.append(
            f"[report] *** stripped a leading label from {stripped} rewrites. The model is "
            f"writing `error_kind` INTO `text`, which is a token in every negative and no "
            f"positive -- a perfect shortcut into act_emb. The strip saved this run; fix the "
            f"prompt before scaling it. ***"
        )
    lines += [
        "",
        "[report] LEXICAL SHORTCUT -- does bag-of-words overlap with the anchor rank the",
        "         positives above the negatives?  CHANCE IS 0.5 AND BOTH DIRECTIONS FAIL:",
        "         >0.5 means the positive is picked by looking most like the anchor;",
        "         <0.5 means it is picked by looking least like it. Either way L_CF is",
        "         solved on spelling, which is the thing this dataset exists to avoid.",
        f"           token-id AUC   {token_auc:.3f}   <- the statistic. tokenizer: {tokenizer_name}"
        if token_auc is not None
        else "           token-id AUC   n/a",
        f"           normalised AUC {norm_auc:.3f}   (normalise_step: punctuation and "
        f"operators stripped -- readable, not what act_emb sees)"
        if norm_auc is not None
        else "           normalised AUC n/a",
        f"           mean token overlap  positive {_mean(tok_pos):.3f}   "
        f"negative {_mean(tok_neg):.3f}",
        f"           mean normalised     positive {_mean(norm_pos):.3f}   "
        f"negative {_mean(norm_neg):.3f}",
    ]
    if token_auc is not None:
        deviation = abs(token_auc - 0.5)
        if deviation > args.max_auc_deviation:
            side = "ABOVE" if token_auc > 0.5 else "BELOW"
            lines.append(
                f"           *** WARNING: {deviation:.3f} {side} chance, over the "
                f"{args.max_auc_deviation:.2f} flag. Wording predicts correctness. ***"
            )
        else:
            lines.append(
                f"           within {args.max_auc_deviation:.2f} of chance -- parity held."
            )

    lines += [
        "",
        "[report] ERROR KINDS -- reported, NOT gated. No rate here has ever been measured, so",
        "         there is no threshold to gate on; this run is the measurement.",
        f"           known vocabulary {sum(known.values())}/{total_kinds} "
        f"({sum(known.values()) / total_kinds:.1%})   coined {sum(coined.values())} "
        f"({sum(coined.values()) / total_kinds:.1%})",
        f"           local edits {local} ({local / total_kinds:.1%})   "
        f"reasoning failures {reasoning} ({reasoning / total_kinds:.1%})   "
        f"unclassified {total_kinds - local - reasoning}",
        # The quota is the lever on the AUC above: a local edit keeps the anchor's whole
        # sentence and is therefore the highest-overlap negative there is (§7.5.2). Reported
        # against what was asked for, so the two numbers can be read together.
        f"           asked for {local_negative_quota(args.negatives)}/{args.negatives} local "
        f"({local_negative_quota(args.negatives) / args.negatives:.0%})   "
        f"got {local / total_kinds:.0%}   "
        f"-- this is the lever on the AUC above; local edits sit closest to the anchor",
        # The label count above answers "what did the model CALL its errors". This answers
        # "did it actually write the local edits", which is a different question and the one
        # the quota is about -- see LOCAL_EDIT_OVERLAP.
        f"           local edits BY MEASURED OVERLAP (token jaccard >= {LOCAL_EDIT_OVERLAP:.2f} "
        f"vs anchor): {sum(local_by_overlap)}/{sum(len(v.example.negative_rewrites) for v in kept)}"
        f"   per item {local_by_overlap}",
        f"           examples meeting the {local_negative_quota(args.negatives)}/"
        f"{args.negatives} local quota: {items_meeting_local_quota}/{n} "
        f"({items_meeting_local_quota / n:.0%})   (prompt asks for exactly "
        f"{local_negative_quota(args.negatives)} in EVERY item)",
        f"           examples with >=1 reasoning negative: {with_reasoning}/{n} "
        f"({with_reasoning / n:.1%})   (the prompt asks for a majority)",
        f"           known:  {dict(known.most_common())}",
        f"           coined: {dict(coined.most_common())}",
    ]
    return DatasetReport(n, "\n".join(lines), token_auc, norm_auc)


MARKDOWN_EXAMPLES = 60


def write_markdown(verdicts: Sequence[Validated], report: DatasetReport, path: Path) -> None:
    """A readable companion to the JSONL. The JSONL is what the loader reads; this is what a
    human reads to decide whether the generation is any good.

    **Capped at `MARKDOWN_EXAMPLES`.** Nobody reads the 4,000th example, and at 70k anchors an
    uncapped file is ~200 MB that no editor opens -- which makes the whole artifact useless
    rather than merely long. The full report block at the top is over EVERY example either way,
    so the statistics are not sampled; only the walkthrough is.
    """
    kept = [v for v in verdicts if v.example is not None]
    shown, total = kept[:MARKDOWN_EXAMPLES], len(kept)
    out = [
        "# L_CF counterfactual dataset",
        "",
        "Anchor = the original step. Positives preserve the mathematics, negatives break it.",
        "The anchor and its positives are ONE equivalence class (multi-positive L_CF, §7.5).",
        "",
        "```",
        report.text,
        "```",
        "",
        "---",
        "",
    ]
    if total > len(shown):
        out += [f"*Showing the first {len(shown)} of {total:,} kept examples.*", "", "---", ""]
    for i, verdict in enumerate(shown):
        ex = verdict.example
        anchor = ex.steps[ex.step_index]
        out += [
            f"## {i + 1}. step [{ex.step_index}] of {len(ex.steps)}",
            "",
            f"**Problem** — {ex.question.strip()}",
            "",
            "**Chain**",
            "",
        ]
        for j, step in enumerate(ex.steps):
            marker = "**>>**" if j == ex.step_index else "  "
            out.append(f"- {marker} `[{j}]` {step.strip()}")
        out += ["", f"**Anchor** — {anchor.strip()}", "", "**Positives** (same mathematics)", ""]
        for text in ex.positive_rewrites:
            out.append(f"- {text.strip()}")
        out += ["", "**Negatives** (different mathematics)", ""]
        for text, kind in zip(ex.negative_rewrites, verdict.error_kinds or [""] * 99):
            label = f"`{kind}` " if kind else ""
            out.append(f"- {label}{text.strip()}")
        out += ["", "---", ""]
    path.write_text("\n".join(out))


def write_and_report(
    verdicts: list[Validated],
    reasons: Counter,
    args: argparse.Namespace,
    raw: list[dict] | None = None,
) -> int:
    """Shared tail of `collect` and `generate`: write the dataset, the rejects, the readable
    report, and print the summary. `write_jsonl` is the loader's own writer, so the on-disk
    format is the loader's format by construction, not by agreement."""
    kept = [v for v in verdicts if v.example is not None]
    out = Path(args.out)
    write_jsonl([v.example for v in kept], out)

    rejects = [v for v in verdicts if v.example is None]
    if rejects:
        with out.with_suffix(".rejected.jsonl").open("w") as fh:
            for v in rejects:
                # `{"reason": ...}` alone is what this used to write, and it is a histogram
                # spread over lines -- `reasons` in the report already says how many were
                # rejected for what. The custom_id is what makes a row actionable: it joins to
                # `.responses.jsonl` for the raw response and to `cf_items.jsonl` for the
                # anchor, and the discards say which check fired on which rewrite.
                fh.write(json.dumps({
                    "custom_id": v.custom_id,
                    "reason": v.reason,
                    "dropped_positives": v.dropped_positives,
                    "dropped_negatives": v.dropped_negatives,
                    "discards": list(v.discards),
                }) + "\n")

    # Every rewrite thrown away, from KEPT and REJECTED items alike, one row per rewrite.
    # An item that lost 2 of 7 negatives is invisible in `.rejected.jsonl` -- it was kept --
    # and those are exactly the drops that decide whether a floor of 5 is the right floor.
    discarded = [d for v in verdicts for d in v.discards]
    if discarded:
        with out.with_suffix(".discarded.jsonl").open("w") as fh:
            for row in discarded:
                fh.write(json.dumps(row) + "\n")
    if raw:
        with out.with_suffix(".raw.jsonl").open("w") as fh:
            for row in raw:
                fh.write(json.dumps(row) + "\n")

    tokenizer, tokenizer_name = load_act_emb_tokenizer(args.tokenizer)
    report = build_report(verdicts, reasons, tokenizer, tokenizer_name, args)
    print(f"\n[report] wrote {len(kept)} examples to {out}")
    print(report.text)
    if not kept:
        return 1

    md = out.with_suffix(".report.md")
    write_markdown(verdicts, report, md)
    print(f"\n[report] readable version: {md}")
    if rejects:
        print(f"[report] rejects:          {out.with_suffix('.rejected.jsonl')}")
    if discarded:
        by_reason = Counter(f"{d['kind']}/{d['reason']}" for d in discarded)
        print(f"[report] discarded rewrites: {out.with_suffix('.discarded.jsonl')} "
              f"({len(discarded)} rows) {dict(by_reason.most_common())}")
    if raw:
        print(f"[report] raw model output: {out.with_suffix('.raw.jsonl')}")
    return 0


# ------------------------------------------------------------------------------- collect

def cmd_collect(args: argparse.Namespace) -> int:
    import anthropic

    manifest = json.loads(Path(args.batch).read_text())
    items = {item["custom_id"]: item for item in read_items(manifest["items"])}
    args.positives = manifest.get("positives", args.positives)
    args.negatives = manifest.get("negatives", args.negatives)
    client = anthropic.Anthropic()

    for batch_id in manifest["batch_ids"]:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            if not args.wait:
                print(f"[collect] {batch_id} is {batch.processing_status}; --wait to poll")
                return 1
            print(f"[collect] {batch_id} {batch.processing_status} "
                  f"({batch.request_counts.processing} processing)")
            time.sleep(args.poll_seconds)

    verdicts, reasons, raw = [], Counter(), []
    for batch_id in manifest["batch_ids"]:
        for result in client.messages.batches.results(batch_id):
            item = items.get(result.custom_id)
            if item is None:
                reasons["unknown_custom_id"] += 1
                continue
            if result.result.type != "succeeded":
                reasons[f"api_{result.result.type}"] += 1
                continue
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"), ""
            )
            payload = extract_json(text)
            if payload is None:
                reasons["unparseable_json"] += 1
                continue
            raw.append({"custom_id": result.custom_id, "payload": payload})
            verdict = validate(item, payload, args)
            reasons[verdict.reason] += 1
            verdicts.append(verdict)

    return write_and_report(verdicts, reasons, args, raw)


# ------------------------------------------- generate: the OpenAI-compatible endpoint path
#
# `submit`/`collect` above talk to the Anthropic Batch API. `generate` talks to any
# OpenAI-compatible `/chat/completions` endpoint -- BharatCode is the one in `secret.txt` --
# and it is the path that runs today.
#
# PROMPT CACHING. There is no `cache_control` field in the OpenAI schema; caching on these
# endpoints is automatic PREFIX caching, so the only thing that buys it is keeping the long
# shared text FIRST and BYTE-IDENTICAL in every request. That is exactly what happens here:
# message[0] is the system prompt, unchanged across the whole run, and only the short user
# message varies. The first request is sent ALONE to populate the cache; the rest then go out
# concurrently against a warm prefix rather than all missing at once.
#
# ONE CALL PER ITEM, NO ACCUMULATED HISTORY. Each item is a fresh two-message conversation.
# Nothing from item k is in item k+1's context, so quality does not decay as the run goes on
# and the input length is constant -- which is also what makes the shared prefix cacheable.

DEFAULT_MODEL_OPENAI = "bharatcode:qwen36-35b-q6-256k-vision"
_FENCE = re.compile(r"^```[a-zA-Z]*\n(.*?)\n?```$", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict | None:
    """Parse a model response into a dict, tolerating fences, reasoning blocks and prose.

    Structured output is requested (see `negotiate_response_format`), but not every
    OpenAI-compatible server honours a json_schema, so the parser has to be the backstop.
    """
    if not text:
        return None
    text = _THINK.sub("", text).strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost balanced {...}.
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : i + 1])
                    return payload if isinstance(payload, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _key_env_names(args: argparse.Namespace) -> list[str]:
    return [n.strip() for n in (getattr(args, "api_key_env", "") or "").split(",") if n.strip()]


def read_keys(args: argparse.Namespace, primary: str) -> list[str]:
    """Every key this session may rotate through, in order. One entry unless asked otherwise.

    `--api-key-env GEMINI_API_KEY,GEMINI_API_KEY_2,GEMINI_API_KEY_3` names THREE variables.
    Each key carries its own requests-per-day quota, so N keys is N times the day's anchors --
    **but only if the keys belong to different projects.** Keys minted inside one Google Cloud
    project share one quota, and rotating them then spends that quota N times over while the
    budget counter still reads healthy: the day ends in a wall of 429s and the counter says
    everything was fine. That is B11/B12's family (§14), so the two failures it can be caught
    by are checked here rather than left to the run.

    Passing the same key twice IS detectable and is refused outright. Two DIFFERENT keys on one
    project is not detectable from here; the per-key spend and the 429 count printed at the end
    of the run are what say which of the two you have.
    """
    names = _key_env_names(args)
    if len(names) <= 1:
        return [primary]
    keys, missing = [], []
    for name in names:
        value = os.environ.get(name, "")
        if value:
            keys.append(value)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"[generate] --api-key-env names {', '.join(missing)}, which is unset or empty. "
            f"Rotation is all-or-nothing: silently dropping a key would spend the remaining "
            f"ones' quota faster than the budget expects and end the day in 429s."
        )
    if len(set(keys)) != len(keys):
        raise SystemExit(
            f"[generate] two of {', '.join(names)} hold the SAME key. Rotating between them "
            f"spends ONE quota {len(keys)} times while --request-budget counts it as "
            f"{len(keys)} quotas, so the day ends in 429s with the counter reading healthy."
        )
    return keys


def read_endpoint(args: argparse.Namespace) -> tuple[str, str]:
    """Endpoint and PRIMARY key from --base-url/env, else parsed out of `secret.txt`.

    Returns one key. `read_keys` is what expands `--api-key-env A,B,C` into the rotation --
    kept separate so that every existing caller (and every test that monkeypatches this) sees
    exactly the two-tuple it always saw.
    """
    base_url = args.base_url or os.environ.get("BHARATCODE_BASE_URL") or ""
    api_key = args.api_key or os.environ.get("BHARATCODE_API_KEY") or ""
    # `--api-key-env OPENAI_API_KEY` reads the key from the environment by NAME. Passing the
    # key itself puts it in `ps` output and in shell history for the whole night; passing the
    # variable name does not, and it is the same one line at the call site.
    # A comma-separated list names SEVERAL variables (see `read_keys`); the PRIMARY key is the
    # first of them, which is what every single-key caller has always got.
    key_env = _key_env_names(args)
    if not api_key and key_env:
        api_key = os.environ.get(key_env[0], "")
        if not api_key:
            raise SystemExit(f"[generate] --api-key-env {key_env[0]} is unset or empty")
    secret = Path(args.secret)
    if (not base_url or not api_key) and secret.exists():
        text = secret.read_text()
        if not base_url:
            match = re.search(r"Endpoint:\s*(\S+)", text)
            base_url = match.group(1) if match else ""
        if not api_key:
            match = re.search(r"Bearer\s+(\S+)", text)
            api_key = match.group(1) if match else ""
    if not base_url or not api_key:
        raise SystemExit(
            f"[generate] no endpoint/key. Pass --base-url/--api-key, set BHARATCODE_BASE_URL "
            f"and BHARATCODE_API_KEY, or keep them in {secret}."
        )
    return base_url.rstrip("/"), api_key


class BudgetExhausted(Exception):
    """The per-session REQUEST budget is spent. Not retryable, and not an endpoint error."""


class RateLimiter:
    """A shared gate in front of every HTTP request: a minimum spacing between them, and a
    hard cap on how many there may be in one session.

    **It counts HTTP REQUESTS, not items**, and that distinction is the whole reason it
    exists rather than `--limit` doing the job. A retry is a request; the
    `negotiate_response_format` probe is a request; a resample after a degenerate answer is a
    request. An endpoint metered on requests-per-day charges for all of them, so a cap
    expressed in items would undercount by exactly the failure rate -- i.e. by the most on
    the worst night.

    **Spacing, not a sliding window.** A window of `rpm` permits `rpm` requests in the first
    second and none for the next 59, which satisfies a rolling-minute meter and trips a
    stricter one; spacing at `60/rpm` seconds satisfies both, and it costs nothing here
    because the campaign is bounded by requests-per-day, not by how fast a minute is used.
    `_next_slot` is a projected send time, so callers queue up behind each other rather than
    all reading the same `now`.

    Both knobs are OFF (`0`) by default. Nothing about the bharatcode, Codex or OpenAI runs
    changes: `install_rate_limiter` puts a no-op in place unless a limit is asked for.
    """

    def __init__(self, rpm: float = 0.0, budget: int = 0) -> None:
        self._interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._budget = max(0, int(budget))
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self.spent = 0

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def exhausted(self) -> bool:
        """True once the budget is gone. Read by the submit loop to stop cleanly, which is
        the same contract `--stop-after` and `--token-budget` have: stop SUBMITTING, drain
        what is in flight, write the report, and let `--resume` continue tomorrow."""
        with self._lock:
            return bool(self._budget) and self.spent >= self._budget

    def acquire(self) -> str | None:
        """Reserve a slot. Returns the key to use, or None to mean "the caller's own"."""
        with self._lock:
            if self._budget and self.spent >= self._budget:
                raise BudgetExhausted(f"{self.spent} of {self._budget} requests used")
            self.spent += 1        # a slot we are about to occupy is spent, sleep or not
            if not self._interval:
                return None
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
            wait = slot - now
        if wait > 0:
            time.sleep(wait)
        return None


class KeyPool:
    """N API keys rotated round-robin, EACH WITH ITS OWN `RateLimiter`.

    **The per-key part is the whole point.** A requests-per-day quota is a property of a key,
    so two keys is two quotas: the day's anchors double, and so does the rate the endpoint
    will accept. One shared limiter across N keys would cap the session at one key's quota and
    the extra keys would buy nothing; one limiter per key with a shared cursor is what turns
    them into N days of budget spent in one.

    So `--rpm` and `--request-budget` are **PER KEY**, and `budget`/`spent` below report the
    session totals across the pool. The nightly script prints both, because `--request-budget
    480` on three keys is 1,440 requests and a reader who assumed 480 would stop the campaign
    two thirds early.

    Round-robin rather than "whichever key is free soonest": with equal intervals the two are
    the same schedule, and a cursor needs no cross-key locking. **An exhausted key is skipped,
    not waited on** -- the loop tries every key before giving up, so a key that hits its budget
    (or that the endpoint is rate-limiting into retries) hands its traffic to the others
    instead of stalling the run behind it.
    """

    def __init__(self, keys: Sequence[str], rpm: float = 0.0, budget: int = 0) -> None:
        self._keys = list(keys)
        self._limiters = [RateLimiter(rpm, budget) for _ in self._keys]
        self._per_key_budget = max(0, int(budget))
        self._cursor = itertools.count()

    @property
    def keys(self) -> int:
        return len(self._keys)

    @property
    def budget(self) -> int:
        """The SESSION total: per-key budget times the number of keys."""
        return self._per_key_budget * len(self._keys)

    @property
    def per_key_budget(self) -> int:
        return self._per_key_budget

    @property
    def spent(self) -> int:
        return sum(limiter.spent for limiter in self._limiters)

    @property
    def per_key_spent(self) -> list[int]:
        return [limiter.spent for limiter in self._limiters]

    @property
    def exhausted(self) -> bool:
        return all(limiter.exhausted for limiter in self._limiters)

    def acquire(self) -> str:
        start = next(self._cursor)
        for offset in range(len(self._limiters)):
            index = (start + offset) % len(self._limiters)
            try:
                self._limiters[index].acquire()
            except BudgetExhausted:
                continue      # this key is done for the day; the others may not be
            return self._keys[index]
        raise BudgetExhausted(
            f"{self.spent} of {self.budget} requests used across {len(self._keys)} keys"
        )


_NO_LIMIT = RateLimiter()
_LIMITER: RateLimiter | KeyPool = _NO_LIMIT


def install_rate_limiter(
    rpm: float = 0.0, budget: int = 0, keys: Sequence[str] = ()
) -> RateLimiter | KeyPool:
    """Install the gate `post_chat` consults, and return it.

    A module-level object rather than an argument threaded through `post_chat` ->
    `negotiate_response_format` / `call_with_retries` -> the thread pool: the limit has to be
    shared by every worker, and every one of those signatures is monkeypatched in
    `tests/test_generate_counterfactuals.py`. `cmd_generate` installs a fresh one on every
    call, so it cannot leak state between runs in one process.

    Two or more `keys` install a `KeyPool`, which also chooses the key for each request. One
    key or none leaves that to the caller, exactly as before.
    """
    global _LIMITER
    if len(keys) > 1:
        _LIMITER = KeyPool(keys, rpm, budget)
    elif rpm or budget:
        _LIMITER = RateLimiter(rpm, budget)
    else:
        _LIMITER = _NO_LIMIT
    return _LIMITER


def post_chat(base_url: str, api_key: str, body: dict, timeout: float) -> dict:
    # The gate reserves the slot AND, when several keys are rotating, says which key that slot
    # belongs to. `api_key` is the fallback for every single-key caller, and a retry re-enters
    # here, so a request that 429'd on one key goes back out on the next one.
    api_key = _LIMITER.acquire() or api_key
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # MEASURED 2026-08-09: Groq sits behind Cloudflare, which rejects urllib's default
            # `Python-urllib/3.12` User-Agent with **403 error code 1010** -- before the request
            # reaches the API, so the key, the model and the body are all irrelevant to it.
            # `negotiate_response_format` then walks json_schema -> json_object -> none, gets
            # the same 403 three times and exits "the endpoint refused every request shape",
            # which names the wrong cause (B11/B12's family, §14). Any UA that is not the
            # urllib default passes; the same curl body returns 200.
            "User-Agent": "feynman-prm/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


# Groq (and some other OpenAI-compatible servers) REFUSE `response_format: json_object` with
# 400 unless the literal word "json" appears somewhere in `messages`. This prompt never says
# it -- the shape has always been carried by the json_schema, which weaker models do not
# support -- so on such an endpoint the negotiation falls all the way through to `none` and a
# 8B model answers in prose. The hint goes on the USER message, never the system one: the
# system prompt is the cacheable prefix (§7.5.6) and must stay byte-identical across runs and
# endpoints. It is appended only when it is actually required, so no other endpoint's requests
# change by one byte.
_JSON_HINT = (
    "\n\nReturn a single json object with keys anchor_result, positive_rewrites and "
    "negative_rewrites, and no other text."
)


# OpenAI's own reasoning models are OpenAI-compatible in name only on three parameters, and
# each one is a 400 that kills the run at request 1 rather than degrading quietly:
#   * `max_tokens` is REJECTED -- "Unsupported parameter ... use 'max_completion_tokens'".
#   * `temperature` accepts 1.0 and nothing else, so the 0.7 that suits the vLLM endpoint is
#     also a 400. Omitting it entirely is what the API wants, not sending 1.0.
#   * `chat_template_kwargs` is a vLLM extension and is not a valid OpenAI field; the analogue
#     is `reasoning_effort`, which is why `--no-thinking` must NOT be passed to this family.
# Matching on the model name is a heuristic, and `--completion-token-param` /
# `--temperature` override it in both directions, so a new model prefix cannot wedge the run.
_OPENAI_REASONING = re.compile(r"^(gpt-5|o[1345])(\b|[-.])", re.IGNORECASE)


def is_openai_reasoning_model(model: str) -> bool:
    """True for gpt-5* and the o-series, on a bare id or a `provider/model` one."""
    return bool(_OPENAI_REASONING.match(model.split("/")[-1]))


def as_group(item: dict | Sequence[dict]) -> list[dict]:
    """One anchor or several, always as a list. The adapter for callers that predate batching
    (`scripts/concurrency_sweep.py`) and for every hand-built fixture."""
    return [item] if isinstance(item, dict) else list(item)


def is_grouped(args: argparse.Namespace) -> bool:
    """Whether THIS SESSION batches anchors -- read off the flag, never off `len(group)`.

    The probe fixes one `response_format` for the whole run, so the shape of the request may
    not vary with how many anchors happen to be left. A file with an odd number of anchors ends
    on a group of one, and that request still has to ask for `{"anchors": [...]}`.
    """
    return int(getattr(args, "anchors_per_request", 1) or 1) > 1


def build_body(
    item: dict | Sequence[dict], args: argparse.Namespace, response_format: dict | None
) -> dict:
    group = as_group(item)
    grouped = is_grouped(args)
    system = SYSTEM_PROMPT_GROUPED if grouped else SYSTEM_PROMPT
    user = (
        group_user_message(group, args.positives, args.negatives) if grouped
        else user_message(group[0], args.positives, args.negatives)
    )
    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_object"
        and "json" not in (system + user).lower()
    ):
        user += _JSON_HINT
    reasoning = is_openai_reasoning_model(args.model)
    body = {
        "model": args.model,
        # The system prompt is the CACHEABLE PREFIX: identical bytes, always first. The grouped
        # one keeps the single-anchor prompt as an exact prefix, so batching does not cost the
        # cache hit on the ~2,300 tokens that dominate the input meter.
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    param = getattr(args, "completion_token_param", "auto")
    if param == "auto":
        param = "max_completion_tokens" if reasoning else "max_tokens"
    body[param] = args.max_tokens
    # A negative temperature means OMIT THE FIELD. That is not the same as sending the
    # default: on gpt-5 the field itself is the error, and on an endpoint that has its own
    # sampling defaults, omitting is how you get them.
    temperature = args.temperature
    if temperature is None or temperature < 0:
        pass
    elif not reasoning:
        body["temperature"] = temperature
    effort = getattr(args, "reasoning_effort", "")
    if effort:
        body["reasoning_effort"] = effort
    cache_key = getattr(args, "prompt_cache_key", "")
    if cache_key:
        # OpenAI caches the shared prefix automatically above 1,024 tokens; this only ROUTES
        # requests carrying the same key to the same cache, which raises the hit rate under
        # concurrency. It is a no-op for correctness and is off unless asked for, because it
        # is an OpenAI field and a 400 on any endpoint that does not know it.
        body["prompt_cache_key"] = cache_key
    if response_format is not None:
        body["response_format"] = response_format
    if args.no_thinking:
        # vLLM passes this through to the Qwen chat template. Much cheaper and much faster;
        # the task is arithmetic-heavy and parity-constrained, so it is not the default.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def negotiate_response_format(
    base_url: str, api_key: str, item: dict | Sequence[dict], args: argparse.Namespace
) -> tuple[dict | None, dict | None]:
    """Try json_schema, then json_object, then nothing, and KEEP whichever worked.

    Returns (response_format, the first response) -- the probe is a real request, so its
    answer is used rather than thrown away, and it is what warms the prefix cache.
    """
    grouped = is_grouped(args)
    candidates: list[dict | None] = [
        {
            "type": "json_schema",
            "json_schema": {
                "name": "counterfactual_rewrites",
                "strict": True,
                "schema": grouped_response_schema() if grouped else response_schema(),
            },
        },
        {"type": "json_object"},
        None,
    ]
    last_error = None
    for candidate in candidates:
        label = "none" if candidate is None else candidate["type"]
        try:
            response = post_chat(base_url, api_key, build_body(item, args, candidate), args.timeout)
            print(f"[generate] response_format={label} accepted")
            return candidate, response
        except BudgetExhausted:
            # Must not be swallowed by the `except Exception` below: it would be reported as
            # "the endpoint refused every request shape", which names the wrong cause on a
            # night that simply had no quota left (B11/B12's family, §14).
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            print(f"[generate] response_format={label} rejected ({exc.code}): {detail}")
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            print(f"[generate] response_format={label} failed: {type(exc).__name__}: {exc}")
            last_error = exc
    raise SystemExit(f"[generate] the endpoint refused every request shape: {last_error}")


class _Resample(Exception):
    """The server answered, but the answer is unusable and a fresh sample may not be."""


def call_with_retries(
    base_url: str,
    api_key: str,
    item: dict | Sequence[dict],
    args: argparse.Namespace,
    response_format: dict | None,
) -> tuple[dict | None, str]:
    group = as_group(item)
    # One label for the log lines, whatever the group size. A batched failure is a failure of
    # every anchor in it, and naming only the first would send the next reader to the wrong row.
    label = "+".join(one["custom_id"] for one in group)
    body = build_body(group, args, response_format)
    delay = 2.0
    for attempt in range(args.retries + 1):
        try:
            response = post_chat(base_url, api_key, body, args.timeout)
            # A degenerate response is a RETRYABLE failure, not a returned answer. Measured
            # 2026-08-08: 2 of 6 items looped on whitespace until the budget ran out, and a
            # bigger budget buys more whitespace (see `truncation_reason`). A fresh sample
            # usually escapes the loop, so spend one. This was ~33% of the pipeline's total
            # loss and the single cheapest yield fix available.
            text, finish = response_text(response)
            if attempt < args.retries and finish == "length" and extract_json(text) is None:
                why = truncation_reason(text)
                print(f"[generate] {label} {why}; resampling "
                      f"({attempt + 1}/{args.retries})", file=sys.stderr)
                delay = 0.0   # nothing is rate-limiting us; the server answered fine
                raise _Resample
            return response, "ok"
        except _Resample:
            pass
        except BudgetExhausted:
            # Nothing was sent, so nothing is retryable and nothing was paid for. Returning a
            # `response: null` row keeps the item ELIGIBLE for `--resume`, which is what a
            # request that never went out has to be.
            return None, "budget_exhausted"
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            retryable = exc.code == 429 or exc.code >= 500
            if not retryable or attempt == args.retries:
                return None, f"api_http_{exc.code}"
            print(f"[generate] {label} HTTP {exc.code} ({detail}); "
                  f"retry in {delay:.0f}s", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 -- timeouts, resets, malformed responses
            if attempt == args.retries:
                return None, f"api_{type(exc).__name__}"
            print(f"[generate] {label} {type(exc).__name__}: {exc}; "
                  f"retry in {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)
        delay = min(delay * 2, 30.0)
    return None, "api_exhausted_retries"


def response_text(response: dict) -> tuple[str, str]:
    """Return (text, finish_reason).

    **The model behind `secret.txt` is a REASONING model on vLLM**: it puts its chain of
    thought in `message.reasoning` (or `reasoning_content`) and the answer in
    `message.content`, and `usage.completion_tokens` counts BOTH. Run it with too small a
    `--max-tokens` and it spends the whole budget thinking, returns `content: ""` with
    `finish_reason: "length"`, and every item comes back empty.

    That happened on the first live run and this function reported it as unparseable JSON --
    a guard failing toward the wrong diagnosis, which is B11/B12's family (§14). The
    finish_reason is returned so `cmd_generate` can name truncation as truncation, and the
    reasoning field is a last-resort place to look for a JSON block the model wrote there.
    """
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return "", "malformed_response"
    finish = choice.get("finish_reason") or "unknown"
    content = message.get("content")
    if isinstance(content, list):   # some servers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if content:
        return content, finish
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    return reasoning, finish


# re.S matters: the measured loops are `\t\t\t\t\t\t\t\t\n` repeating, and without DOTALL a
# group spanning the newline cannot be captured, so the tab-and-newline loop reads as a clean
# truncation and gets the opposite advice.
_REPEAT_RUN = re.compile(r"(.{1,12}?)\1{40,}\s*$", re.S)


def truncation_reason(text: str) -> str:
    """Tell the two `finish_reason: "length"` failures apart. They need opposite fixes.

    (a) `content` is empty -- the chain of thought ate the whole budget. Raising
        `--max-tokens` (or `--no-thinking`) fixes it. This is what the 2026-08-08 4000-token
        run hit.
    (b) `content` started, then the sampler fell into a DEGENERATE REPETITION and emitted the
        same short run until the budget ran out. Measured on the 16000-token re-run: 2/6 items,
        one mid-`"anchor_result":` on 100k characters of spaces, one on tabs after a complete
        `unsuitable` verdict it never closed the brace on. Raising `--max-tokens` here buys
        MORE WHITESPACE and nothing else -- the remedy is a fresh sample or a repetition
        penalty.

    Reporting (b) as (a) would be the same class of mistake as reporting either as malformed
    JSON: a guard that names the wrong diagnosis sends the next reader to the wrong place
    (B11/B12's family, §14).
    """
    if not text.strip():
        return "api_truncated_while_thinking"
    return (
        "api_degenerate_repetition" if _REPEAT_RUN.search(text[-4096:])
        else "api_truncated_mid_answer"
    )


def split_group_response(
    response: dict, group: Sequence[dict]
) -> list[tuple[dict | None, str]]:
    """One response holding N anchors -> N responses, each shaped like a single-anchor one.

    **This is the reason batching costs nothing downstream.** `--resume`, `_already_done`,
    `_stream_responses`, `cf_exclude_generated.py`, `validate`, `--replay` and the report all
    key on one `custom_id` per row and one payload per row, and none of them learns that a
    request ever carried two anchors. The alternative -- a pair-shaped row that every reader
    has to special-case -- is the same kind of change as re-slicing a live items file: it works
    until one reader has not been updated, and then it works incorrectly and silently.

    Three outcomes per anchor:
      * a block for it -> a response whose `content` is that block alone.
      * the response did not PARSE at all (truncated, degenerate, prose) -> every anchor gets
        the ORIGINAL response back untouched, so `truncation_reason` reads the real failure
        rather than a repackaging of it. A batched truncation costs both anchors, and that is
        the honest price of batching, not something to paper over.
      * it parsed and the anchor's block is MISSING (the model answered for one and forgot the
        other) -> `response: null` with `pair_block_missing`, which keeps that anchor eligible
        for `--resume` and names the cause. Reporting it as a validation failure would blame
        the model for a rewrite it never wrote, which is the mistake `dedup_key` cost us once.

    `usage` rides on the FIRST row only. It is a per-REQUEST quantity and copying it onto every
    anchor would multiply the session's token count by the group size.
    """
    text, finish = response_text(response)
    payload = extract_json(text)
    blocks = (payload or {}).get("anchors")
    if not isinstance(blocks, list):
        return [(response, "ok") for _ in group]

    labels = GROUP_LABELS[: len(group)]
    clean = [block for block in blocks if isinstance(block, dict)]
    # Prefer the model's own labels, and fall back to position. Labels are used only when they
    # cover the group EXACTLY -- a partial or duplicated set of ids is worse than no ids,
    # because it silently pairs one anchor's answer with another anchor's question.
    by_label: dict[str, dict] = {}
    for block in clean:
        label = str(block.get("anchor_id") or "").strip().upper()[:1]
        if label in labels and label not in by_label:
            by_label[label] = block
    keyed = len(by_label) == len(group)

    out: list[tuple[dict | None, str]] = []
    for position, _item in enumerate(group):
        if keyed:
            block = by_label[labels[position]]
        else:
            block = clean[position] if position < len(clean) else None
        if block is None:
            out.append((None, "pair_block_missing"))
            continue
        single: dict = {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": json.dumps(block, ensure_ascii=False)},
                "finish_reason": finish,
            }],
        }
        for key in ("id", "model", "created"):
            if key in response:
                single[key] = response[key]
        if position == 0 and response.get("usage"):
            single["usage"] = response["usage"]
        out.append((single, "ok"))
    return out


def tokens_spent(totals: Counter, scope: str) -> int:
    """How much of a token GRANT this session has consumed so far.

    `total` is the honest default because a grant is billed on prompt AND completion, and on
    this workload the prompt is the larger half: a ~2,300-token system prompt plus a ~800-token
    user message against ~700-2,000 completion tokens. Counting only completions would let a
    2M-token budget spend ~5M.

    Cached prompt tokens are STILL COUNTED here, and deliberately -- caching makes them cheaper
    per token (10x on gpt-5-nano), not free, and a budget that ignored them would overshoot by
    whatever the hit rate turns out to be. Read `cached_prompt_tokens` in the closing summary
    to see what the discount actually was.
    """
    if scope == "completion":
        return totals["completion_tokens"]
    if scope == "input":
        return totals["prompt_tokens"]
    return totals["prompt_tokens"] + totals["completion_tokens"]


def accumulate_usage(totals: Counter, response: dict) -> None:
    usage = (response or {}).get("usage") or {}
    totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    totals["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = usage.get("prompt_cache_hit_tokens")
    totals["cached_prompt_tokens"] += int(cached or 0)
    totals["requests"] += 1


def cmd_generate(args: argparse.Namespace) -> int:
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    base_url, api_key = read_endpoint(args)
    items = read_items(args.items)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("[generate] no items", file=sys.stderr)
        return 1

    keys = read_keys(args, api_key)
    per_request = max(1, int(getattr(args, "anchors_per_request", 1) or 1))
    # Installed even when everything is 0, so a second `cmd_generate` in one process cannot
    # inherit the previous run's spend.
    limiter = install_rate_limiter(
        getattr(args, "rpm", 0.0), getattr(args, "request_budget", 0), keys
    )
    if limiter is not _NO_LIMIT:
        per_key = "" if len(keys) == 1 else f" PER KEY, x{len(keys)} keys"
        print(f"[generate] rate limit: {args.rpm or 'unlimited'} requests/min"
              f"{per_key}, {args.request_budget or 'unlimited'} requests this session"
              f"{per_key} (RETRIES AND THE PROBE COUNT)")
    if len(keys) > 1:
        print(f"[generate] {len(keys)} API keys, rotated round-robin, EACH WITH ITS OWN QUOTA "
              f"-> {limiter.budget:,} requests and {limiter.budget * per_request:,} anchors "
              f"this session. That holds only if the keys are on DIFFERENT PROJECTS; keys "
              f"sharing a project share a quota, and the tell is 429s with the counter still "
              f"reading healthy.")

    print(f"[generate] {len(items)} items -> {base_url} model={args.model}")
    print(f"[generate] asking for {args.positives} positives / {args.negatives} negatives; "
          f"keeping >={args.min_positives} / >={args.min_negatives}")
    if per_request > 1:
        print(f"[generate] {per_request} anchors per request, answered as separate elements "
              f"of one `anchors` list and SPLIT BACK to one row per anchor on disk -- "
              f"--resume, --replay and the report never see a batched row. Note a truncated "
              f"or degenerate response now costs {per_request} anchors, not one.")
    print(f"[generate] system prompt is "
          f"{len(SYSTEM_PROMPT_GROUPED if per_request > 1 else SYSTEM_PROMPT):,} chars, "
          f"identical and FIRST in every request -- that is what the endpoint's prefix cache "
          f"keys on")

    # **Every response is appended to disk THE MOMENT IT LANDS, and an interrupted run is
    # resumed by skipping what is already there.** All three properties matter at 50k and none
    # of them held before 2026-08-08:
    #   * `pool.map` into a list held every response in RAM until the last one returned. At
    #     ~58 KB per response (measured on the 6-item runs) that is ~2.9 GB at 50k.
    #   * Nothing reached disk until every request had finished, so a Ctrl-C, an OOM or a
    #     dropped connection at item 49,999 threw away the whole run. This file already
    #     records one paid run destroyed by a late crash; that fix persisted BEFORE
    #     validating, which is not the same thing as persisting AS YOU GO.
    #   * There was no way to resume, so the retry for a partial run was "start again".
    responses_path = Path(args.out).with_suffix(".responses.jsonl")
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(responses_path) if args.resume else set()
    if done:
        print(f"[generate] --resume: {len(done):,} responses already in {responses_path}, "
              f"{len(items) - len(done):,} to go")
    todo = [item for item in items if item["custom_id"] not in done]
    if not todo:
        print("[generate] nothing left to request; validating what is on disk")
        return validate_responses(_stream_responses(responses_path, items), args)

    # One request may carry several anchors, but the FILE is still one row per anchor. The
    # chunking lives here and nowhere else, which is what keeps every reader downstream --
    # including tomorrow's `--resume` -- unaware that batching exists.
    groups = [todo[i:i + per_request] for i in range(0, len(todo), per_request)]

    started = time.time()
    usage: Counter = Counter()
    sink = responses_path.open("a" if done else "w")

    def record(group: Sequence[dict], response: dict | None, status: str) -> None:
        rows = (
            split_group_response(response, group) if response is not None and len(group) > 1
            else [(response, status)] * len(group)
        )
        for position, (item, (row_response, row_status)) in enumerate(zip(group, rows)):
            row = {"custom_id": item["custom_id"], "status": row_status,
                   "response": row_response}
            if len(group) > 1:
                # Forensics only -- nothing reads these. They are what lets a bad row be traced
                # back to the request it shared, which is the first question anyone asks when a
                # batched anchor looks wrong.
                row["pair"] = [one["custom_id"] for one in group]
                row["pair_index"] = position
            sink.write(json.dumps(row) + "\n")
        sink.flush()    # a crash must not cost more than the request in flight
        if response is not None:
            accumulate_usage(usage, response)   # once per REQUEST, never once per anchor

    try:
        # The probe is a real request for the first group, and it warms the shared prefix
        # before the rest of the run goes out concurrently.
        response_format, first_response = negotiate_response_format(
            base_url, api_key, groups[0], args
        )
        record(groups[0], first_response, "ok")

        rest = groups[1:]
        n_done = len(groups[0])
        if rest:
            every = max(1, min(500, len(todo) // 20 or 1))
            stop_after = getattr(args, "stop_after", 0.0) or 0.0
            deadline = started + stop_after * 3600 if stop_after else None
            budget = int(getattr(args, "token_budget", 0) or 0)
            scope = getattr(args, "token_budget_scope", "total")
            # **A BOUNDED WINDOW, not one submit per item. Rewritten 2026-08-09 for the
            # nightly-session plan, and the old form could not be stopped.**
            #
            # `{pool.submit(...): item for item in rest}` queues all 70,000 requests up front.
            # `ThreadPoolExecutor.__exit__` then calls `shutdown(wait=True)`, which waits for
            # every QUEUED task, not just the running ones -- so a Ctrl-C or a deadline at
            # hour 7 of an 8-hour session would have hung until the remaining ~60,000 requests
            # drained, i.e. for days. The `--stop-after` flag would have been decorative.
            #
            # Submitting `2 x concurrency` at a time and topping up as results land is the
            # same throughput (the pool never starves at 2x depth) and bounds the stop at
            # whatever is genuinely in flight -- at most `window` requests, each capped by
            # `--timeout`. It also drops the futures dict from 70,000 entries to ~12.
            window = max(1, args.concurrency) * 2
            queue, pending, stopping = iter(rest), {}, ""
            last_print = n_done
            with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
                while True:
                    # `limiter.exhausted` is checked HERE as well as after the wait below,
                    # because the window is topped up before anything has a chance to set
                    # `stopping`. Without it the last ~`window` items are submitted only to
                    # come straight back as `budget_exhausted` -- harmless (they write a null
                    # row and stay eligible for `--resume`) but it makes the session's own
                    # "requested" count larger than the number of requests actually sent.
                    while not stopping and not limiter.exhausted and len(pending) < window:
                        group = next(queue, None)
                        if group is None:
                            break
                        pending[pool.submit(
                            call_with_retries, base_url, api_key, group, args, response_format
                        )] = group
                    if not pending:
                        break
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        group = pending.pop(future)
                        try:
                            response, status = future.result()
                        except Exception as exc:        # noqa: BLE001 -- never lose the rest
                            response, status = None, f"api_{type(exc).__name__}"
                        record(group, response, status)
                        n_done += len(group)
                        # `>= every` rather than `% every`: n_done advances by the group size,
                        # so a modulus can step straight over the multiple and print nothing
                        # for the whole run.
                        if n_done - last_print >= every or n_done >= len(todo):
                            last_print = n_done
                            rate = n_done / max(time.time() - started, 1e-9)
                            left = (len(todo) - n_done) / rate
                            print(f"[generate] {n_done:,}/{len(todo):,}  {rate * 60:.1f}/min  "
                                  f"eta {left / 3600:.1f}h", flush=True)
                    if deadline and not stopping and time.time() >= deadline:
                        # Stop SUBMITTING; the in-flight ones are already paid for, so let
                        # them land rather than throwing away work the endpoint has done.
                        stopping = "deadline"
                        print(f"\n[generate] --stop-after {stop_after}h reached. No new "
                              f"requests; draining {len(pending)} in flight, then writing the "
                              f"report over everything on disk. Re-run the SAME command "
                              f"tomorrow -- --resume picks up exactly here.", flush=True)
                    # The token budget stops the same way the clock does, and for the same
                    # reason: the requests already in flight are paid for either way, so
                    # draining them is free and killing them is not. It OVERSHOOTS by at most
                    # the window (`2 x concurrency` requests), which is why the check is
                    # against the budget rather than against budget-minus-one-request --
                    # under-spending a grant by a whole window every night is the larger cost.
                    # The REQUEST budget stops exactly as the clock and the token budget do.
                    # It is the binding constraint on a requests-per-day endpoint, where the
                    # token meter never comes close and the clock is irrelevant -- 480
                    # requests at 12/min is ~40 minutes of a 7.5-hour window.
                    if not stopping and limiter.exhausted:
                        stopping = "requests"
                        print(f"\n[generate] --request-budget {limiter.budget:,} reached. No "
                              f"new requests; draining {len(pending)} in flight. --resume "
                              f"continues from exactly here when the quota resets.", flush=True)
                    if budget and not stopping and tokens_spent(usage, scope) >= budget:
                        stopping = "budget"
                        spent = tokens_spent(usage, scope)
                        print(f"\n[generate] --token-budget {budget:,} reached "
                              f"({spent:,} {scope} tokens). No new requests; draining "
                              f"{len(pending)} in flight. --resume continues from exactly "
                              f"here when there is more budget.", flush=True)
        remaining = len(todo) - n_done
        print(f"[generate] SESSION {n_done:,} requested this session, {remaining:,} of "
              f"{len(items):,} still to do")
    except KeyboardInterrupt:
        print(f"\n[generate] interrupted. {responses_path} holds every response that landed; "
              f"re-run the same command with --resume to continue.", file=sys.stderr)
        return 130
    finally:
        sink.close()

    print(f"[generate] raw responses saved to {responses_path} (replay with --replay)")

    elapsed = time.time() - started
    cached = usage["cached_prompt_tokens"]
    prompt_tokens = usage["prompt_tokens"]
    share = f"{cached / prompt_tokens:.1%}" if prompt_tokens else "n/a"
    print(
        f"\n[generate] {usage['requests']} responses in {elapsed:.1f}s "
        f"({elapsed / max(usage['requests'], 1):.1f}s each at concurrency {args.concurrency})\n"
        f"[generate] tokens: {prompt_tokens:,} prompt / {usage['completion_tokens']:,} "
        f"completion (reasoning INCLUDED); {cached:,} prompt tokens served from cache "
        f"({share})\n"
        f"           a 0 there means the endpoint reports no cache field, NOT that the prefix "
        f"was not reused -- vLLM's prefix cache is on and unbilled, so it is not in `usage`."
    )
    n_ok = max(usage["requests"], 1)
    print(
        f"[generate] per item: {prompt_tokens / n_ok:,.0f} prompt + "
        f"{usage['completion_tokens'] / n_ok:,.0f} completion = "
        f"{(prompt_tokens + usage['completion_tokens']) / n_ok:,.0f} total tokens"
    )
    if getattr(args, "token_budget", 0):
        scope = getattr(args, "token_budget_scope", "total")
        spent = tokens_spent(usage, scope)
        print(f"[generate] BUDGET {spent:,} of {args.token_budget:,} {scope} tokens used this "
              f"session ({spent / args.token_budget:.0%})")
    if limiter is not _NO_LIMIT:
        # `limiter.spent` counts HTTP calls and `usage['requests']` counts answers, so the
        # gap between them is what retries, resamples and failures cost. On a
        # requests-per-day endpoint that gap IS the overhead, and it is the number that says
        # whether tomorrow's budget buys as many anchors as today's did.
        answered = usage["requests"]
        print(f"[generate] REQUESTS {limiter.spent:,} of "
              f"{limiter.budget or 'unlimited'} sent this session; {answered:,} answered "
              f"({limiter.spent - answered:,} spent on retries, resamples and failures)")
        if isinstance(limiter, KeyPool):
            # **Read this, and read it against the budget rather than against itself.** Even
            # spend across the keys means the rotation worked. A key stuck well below the
            # others means it exhausted early -- which on a shared-project quota is what
            # happens first, and it is the only signal from inside the process that the N
            # keys are not N quotas.
            spread = " / ".join(f"{n:,}" for n in limiter.per_key_spent)
            print(f"[generate] per key: {spread} of {limiter.per_key_budget or '-'} each, "
                  f"at {per_request} anchor(s) per request")
    return validate_responses(_stream_responses(responses_path, items), args)


def _already_done(path: Path) -> set[str]:
    """The custom_ids that actually HOLD a response. A half-written final line (killed
    mid-flush) is skipped rather than crashing the resume -- losing one response is the
    correct price.

    **A row is done only when `response` is not null.** `record` writes a line for every
    outcome, including `api_http_429` / `api_exhausted_retries` with `response: null`, so
    keying on `custom_id` alone made `--resume` skip exactly the items that failed -- the only
    ones there is anything to resume. MEASURED 2026-08-09 on Groq's 6,000 TPM free tier: one
    of 6 items exhausted its retries on 429, and re-running with `--resume` reported "nothing
    left to request" and re-validated the same 5. That is the whole purpose of the flag
    failing silently toward "healthy", which is B11/B12's family (§14).
    """
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("response") is not None and "custom_id" in row:
                done.add(row["custom_id"])
    return done


def _stream_responses(path: Path, items: Sequence[dict]) -> Iterator[tuple[dict, dict | None, str]]:
    """Yield (item, response, status) one line at a time.

    Lazy on purpose: validation only needs the parsed payload and then drops the response, so
    reading the file back as a list would put the same ~2.9 GB at 50k back into RAM that
    streaming the writes just took out of it.

    **Exactly one row per custom_id is yielded, and a failure row is dropped once a retry of
    the same item has landed.** `--resume` APPENDS, so after a resumed run the file legitimately
    holds both `{cf000004, api_http_429, response: null}` and the later successful row. Yielding
    both counts the item twice -- once in `outcomes` as a failure and once as kept -- and writes
    it to the dataset twice. The extra pass costs one set of ids, which `_already_done` already
    builds for the resume itself.
    """
    by_id = {item["custom_id"]: item for item in items}
    recovered = _already_done(path)      # ids that hold a response somewhere in the file
    seen: set[str] = set()
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = row.get("custom_id")
            item = by_id.get(cid)
            if item is None or cid in seen:
                continue
            response = row.get("response")
            if response is None and cid in recovered:
                continue                 # a later retry of this item succeeded
            seen.add(cid)
            yield item, response, row.get("status", "ok")


def validate_responses(
    triples: Iterable[tuple[dict, dict | None, str]],
    args: argparse.Namespace,
) -> int:
    """The free half: parse, validate, report. Shared by `generate` and `generate --replay`."""
    verdicts, reasons, raw = [], Counter(), []
    truncated = degenerate = 0
    n_seen = 0
    for item, response, status in triples:
        n_seen += 1
        if response is None:
            reasons[status] += 1
            continue
        text, finish_reason = response_text(response)
        payload = extract_json(text)
        if payload is None:
            # Name the failure precisely. A `finish_reason: "length"` is a --max-tokens
            # problem OR a degenerate-repetition problem, and those want opposite fixes;
            # `truncation_reason` separates them. Counted only when the parse actually
            # failed -- a response that looped on whitespace AFTER writing complete JSON is
            # a usable item, not a loss.
            kind = (
                truncation_reason(text) if finish_reason == "length" else "unparseable_json"
            )
            truncated += kind != "unparseable_json"
            degenerate += kind == "api_degenerate_repetition"
            reasons[kind] += 1
            raw.append(
                {"custom_id": item["custom_id"], "finish_reason": finish_reason,
                 "unparsed_text": text[:4000]}
            )
            continue
        verdict = validate(item, payload, args)
        # **Only UNPARSEABLE responses and REJECTED items go to `.raw.jsonl`, since
        # 2026-08-09.** It used to hold the parsed payload of every response, which at 70k
        # anchors is a ~300 MB file whose every byte is recomputable: `.responses.jsonl` holds
        # the response and `extract_json` is deterministic, so this was a second copy of the
        # first copy. The rows anyone ever opens are the ones that went wrong, and those are
        # still all here. `--replay` reads `.responses.jsonl`, never this, so nothing breaks.
        if verdict.example is None:
            raw.append({"custom_id": item["custom_id"], "payload": payload})
        reasons[verdict.reason] += 1
        verdicts.append(verdict)

    if truncated:
        print(
            f"[generate] *** {truncated}/{n_seen} responses were LOST to --max-tokens "
            f"({args.max_tokens}). ***"
        )
        if degenerate:
            # The advice below is the opposite of the advice above, which is the whole reason
            # these two are counted separately.
            print(
                f"[generate]     {degenerate} of them are DEGENERATE REPETITION, not a budget "
                f"shortfall -- the model wrote real output and then looped on one short run "
                f"until the budget ran out. Raising --max-tokens buys more of the same run. "
                f"Re-run those items (a fresh sample usually escapes) or lower --temperature "
                f"(now {args.temperature})."
            )
        if truncated - degenerate:
            print(
                f"[generate]     {truncated - degenerate} ran out with the answer still "
                f"unfinished. On a reasoning model the budget is SHARED with the chain of "
                f"thought: raise --max-tokens or pass --no-thinking."
            )
    return write_and_report(verdicts, reasons, args, raw)


def cmd_replay(args: argparse.Namespace) -> int:
    """Re-run parsing, validation and the report over a saved `.responses.jsonl`. No API
    calls, so iterating on `validate()` or the report costs nothing."""
    path = Path(args.replay)
    items = read_items(args.items)
    print(f"[replay] streaming saved responses from {path}")
    return validate_responses(_stream_responses(path, items), args)


# ---------------------------------------------------------------------------------- cli

def _add_count_args(p: argparse.ArgumentParser) -> None:
    """The counts, shared by every command that either asks for rewrites or validates them.

    3 requested / >=1 kept positives, 6 requested / >=3 kept negatives. Negatives-per-anchor
    have log-ish returns -- the sixth negative barely moves a softmax the first five already
    shape -- while DISTINCT ANCHORS have linear ones, so the budget goes to more steps rather
    than more negatives per step (§7.5).
    """
    p.add_argument("--positives", type=int, default=3)
    # **BACK TO 6, 2026-08-09, by the human's call, after one run at 7.**
    #
    # 7 was tried for a real reason -- the generator pads to the requested count by handing
    # back the ANCHOR STEP VERBATIM when it runs out of distinct errors (4 of 5 drops in
    # `cf_fix2_nothink`), and at 6 against a floor of 5 one padded slot decides between a kept
    # item and `too_few_usable_negatives`. MEASURED at 7 (`cf_fix3_nothink`): 4 of 5 items
    # returned 7/7 with ZERO drops, so the slack was real.
    #
    # It did not raise the yield, and on the one item it lost it plausibly caused the loss.
    # cf000003's target step is a single subtraction (`100-30=70`); asked for seven distinct
    # errors it returned two verbatim anchor copies and hit `api_degenerate_repetition` three
    # times. **A trivial step cannot support seven errors, and asking harder makes it pad
    # harder** -- the padding is what the extra slot was meant to absorb, so the two effects
    # cancel and the request costs tokens on every item to buy nothing on most.
    # The quota is unchanged either way: round(6/3) == round(7/3) == 2.
    p.add_argument("--negatives", type=int, default=6)
    # **FLOORS LOWERED 2026-08-15, by the human's call, off the measured rejection tables.**
    #
    # At 2/5 the three campaigns rejected 637 / 518 / 277 items for a floor miss. Reconstructing
    # the usable counts from `.rejected.jsonl` against the raw payloads says most of those were
    # near-misses, not empty items:
    #
    #   too_few_usable_negatives   >=3 usable   bharatcode 340/504   gemini 160/170   openai 18/18
    #   too_few_usable_positives   >=1 usable   bharatcode  10/98    gemini  22/203   openai 89/170
    #
    # 260 of bharatcode's 504 and 136 of gemini's 170 sat at EXACTLY 4 negatives -- one short.
    # The requested counts do NOT move: 3/6 still, so this changes what is KEPT and not what is
    # asked for, and every parity statistic stays on the same population of generations.
    p.add_argument("--min-positives", type=int, default=1)
    p.add_argument("--min-negatives", type=int, default=3)


def _add_validate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-positive-overlap", type=float, default=0.9)
    p.add_argument(
        "--max-auc-deviation",
        type=float,
        default=0.15,
        help="flag the lexical-shortcut AUC this far from chance, in EITHER direction",
    )
    p.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="the tokenizer act_emb averages over (§6.4); '' forces the regex fallback",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="select target steps from the train split (no API)")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default="data/cf/cf_items.jsonl")
    p.add_argument("--limit", type=int, default=500, help="0 for no cap")
    p.add_argument("--per-question", type=int, default=1)
    p.add_argument("--include-final-step", action="store_true")
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="restrict to questions appearing in the first N rows of the train split "
             "(0 = all). The val holdout is computed on the FULL split either way.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("submit", help="send the items to the Anthropic Batch API")
    p.add_argument("--items", default="data/cf/cf_items.jsonl")
    p.add_argument("--out", default="data/cf/cf_batch.json")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    _add_count_args(p)
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--assumed-output-tokens", type=int, default=1600)
    p.add_argument("--yes", action="store_true", help="actually submit (default is a dry run)")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("collect", help="fetch an Anthropic batch, validate and write it")
    p.add_argument("--batch", default="data/cf/cf_batch.json")
    p.add_argument("--out", default="data/cf/counterfactuals.jsonl")
    _add_count_args(p)
    _add_validate_args(p)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser(
        "generate", help="run the items against an OpenAI-compatible endpoint, live"
    )
    p.add_argument("--items", default="data/cf/cf_items.jsonl")
    p.add_argument("--out", default="data/cf/counterfactuals.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL_OPENAI)
    p.add_argument("--base-url", default="")
    p.add_argument("--api-key", default="")
    p.add_argument(
        "--api-key-env",
        default="",
        help="read the key from this environment VARIABLE (e.g. OPENAI_API_KEY). Prefer it "
             "over --api-key, which leaks the key into `ps` and shell history. A "
             "COMMA-SEPARATED LIST names several variables and rotates between them "
             "round-robin, EACH WITH ITS OWN --rpm and --request-budget -- so N keys is N "
             "times the day's anchors, provided the keys are on different projects. Two "
             "variables holding the same key are refused.",
    )
    p.add_argument("--secret", default="secret.txt")
    p.add_argument("--limit", type=int, default=0, help="0 for every item in the file")
    # 6, not "as high as you can". MEASURED 2026-08-08 by scripts/concurrency_sweep.py:
    # throughput PEAKS at 6 (1.19 items/min) and degrades above it -- 1 -> 6 buys only 1.5x
    # for 6x the parallelism, and at 8 queueing pushes latency past the gateway read timeout
    # and requests start dying as 504. See §7.5.6.
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=24000,
        help="shared with the chain of thought on a reasoning model. 4000 was measured to be "
             "spent entirely on thinking, returning empty content on every item. Raised to "
             "24000 because this endpoint is free -- but note it does NOT fix the degenerate "
             "repetition that cost 2/6 on the 16000 run; only resampling does.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="a NEGATIVE value omits the field entirely, which is what OpenAI's reasoning "
             "models require (they accept 1.0 and nothing else). Omitted automatically for "
             "gpt-5*/o-series whatever this says.",
    )
    p.add_argument(
        "--completion-token-param",
        default="auto",
        choices=["auto", "max_tokens", "max_completion_tokens"],
        help="which field carries --max-tokens. `auto` sends max_completion_tokens to "
             "gpt-5*/o-series (which 400 on max_tokens) and max_tokens to everything else.",
    )
    p.add_argument(
        "--reasoning-effort",
        default="",
        choices=["", "minimal", "low", "medium", "high"],
        help="OpenAI reasoning models only. This is their analogue of --no-thinking, and it "
             "is the single biggest lever on tokens-per-item, i.e. on how far a grant goes.",
    )
    p.add_argument(
        "--prompt-cache-key",
        default="",
        help="OpenAI only: route requests sharing this key to the same prompt cache. Caching "
             "of the shared system prefix is automatic above 1,024 tokens either way; this "
             "raises the hit rate under concurrency. A 400 on endpoints that do not know it.",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="stop SUBMITTING once this many tokens have been used (0 = no cap). Drains what "
             "is in flight, writes the report, and --resume continues from exactly there -- "
             "the same contract as --stop-after, for a paid grant instead of a clock.",
    )
    p.add_argument(
        "--anchors-per-request",
        type=int,
        default=1,
        help="how many anchors one request carries (1 = one per request, and every other "
             "campaign's requests are then byte-identical). On an endpoint metered in "
             "REQUESTS this multiplies the day's anchors directly: 2 doubles them for one "
             "extra prompt's worth of input tokens, since the ~2,300-token system prefix is "
             "shared. The anchors are answered as separate elements of an `anchors` list "
             "under a prompt that says in detail not to mix them, and are split back to one "
             "row per anchor on disk. Cost: a truncated or degenerate response now loses "
             "every anchor in the group.",
    )
    p.add_argument(
        "--rpm",
        type=float,
        default=0.0,
        help="cap the REQUEST rate at this many per minute (0 = no cap). Enforced as a "
             "minimum spacing of 60/rpm seconds between requests, shared across every "
             "worker thread, so --concurrency sets how many may be in flight and this sets "
             "how fast they go out. For an endpoint metered on requests rather than tokens. "
             "**PER KEY** when --api-key-env names several.",
    )
    p.add_argument(
        "--request-budget",
        type=int,
        default=0,
        help="stop SUBMITTING once this many HTTP REQUESTS have been sent (0 = no cap). "
             "**Counts retries, resamples and the response_format probe**, because a "
             "requests-per-day quota counts them too -- a cap expressed in items would "
             "undercount by exactly the failure rate. Drains what is in flight, writes the "
             "report, and --resume continues from there: the same contract --stop-after has "
             "for a clock and --token-budget for a grant. **PER KEY** when --api-key-env "
             "names several, because a requests-per-day quota is a property of a key.",
    )
    p.add_argument(
        "--token-budget-scope",
        default="total",
        choices=["total", "completion", "input"],
        help="what --token-budget counts. `total` (prompt + completion) is what a grant is "
             "billed on, and on this workload the prompt is the LARGER half.",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="skip the chain of thought. 12.6x cheaper (4,302 vs 54,141 completion tokens on "
             "6 items) and MEASURED WORSE: the model stops doing the per-candidate arithmetic "
             "and reports the anchor's own result for every negative, losing 2 of 6 items to "
             "the result check. Not recommended -- see §7.5.6.",
    )
    _add_count_args(p)
    _add_validate_args(p)
    p.add_argument(
        "--replay",
        default="",
        help="skip the API entirely and re-validate a saved .responses.jsonl",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip items already present in <out>.responses.jsonl and APPEND to it. Safe to "
             "pass always; on a fresh run there is nothing to skip. This is the retry for an "
             "interrupted 50k run -- without it the file is truncated and the work is gone.",
    )
    p.add_argument(
        "--stop-after",
        type=float,
        default=0.0,
        help="hours of REQUESTING after which to stop cleanly (0 = run to the end). At the "
             "deadline nothing new is submitted, the requests already in flight are allowed "
             "to land, and the report is written over everything on disk -- so no paid "
             "request is thrown away and `--resume` continues from exactly there. This is "
             "what makes a nightly session possible; do NOT use `timeout(1)` instead, which "
             "kills the process mid-flush.",
    )
    p.set_defaults(func=cmd_generate)

    args = parser.parse_args(argv)
    if args.command == "generate" and args.replay:
        return cmd_replay(args)
    if args.command in ("submit", "collect") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[warn] ANTHROPIC_API_KEY unset; the SDK will fall back to an `ant auth login` "
              "profile if one exists", file=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
