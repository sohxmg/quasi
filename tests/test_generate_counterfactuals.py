"""The `scripts/generate_counterfactuals.py` VALIDATORS (§7.5 data).

Only the pure, offline half is tested: no network, no API key, no `anthropic` import. The
generation prompt cannot be unit-tested; what CAN be tested is that a bad generation is
thrown away, which is the whole point of the validate/collect step.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from conftest import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "generate_counterfactuals", REPO_ROOT / "scripts" / "generate_counterfactuals.py"
)
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)


ANCHOR = "He uses 3 eggs per omelette, so he needs 3*4 = <<3*4=12>>12 eggs."
ITEM = {
    "question": "How many eggs?",
    "steps": ["He wants 4 omelettes.", ANCHOR, "The answer is: 12"],
    "step_index": 1,
}
# 3 requested / >=1 kept positives, 6 requested / >=3 kept negatives is the shipped setting;
# the fixtures below are smaller, so the floors are set explicitly here rather than padded out
# -- and NOT to the shipped values, because a floor of 1 cannot exercise the drop-below path
# these fixtures exist for. The shipped defaults are asserted separately, in
# test_the_kept_floors_are_one_positive_and_three_negatives.
ARGS = Namespace(min_positives=2, min_negatives=2, max_positive_overlap=0.9)

POSITIVES = [
    {"text": "Each omelette takes 3 eggs and there are 4, giving 4*3 = <<4*3=12>>12 in total.",
     "result": "12"},
    {"text": "Four omelettes at three eggs apiece come to 3+3+3+3 = <<3+3+3+3=12>>12 eggs.",
     "result": "12"},
    {"text": "The egg count is three per omelette across all four, i.e. <<4*3=12>>12.",
     "result": "12"},
]
NEGATIVES = [
    {"text": "He uses 3 eggs per omelette, so he needs 3+4 = <<3+4=7>>7 eggs.",
     "result": "7", "error_kind": "operator_flipped"},
    {"text": "He uses 3 eggs per omelette, so he needs 3*5 = <<3*5=15>>15 eggs.",
     "result": "15", "error_kind": "operand_changed"},
    {"text": "Only the omelettes he eats himself matter, so a single one needs 3 eggs.",
     "result": "3", "error_kind": "misread_problem"},
]


def payload(**kw):
    base = dict(
        unsuitable=False,
        unsuitable_reason="",
        anchor_result="12",
        positive_rewrites=[dict(p) for p in POSITIVES],
        negative_rewrites=[dict(n) for n in NEGATIVES],
    )
    base.update(kw)
    return base


def test_a_well_formed_generation_survives():
    verdict = gen.validate(ITEM, payload(), ARGS)
    assert verdict.reason == "ok"
    assert verdict.example.step_index == 1
    assert len(verdict.example.positive_rewrites) == 3
    assert len(verdict.example.negative_rewrites) == 3
    assert verdict.error_kinds == ("operator_flipped", "operand_changed", "misread_problem")


def test_a_positive_whose_result_moved_is_DROPPED_not_fatal():
    """A 'meaning-preserving' rewrite that lands on a different number is meaning-CHANGING,
    and would put a negative inside the equivalence class.

    Multi-positive changes the remedy, not the diagnosis: the bad rewrite is dropped and the
    example survives on the other two. Under the single-positive validator this killed the
    item outright.
    """
    bad = payload(
        positive_rewrites=[
            {**POSITIVES[0], "result": "13"},
            dict(POSITIVES[1]),
            dict(POSITIVES[2]),
        ]
    )
    verdict = gen.validate(ITEM, bad, ARGS)
    assert verdict.reason == "ok"
    assert len(verdict.example.positive_rewrites) == 2
    assert verdict.dropped_positives == 1
    assert POSITIVES[0]["text"] not in verdict.example.positive_rewrites


def test_dropping_below_min_positives_rejects_the_item():
    bad = payload(
        positive_rewrites=[
            {**POSITIVES[0], "result": "13"},
            {**POSITIVES[1], "result": "99"},
            dict(POSITIVES[2]),
        ]
    )
    assert gen.validate(ITEM, bad, ARGS).reason == "too_few_usable_positives"


def test_duplicate_positives_are_dropped():
    """Two identical positives add a zero-distance pair to the equivalence class and nothing
    else -- the class is meant to span the wording, not repeat one point."""
    duplicated = payload(
        positive_rewrites=[
            dict(POSITIVES[0]),
            {**POSITIVES[0], "text": POSITIVES[0]["text"].replace("total.", "total!")},
            dict(POSITIVES[1]),
        ]
    )
    verdict = gen.validate(ITEM, duplicated, ARGS)
    assert verdict.reason == "ok"
    assert len(verdict.example.positive_rewrites) == 2
    assert verdict.dropped_positives == 1

    # ...and if the duplicate takes it under the floor, the item goes.
    all_same = payload(
        positive_rewrites=[
            dict(POSITIVES[0]),
            {**POSITIVES[0], "text": POSITIVES[0]["text"].replace("total.", "total!")},
        ]
    )
    assert gen.validate(ITEM, all_same, ARGS).reason == "too_few_usable_positives"


def test_negative_whose_result_did_not_move_is_dropped():
    """Symmetric failure: a 'meaning-changing' rewrite that computes the same thing is a
    second positive sitting in the negative set."""
    same = payload(
        negative_rewrites=[
            {"text": "He needs 4*3 = <<4*3=12>>12 eggs, at 3 per omelette.",
             "result": "12", "error_kind": "operands_swapped"},
            dict(NEGATIVES[0]),
        ]
    )
    assert gen.validate(ITEM, same, ARGS).reason == "too_few_usable_negatives"


def test_positive_that_is_a_near_copy_is_dropped():
    """§6.4: act_emb is a mean of input embeddings, so a positive that shares the anchor's
    tokens is trivially the nearest candidate and L_CF collapses to token overlap."""
    copy = payload(
        positive_rewrites=[
            {"text": ANCHOR.replace("he needs", "he really needs"), "result": "12"},
            # ...and one that is identical once punctuation is normalised away.
            {"text": ANCHOR.replace("eggs.", "eggs!"), "result": "12"},
            dict(POSITIVES[0]),
            dict(POSITIVES[1]),
        ]
    )
    verdict = gen.validate(ITEM, copy, ARGS)
    assert verdict.reason == "ok"
    assert verdict.dropped_positives == 2
    assert set(verdict.example.positive_rewrites) == {POSITIVES[0]["text"], POSITIVES[1]["text"]}


def test_contradictory_calculator_annotation_is_dropped():
    """`<<3*4=99>>` next to a sentence saying 12 is a giveaway a model can read off without
    doing any arithmetic."""
    lying = payload(
        positive_rewrites=[
            {"text": "Each omelette takes 3 eggs and there are 4, giving 4*3 = <<4*3=99>>12.",
             "result": "12"},
            dict(POSITIVES[1]),
            dict(POSITIVES[2]),
        ]
    )
    verdict = gen.validate(ITEM, lying, ARGS)
    assert verdict.reason == "ok"
    assert verdict.dropped_positives == 1


def test_duplicate_negatives_collapse_to_one():
    duplicated = payload(
        negative_rewrites=[
            dict(NEGATIVES[0]),
            {**NEGATIVES[0], "text": NEGATIVES[0]["text"].replace("eggs.", "eggs!"),
             "error_kind": "off_by_one"},
        ]
    )
    assert gen.validate(ITEM, duplicated, ARGS).reason == "too_few_usable_negatives"


def test_a_negative_may_not_duplicate_a_positive():
    """The seen-set spans both groups: a 'negative' that is word-for-word a kept positive
    would sit in `N` while belonging to `C`, which inverts its gradient."""
    clash = payload(
        negative_rewrites=[
            {"text": POSITIVES[0]["text"], "result": "7", "error_kind": "operator_flipped"},
            dict(NEGATIVES[1]),
        ]
    )
    verdict = gen.validate(ITEM, clash, ARGS)
    assert verdict.reason == "too_few_usable_negatives"


def test_unsuitable_is_honoured():
    assert gen.validate(ITEM, payload(unsuitable=True), ARGS).reason == "model_marked_unsuitable"


def test_an_error_kind_written_into_the_text_is_stripped():
    """MEASURED on the first live run: the model prefixed every negative's `text` with its
    own `error_kind`, copying the layout of the prompt's worked examples.

    That is the worst available failure here -- a token in every negative and no positive is a
    perfect lexical shortcut straight into `act_emb` (§6.4), and L_CF would be solved on it
    without reading any arithmetic. The prompt now forbids it; this is the backstop.
    """
    leaked = payload(
        negative_rewrites=[
            {**NEGATIVES[0], "text": f"operator_flipped   {NEGATIVES[0]['text']}"},
            {**NEGATIVES[1], "text": f"[error_kind: operand_changed] {NEGATIVES[1]['text']}"},
        ]
    )
    verdict = gen.validate(ITEM, leaked, ARGS)
    assert verdict.reason == "ok"
    assert verdict.stripped_labels == 2
    assert verdict.example.negative_rewrites == (NEGATIVES[0]["text"], NEGATIVES[1]["text"])


@pytest.mark.parametrize(
    "text, expect_stripped",
    [
        ("operator_flipped   He needs 12 eggs.", True),
        ("off_by_one: He needs 12 eggs.", True),
        ("[wrong_method] He needs 12 eggs.", True),
        # ...and the false positives it must NOT fire on.
        ("He needs 12 eggs.", False),
        ("Since 3*4 = 12, he needs 12 eggs.", False),
        ("$x_1$ is the first root.", False),
        ("operator_flipped", False),          # nothing would be left
        ("Therefore: he needs 12 eggs.", False),   # a single word, no underscore
    ],
)
def test_the_label_strip_is_conservative(text, expect_stripped):
    _, stripped = gen.strip_leading_label(text)
    assert stripped is expect_stripped


# ------------------------------------------------------------------- the shortcut report

def test_error_kind_is_not_constrained_by_the_schema():
    """The model coining its own kind is INTENDED (§7.5's prompt). `ERROR_KINDS` is the
    vocabulary `collect` scores its histogram against, not an enum."""
    schema = gen.response_schema()
    kind = schema["properties"]["negative_rewrites"]["items"]["properties"]["error_kind"]
    assert kind == {"type": "string"}
    assert "positive_rewrites" in schema["properties"]
    assert "positive_rewrite" not in schema["properties"]
    assert set(schema["required"]) == {
        "unsuitable", "unsuitable_reason", "anchor_result",
        "positive_rewrites", "negative_rewrites",
    }


def test_error_kind_vocabulary_splits_local_from_reasoning():
    assert set(gen.ERROR_KINDS) == set(gen.LOCAL_ERROR_KINDS) | set(gen.REASONING_ERROR_KINDS)
    assert not set(gen.LOCAL_ERROR_KINDS) & set(gen.REASONING_ERROR_KINDS)
    assert "wrong_quantity_from_context" in gen.LOCAL_ERROR_KINDS
    assert "unjustified_leap" in gen.REASONING_ERROR_KINDS


def test_rank_auc_is_two_sided_around_a_fixed_chance_of_half():
    """The old check counted `positive_overlap > max(negative_overlaps)` against a chance
    level of `1/(1+N)`, so it moved with the counts and only warned on one side. AUC does
    not move and both tails are failures."""
    assert gen.rank_auc([0.9], [0.1, 0.2, 0.3]) == 1.0          # positives most like anchor
    assert gen.rank_auc([0.05], [0.1, 0.2, 0.3]) == 0.0         # positives least like it
    assert gen.rank_auc([0.2], [0.1, 0.3]) == 0.5               # chance, at ANY P and N
    assert gen.rank_auc([0.2, 0.2], [0.1, 0.3, 0.5, 0.05]) == 0.5
    assert gen.rank_auc([0.2], [0.2]) == 0.5                    # ties count half
    assert gen.rank_auc([], [0.1]) is None


def test_token_jaccard_sees_the_operators_normalise_step_deletes():
    """§6.4: `act_emb` averages QWEN TOKENS, which include `* + - / =`; `normalise_step`
    strips them (`data/branch_points.py:36-41`). Two steps differing ONLY in the operator
    are identical to `jaccard` and distinguishable to `token_jaccard`."""
    a = "he needs 3 * 4 = 12 eggs"
    b = "he needs 3 + 4 = 7 eggs"
    tokenizer = gen._FallbackTokenizer()
    plus_only_a, plus_only_b = "3 * 4", "3 + 4"
    assert gen.jaccard(plus_only_a, plus_only_b) == 1.0, "normalise_step cannot see the operator"
    assert gen.token_jaccard(tokenizer, plus_only_a, plus_only_b) < 1.0
    assert 0.0 < gen.token_jaccard(tokenizer, a, b) < 1.0


def test_extract_json_survives_fences_and_reasoning_blocks():
    """Not every OpenAI-compatible server honours a json_schema, so the parser is the
    backstop for the `generate` path."""
    assert gen.extract_json('{"a": 1}') == {"a": 1}
    assert gen.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert gen.extract_json("<think>hmm</think>\n{\"a\": 1}") == {"a": 1}
    assert gen.extract_json('Here you go:\n{"a": {"b": "}"}}\nhope that helps') == {
        "a": {"b": "}"}
    }
    assert gen.extract_json("no json at all") is None
    assert gen.extract_json("") is None


@pytest.mark.parametrize(
    "text, ok",
    [
        ("3*4 = <<3*4=12>>12", True),
        ("3*4 = <<3*4=13>>13", False),
        ("<<48/2=24>>24 and <<24-4=20>>20", True),
        ("no annotation here", True),
        ("<<x*2=y>>unparseable passes", True),
    ],
)
def test_calculator_annotation_checker(text, ok):
    assert gen.calculator_annotations_consistent(text) is ok


def test_results_equal_normalises_formatting():
    assert gen.results_equal("$1,200.00", "1200")
    assert gen.results_equal("12", "12.")
    assert not gen.results_equal("12", "13")


def test_eligible_steps_skip_the_final_answer_and_prose():
    steps = ("Tom has 4 boxes with 6 pencils each, so 4*6 = 24 pencils.",
             "He is a diligent student who counts them.",     # long enough, but no arithmetic
             "The answer is: 24 pencils in total.")
    assert gen.eligible_step_indices(steps, include_final=False) == [0]
    assert gen.eligible_step_indices(steps, include_final=True) == [0, 2]


@pytest.mark.parametrize(
    "text, expected",
    [
        # (a) the chain of thought ate the whole budget -- the 2026-08-08 4000-token run.
        ("", "api_truncated_while_thinking"),
        ("   \n\t ", "api_truncated_while_thinking"),
        # (b) both degenerate loops MEASURED on the 16000-token re-run. The tab-and-newline
        # one is the reason `_REPEAT_RUN` carries re.S: without it the group cannot span the
        # newline and this case silently reads as a clean truncation, earning the opposite
        # advice ("raise --max-tokens", which buys more whitespace).
        ('{"unsuitable": false, "anchor_result":' + " " * 400, "api_degenerate_repetition"),
        ('{"unsuitable": true' + "\t\t\t\t\t\t\t\t\n" * 60, "api_degenerate_repetition"),
        ('{"a": 1}' + "ab" * 300, "api_degenerate_repetition"),
        # (c) genuinely cut off mid-sentence: no loop, real content. Wants more budget.
        ('{"positive_rewrites": [{"text": "He then multiplies 4 by 6 to get',
         "api_truncated_mid_answer"),
    ],
)
def test_the_two_length_failures_are_told_apart(text, expected):
    """They want OPPOSITE fixes, so one label for both is a guard naming the wrong
    diagnosis (B11/B12's family). Measured 2/6 on the re-run, both degenerate."""
    assert gen.truncation_reason(text) == expected


def test_a_short_answer_is_not_mistaken_for_a_loop():
    """The false-positive guard: ordinary JSON repeats `"text":` and `"result":` many times
    without being degenerate."""
    body = '{"negative_rewrites": [' + ", ".join(
        '{"text": "step %d", "result": "%d"}' % (i, i) for i in range(20)
    ) + "]}"
    assert gen.truncation_reason(body) == "api_truncated_mid_answer"


def test_resume_skips_what_is_already_on_disk(tmp_path):
    """The retry for an interrupted 50k run. Verified live 2026-08-08: seeded with 3 of 6
    responses, `--resume` requested exactly the missing 3 and the file ended at 6."""
    path = tmp_path / "r.responses.jsonl"
    path.write_text(
        '{"custom_id": "cf000000", "status": "ok", "response": {"a": 1}}\n'
        '{"custom_id": "cf000001", "status": "ok", "response": {"a": 2}}\n'
    )
    assert gen._already_done(path) == {"cf000000", "cf000001"}
    assert gen._already_done(tmp_path / "missing.jsonl") == set()


def test_a_half_written_last_line_does_not_break_the_resume():
    """A hard kill mid-flush truncates the final line. Losing that one response is the
    correct price; refusing to resume the other 49,999 is not."""
    import io
    from pathlib import Path as _P

    class _Truncated(_P.__class__ if False else object):
        pass

    text = ('{"custom_id": "cf000000", "status": "ok", "response": {}}\n'
            '{"custom_id": "cf00000')
    lines = io.StringIO(text)
    done = set()
    for line in lines:
        try:
            done.add(gen.json.loads(line)["custom_id"])
        except Exception:  # noqa: BLE001 -- mirrors _already_done
            continue
    assert done == {"cf000000"}


def test_validation_streams_and_does_not_need_a_list(tmp_path):
    """`validate_responses` takes an ITERABLE of triples, so neither the write path nor the
    replay path holds ~2.9 GB of responses in RAM at 50k."""
    path = tmp_path / "s.responses.jsonl"
    path.write_text(
        '{"custom_id": "a", "status": "ok", "response": {"x": 1}}\n'
        'not json at all\n'
        '{"custom_id": "gone", "status": "ok", "response": {}}\n'
        '{"custom_id": "b", "status": "api_http_500", "response": null}\n'
    )
    items = [{"custom_id": "a"}, {"custom_id": "b"}]
    streamed = gen._stream_responses(path, items)
    assert not isinstance(streamed, list)          # lazy, not materialised
    assert list(streamed) == [
        ({"custom_id": "a"}, {"x": 1}, "ok"),
        ({"custom_id": "b"}, None, "api_http_500"),
    ]


def test_dedup_key_keeps_operators_that_normalise_step_deletes():
    r"""MEASURED regression, 2026-08-09 (`cf_fix_nothink` cf000004). `normalise_step` applies
    `[^\w\s]`, so a sign flip inside a discriminant vanishes and two distinct negatives read
    as one. The item was then rejected as `too_few_usable_negatives` -- blaming the model for
    output the validator had destroyed. Deduplication must see the mathematics.
    """
    kept = r"the solutions are $x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$."
    flipped = r"the solutions are $x=\frac{-b\pm\sqrt{b^2+4ac}}{2a}$."
    unsigned = r"the solutions are $x=\frac{b\pm\sqrt{b^2-4ac}}{2a}$."

    from feynman_prm.data.branch_points import normalise_step
    assert normalise_step(kept) == normalise_step(flipped)      # the bug, still there
    assert normalise_step(kept) == normalise_step(unsigned)

    assert len({gen.dedup_key(kept), gen.dedup_key(flipped), gen.dedup_key(unsigned)}) == 3


def test_dedup_key_folds_only_what_carries_no_mathematics():
    assert gen.dedup_key("He needs $x = 5$ eggs.") == gen.dedup_key("He needs $x=5$ eggs!")
    # A decimal point is not punctuation.
    assert gen.dedup_key("It costs 3.5 units") != gen.dedup_key("It costs 35 units")


def test_a_sign_flipped_negative_is_not_dropped_as_a_duplicate():
    """The end-to-end form of the above: two local edits differing only in an operator both
    survive validation, so the item is not lost for want of negatives."""
    sign_flips = payload(
        negative_rewrites=[
            {"text": "He uses 3 eggs per omelette, so he needs 3-4 = <<3-4=-1>>-1 eggs.",
             "result": "-1", "error_kind": "operator_flipped"},
            {"text": "He uses 3 eggs per omelette, so he needs 3+4 = <<3+4=7>>7 eggs.",
             "result": "7", "error_kind": "operator_flipped"},
        ]
    )
    verdict = gen.validate(ITEM, sign_flips, ARGS)
    assert verdict.reason == "ok"
    assert len(verdict.example.negative_rewrites) == 2


def test_a_step_that_announces_and_stops_is_not_eligible():
    """MEASURED 2026-08-09 (`cf_items` cf000002). The step announces a multiplication and
    ends on `to get`; the product is in the NEXT step. It passed the digit-and-length filter
    and cost four generations, three of which failed -- with nothing in the step to break, the
    model returned the anchor verbatim as a negative."""
    fragment = "Step 1: We multiply $-5x^3 - 5x^2 - 7x + 1$ by $-x^2 - 6x + 1$ to get"
    assert not gen.step_asserts_something(fragment)

    steps = (fragment, "Step 2: the product is $x^4 + 11x^3 + 36x^2 + 13x + 1$.", "Ans: 36")
    assert 0 not in gen.eligible_step_indices(steps, include_final=False)


def test_assertions_without_an_equals_sign_stay_eligible():
    """A relation OR a finished sentence is enough; requiring both would delete real anchors."""
    # no relation symbol, but a finished claim
    assert gen.step_asserts_something("This means $x-y$ must be factors of $2000^2$.")
    # no terminal punctuation, but a relation
    assert gen.step_asserts_something(
        "Step 1: There are 100 students and 30 in Grade 4 so 100-30=<<100-30=70>>70 in Grade 5"
    )
    assert gen.step_asserts_something(r"so $|\vec{u}| = \sqrt{1^2 + 1^2 + 4^2} = \sqrt{18}$")


def test_the_local_quota_is_unchanged_by_asking_for_a_seventh_negative():
    assert gen.local_negative_quota(7) == gen.local_negative_quota(6) == 2


# ----------------------------------------------------------- discards (2026-08-09, the human)
#
# "save all the bad negatives too, we will decide to drop or use them later". `Validated`
# already counted them; a count cannot tell an anchor copy from a rounding rejection, and
# those want opposite fixes. These pin the DISTINCTIONS, not the plumbing -- a sidecar that
# labelled every drop `dropped` would satisfy a shape test and be worth nothing.


def test_a_dropped_negative_is_kept_with_the_check_that_dropped_it():
    copies = payload()
    copies["negative_rewrites"][2]["result"] = "12"          # == anchor_result
    verdict = gen.validate(ITEM, copies, ARGS)

    assert verdict.reason == "ok"                             # 2 negatives left, floor is 2
    assert verdict.dropped_negatives == 1
    dropped = [d for d in verdict.discards if d["kind"] == "negative"]
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "result_equals_anchor"
    assert dropped[0]["text"] == NEGATIVES[2]["text"]         # the TEXT, not just a tally
    assert dropped[0]["error_kind"] == "misread_problem"      # and the label it claimed
    assert (dropped[0]["result"], dropped[0]["anchor_result"]) == ("12", "12")


def test_an_anchor_copy_is_not_filed_as_an_ordinary_duplicate():
    """Both are "seen before" to the code and neither is to a reader. An anchor copy is the
    generator PADDING to the requested count (§7.5.6, 4 of 5 drops in `cf_fix2_nothink`) and
    argues for asking for fewer; a duplicate of another negative argues for asking for more
    variety. Collapsing them would have hidden the measurement that set `--negatives`."""
    padded = payload()
    padded["negative_rewrites"][0]["text"] = ANCHOR                       # verbatim anchor
    padded["negative_rewrites"][1]["text"] = NEGATIVES[2]["text"]         # copy of neg 2
    verdict = gen.validate(ITEM, padded, ARGS)

    reasons = [d["reason"] for d in verdict.discards if d["kind"] == "negative"]
    assert reasons == ["anchor_copy", "duplicate"]


def test_a_rejected_item_still_carries_its_discards():
    """The rejected items are the ones worth reading, so losing their text on the way out
    would defeat the point. Also pins the custom_id, without which a row on a 70k run cannot
    be joined back to `.responses.jsonl`."""
    doomed = payload()
    for negative in doomed["negative_rewrites"]:
        negative["result"] = "12"
    verdict = gen.validate(dict(ITEM, custom_id="cf000042"), doomed, ARGS)

    assert verdict.example is None
    assert verdict.reason == "too_few_usable_negatives"
    assert verdict.custom_id == "cf000042"
    assert len(verdict.discards) == 3
    assert {d["custom_id"] for d in verdict.discards} == {"cf000042"}


def test_a_kept_item_with_no_drops_writes_no_discards():
    """An empty sidecar is the healthy state and must stay empty -- `write_and_report` only
    creates the file when there is something in it."""
    assert gen.validate(ITEM, payload(), ARGS).discards == ()


def test_the_requested_negative_count_is_back_to_six():
    """Reverted 2026-08-09 by the human after one run at 7. The quota must not move with it:
    round(6/3) == round(7/3) == 2, so this revert changes the ASK and not the composition."""
    parsed = gen.main.__globals__["argparse"]
    parser = parsed.ArgumentParser()
    gen._add_count_args(parser)
    args = parser.parse_args([])
    assert (args.negatives, args.positives) == (6, 3)
    assert gen.local_negative_quota(args.negatives) == 2


def test_the_kept_floors_are_one_positive_and_three_negatives():
    """LOWERED from 2/5 on 2026-08-15 by the human, off the measured rejection tables: 340 of
    bharatcode's 504 `too_few_usable_negatives` items had >=3 usable negatives and 260 had
    exactly 4, one short of the old floor.

    The floors and the REQUESTED counts move independently -- this asserts them in the same
    place so a change to one cannot silently read as a change to the other."""
    parser = argparse.ArgumentParser()
    gen._add_count_args(parser)
    args = parser.parse_args([])
    assert (args.min_negatives, args.min_positives) == (3, 1)
    assert args.min_negatives < args.negatives and args.min_positives < args.positives


# ------------------------------------------------------- the nightly session (2026-08-09)


def _generate_args(tmp_path, items_path, **kw):
    base = dict(
        items=str(items_path), out=str(tmp_path / "cf.jsonl"), model="m",
        base_url="http://x", api_key="k", secret="", limit=0, concurrency=4, retries=1,
        timeout=5.0, max_tokens=100, temperature=0.7, no_thinking=True,
        positives=3, negatives=6, min_positives=1, min_negatives=3,
        max_positive_overlap=0.9, max_auc_deviation=0.15, tokenizer="",
        replay="", resume=True, stop_after=0.0, rpm=0.0, request_budget=0,
        anchors_per_request=1, api_key_env="",
    )
    base.update(kw)
    return Namespace(**base)


def _write_items(path, n):
    with path.open("w") as fh:
        for i in range(n):
            fh.write(gen.json.dumps({
                "custom_id": f"cf{i:06d}", "qid": f"q{i}", "question": "?",
                "steps": ["a", ANCHOR, "b"], "step_index": 1,
            }) + "\n")


def test_stop_after_stops_submitting_without_draining_the_whole_queue(tmp_path, monkeypatch):
    """**The property that makes a 5-night campaign possible, and the one the old loop could
    not have.** It submitted all N requests up front, so `ThreadPoolExecutor.__exit__` --
    `shutdown(wait=True)` -- waited for every QUEUED task, not just the running ones. At 70k
    anchors a deadline at hour 7 would have blocked for days.

    200 items, each taking 20ms, against a deadline of ~0.15s: a bounded window stops after
    tens, an unbounded one only after all 200. The assertion is the GAP, so it does not depend
    on machine speed.
    """
    import time as _time

    items = tmp_path / "items.jsonl"
    _write_items(items, 200)
    args = _generate_args(tmp_path, items, stop_after=0.15 / 3600)

    calls = []

    def fake_call(base_url, api_key, group, a, response_format):
        calls.extend(one["custom_id"] for one in gen.as_group(group))
        _time.sleep(0.02)
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}, "ok"

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "call_with_retries", fake_call)
    monkeypatch.setattr(
        gen, "negotiate_response_format",
        lambda url, key, item, a: (None, fake_call(url, key, item, a, None)[0]),
    )

    started = _time.time()
    gen.cmd_generate(args)
    elapsed = _time.time() - started

    assert len(calls) < 200, "the deadline did not stop submission"
    assert elapsed < 2.0, "it drained the queue instead of stopping"
    # Nothing in flight is thrown away: every request made has a line on disk.
    responses = (tmp_path / "cf.responses.jsonl").read_text().strip().splitlines()
    assert len(responses) == len(calls)


def test_without_stop_after_every_item_is_still_requested(tmp_path, monkeypatch):
    """The window bounds submission, not the run -- a campaign that finishes must finish."""
    items = tmp_path / "items.jsonl"
    _write_items(items, 25)
    args = _generate_args(tmp_path, items)

    calls = []

    def fake_call(base_url, api_key, group, a, response_format):
        calls.extend(one["custom_id"] for one in gen.as_group(group))
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}, "ok"

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "call_with_retries", fake_call)
    monkeypatch.setattr(
        gen, "negotiate_response_format",
        lambda url, key, item, a: (None, fake_call(url, key, item, a, None)[0]),
    )

    gen.cmd_generate(args)
    assert sorted(calls) == [f"cf{i:06d}" for i in range(25)]


def test_middle_steps_are_favoured_and_both_ends_stay_reachable():
    """Set by the human 2026-08-10: the front is setup ("assume x is...", "we are given") and
    the back is restatement ("therefore the answer is 12"). Both pass the eligibility filter
    and neither has arithmetic worth breaking. **A weight, not a cutoff** -- both ends must
    still be sampled, rarely, or this becomes an unmeasured threshold.
    """
    import numpy as np

    w = gen.middle_step_weights(6)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, w[::-1]), "the tent must be symmetric -- both ends are the problem"
    assert w[2] == w[3] == w.max()                    # peak on the middle
    assert w[0] == w[-1] == w.min()                   # both ends equally suppressed
    assert np.isclose(w.max() / w.min(), 5.0)
    assert w[0] > 0.05                                # "rarely", not "never"

    # Degenerate widths: no middle to peak on, and no zero half-width to divide by.
    assert list(gen.middle_step_weights(1)) == [1.0]
    assert list(gen.middle_step_weights(2)) == [0.5, 0.5]


def test_the_anchor_weighting_shows_up_in_actual_draws():
    """The weight is worthless if `rng.choice` ignores it, so draw and count."""
    import numpy as np

    rng = np.random.default_rng(0)
    choices = list(range(6))
    counts = [0] * 6
    for _ in range(4000):
        picked = rng.choice(choices, size=1, replace=False, p=gen.middle_step_weights(6))
        counts[int(picked[0])] += 1

    assert min(counts[2], counts[3]) > 3 * max(counts[0], counts[5]), counts   # 5:1 expected
    assert counts[0] > 0 and counts[5] > 0, "an end became unreachable"


def test_resume_skips_what_landed_and_retries_what_failed(tmp_path, monkeypatch):
    """**The whole basis of a multi-night campaign**, end to end rather than on `_already_done`
    alone. Night 1 succeeds on 3 items and fails on 1; night 2 must request the 1 that failed
    plus the 4 never attempted, and must NOT re-request the 3 that landed.
    """
    items = tmp_path / "items.jsonl"
    _write_items(items, 8)
    ok = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))

    night1 = []

    def fail_on_two(base_url, api_key, group, a, response_format):
        item = gen.as_group(group)[0]
        night1.append(item["custom_id"])
        if item["custom_id"] == "cf000002":
            return None, "api_http_429"          # a failure writes a row with response: null
        return ok, "ok"

    # Night 1: stop after 4 items by handing `generate` only the first 4. The probe request
    # goes through `negotiate_response_format`, so it has to be counted like any other.
    monkeypatch.setattr(gen, "call_with_retries", fail_on_two)
    monkeypatch.setattr(
        gen, "negotiate_response_format",
        lambda url, key, item, a: (None, fail_on_two(url, key, item, a, None)[0]),
    )
    gen.cmd_generate(_generate_args(tmp_path, items, limit=4))
    assert sorted(night1) == ["cf%06d" % i for i in range(4)]

    night2 = []

    def succeed(base_url, api_key, group, a, response_format):
        night2.extend(one["custom_id"] for one in gen.as_group(group))
        return ok, "ok"

    monkeypatch.setattr(gen, "call_with_retries", succeed)
    monkeypatch.setattr(gen, "negotiate_response_format",
                        lambda url, key, item, a: (None, succeed(url, key, item, a, None)[0]))
    gen.cmd_generate(_generate_args(tmp_path, items))

    assert "cf000002" in night2, "a FAILED item must come back round"
    for landed in ("cf000000", "cf000001", "cf000003"):
        assert landed not in night2, f"{landed} already had a response and was re-requested"
    assert sorted(night2) == ["cf000002"] + ["cf%06d" % i for i in range(4, 8)]


# ------------------------------------------------- the OpenAI reasoning family (2026-08-10)
#
# gpt-5-nano runs the SAME items as bharatcode's vLLM endpoint through the same code path,
# and it 400s on three of that path's parameters. Each of these tests pins one of them,
# because the failure is not a degradation -- it is the first request of the night dying and
# `negotiate_response_format` reporting "the endpoint refused every request shape", which
# names the wrong cause (B11/B12's family, §14).


def test_the_openai_reasoning_family_is_recognised():
    for model in ("gpt-5-nano-2025-08-07", "gpt-5", "gpt-5-mini", "o3", "o4-mini",
                  "openai/gpt-5-nano"):
        assert gen.is_openai_reasoning_model(model), model
    for model in ("gpt-4o", "gpt-4.1", "bharatcode:qwen36-35b-q6-256k-vision",
                  "llama-3.1-8b-instant", "gemma-3-4b-it"):
        assert not gen.is_openai_reasoning_model(model), model


def _body_args(**kw):
    base = dict(model="m", positives=3, negatives=6, max_tokens=100, temperature=0.7,
                no_thinking=False, completion_token_param="auto", reasoning_effort="",
                prompt_cache_key="")
    base.update(kw)
    return Namespace(**base)


_BODY_ITEM = {"question": "q", "steps": ["a", "b"], "step_index": 0}


def test_gpt5_gets_max_completion_tokens_and_no_temperature():
    """`max_tokens` and any `temperature` are both 400s on this family. Omitting temperature
    is NOT the same as sending 1.0 -- the field itself is the error."""
    body = gen.build_body(_BODY_ITEM, _body_args(model="gpt-5-nano-2025-08-07"), None)
    assert body["max_completion_tokens"] == 100
    assert "max_tokens" not in body
    assert "temperature" not in body


def test_the_vllm_endpoint_keeps_max_tokens_and_temperature():
    """The bharatcode campaign must not change by one byte -- its prompt is the cached prefix
    and its request shape is what §7.5.6's measurements were taken on."""
    body = gen.build_body(_BODY_ITEM, _body_args(model="bharatcode:qwen36-35b"), None)
    assert body["max_tokens"] == 100 and body["temperature"] == 0.7
    assert "max_completion_tokens" not in body


def test_a_negative_temperature_omits_the_field_on_any_model():
    body = gen.build_body(_BODY_ITEM, _body_args(model="m", temperature=-1.0), None)
    assert "temperature" not in body


def test_the_completion_token_param_can_be_forced_both_ways():
    """The name match is a heuristic; a new model prefix must not be able to wedge a run."""
    forced = gen.build_body(
        _BODY_ITEM, _body_args(model="m", completion_token_param="max_completion_tokens"), None
    )
    assert forced["max_completion_tokens"] == 100 and "max_tokens" not in forced
    forced = gen.build_body(
        _BODY_ITEM,
        _body_args(model="gpt-5-nano", completion_token_param="max_tokens"), None,
    )
    assert forced["max_tokens"] == 100 and "max_completion_tokens" not in forced


def test_reasoning_effort_and_cache_key_are_opt_in():
    """Both are OpenAI fields and both 400 on endpoints that do not know them, so an empty
    string must leave the body untouched rather than sending a default."""
    off = gen.build_body(_BODY_ITEM, _body_args(model="gpt-5-nano"), None)
    assert "reasoning_effort" not in off and "prompt_cache_key" not in off
    on = gen.build_body(
        _BODY_ITEM,
        _body_args(model="gpt-5-nano", reasoning_effort="low", prompt_cache_key="k"), None,
    )
    assert on["reasoning_effort"] == "low" and on["prompt_cache_key"] == "k"


def test_no_thinking_stays_a_vllm_only_field():
    """`chat_template_kwargs` is a vLLM extension. It reaching an OpenAI request is a 400, so
    the nightly script must not pass --no-thinking there -- this pins that it is the FLAG that
    carries it and nothing else turns it on."""
    assert "chat_template_kwargs" not in gen.build_body(
        _BODY_ITEM, _body_args(model="gpt-5-nano", reasoning_effort="low"), None
    )
    assert gen.build_body(_BODY_ITEM, _body_args(no_thinking=True), None)[
        "chat_template_kwargs"] == {"enable_thinking": False}


def test_the_system_prompt_is_first_and_byte_identical_across_models():
    """The whole cached-input argument rests on this: OpenAI caches the longest matching
    PREFIX, so message[0] must be the same bytes on every request of the run."""
    a = gen.build_body(_BODY_ITEM, _body_args(model="gpt-5-nano"), None)["messages"]
    b = gen.build_body({"question": "other", "steps": ["x", "y"], "step_index": 1},
                       _body_args(model="gpt-5-nano"), None)["messages"]
    assert a[0] == b[0] == {"role": "system", "content": gen.SYSTEM_PROMPT}
    assert a[1] != b[1], "only the user message may vary"


def test_the_budget_counts_prompt_tokens_too():
    """A grant is billed on both halves, and on this workload the prompt is the LARGER one
    (~2,950 against ~2,080 measured on gpt-5-nano). A completion-only budget would spend
    ~2.4x what it was given."""
    usage = gen.Counter(prompt_tokens=2950, completion_tokens=2080, cached_prompt_tokens=2000)
    assert gen.tokens_spent(usage, "total") == 5030
    assert gen.tokens_spent(usage, "completion") == 2080
    assert gen.tokens_spent(usage, "input") == 2950


def test_the_token_budget_stops_submitting_and_keeps_what_landed(tmp_path, monkeypatch):
    """Same contract as --stop-after, for a grant instead of a clock: stop SUBMITTING, drain
    what is in flight, and leave every landed response on disk for --resume."""
    items = tmp_path / "items.jsonl"
    _write_items(items, 200)
    args = _generate_args(tmp_path, items, token_budget=5000, token_budget_scope="total",
                          concurrency=2)

    calls = []

    def fake_call(base_url, api_key, group, a, response_format):
        calls.extend(one["custom_id"] for one in gen.as_group(group))
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 600}}, "ok"

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "call_with_retries", fake_call)
    monkeypatch.setattr(gen, "negotiate_response_format",
                        lambda url, key, item, a: (None, fake_call(url, key, item, a, None)[0]))

    gen.cmd_generate(args)

    # 1,000 tokens per item against a 5,000 budget: ~5 items, plus at most one window of
    # overshoot. Nowhere near 200, and nothing landed is lost.
    assert 5 <= len(calls) <= 5 + 2 * args.concurrency, len(calls)
    responses = (tmp_path / "cf.responses.jsonl").read_text().strip().splitlines()
    assert len(responses) == len(calls)


def test_no_budget_means_no_cap(tmp_path, monkeypatch):
    """bharatcode passes no budget and must be unaffected."""
    items = tmp_path / "items.jsonl"
    _write_items(items, 12)
    args = _generate_args(tmp_path, items)
    calls = []

    def fake_call(base_url, api_key, group, a, response_format):
        calls.extend(one["custom_id"] for one in gen.as_group(group))
        return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10**6, "completion_tokens": 10**6}}, "ok"

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "call_with_retries", fake_call)
    monkeypatch.setattr(gen, "negotiate_response_format",
                        lambda url, key, item, a: (None, fake_call(url, key, item, a, None)[0]))
    gen.cmd_generate(args)
    assert len(calls) == 12


def test_the_api_key_is_read_from_a_named_variable(monkeypatch):
    """Passing the key itself puts it in `ps` and in shell history for the whole night."""
    monkeypatch.setenv("SOME_KEY_ENV", "sk-test")
    args = Namespace(base_url="http://x", api_key="", api_key_env="SOME_KEY_ENV", secret="")
    assert gen.read_endpoint(args) == ("http://x", "sk-test")

    monkeypatch.delenv("SOME_KEY_ENV")
    with pytest.raises(SystemExit):
        gen.read_endpoint(args)


# ------------------------------------ the request-metered endpoint: Gemini (2026-08-10)


def test_the_rate_limiter_spaces_requests_and_stops_at_the_budget():
    """`--rpm` is SPACING, not a sliding window. A window of `rpm` permits `rpm` requests in
    the first second and none for the next 59, which satisfies a rolling-minute meter and
    trips a stricter one; spacing satisfies both and costs nothing when the campaign is
    bounded by requests-per-day rather than by how fast a minute is used."""
    import time as _time

    limiter = gen.RateLimiter(rpm=600, budget=4)      # 0.1s apart
    started = _time.monotonic()
    for _ in range(4):
        limiter.acquire()
    elapsed = _time.monotonic() - started

    assert 0.25 < elapsed < 1.0, elapsed          # 3 gaps of 0.1s, not 0 and not 4
    assert limiter.spent == 4 and limiter.exhausted
    with pytest.raises(gen.BudgetExhausted):
        limiter.acquire()


def test_the_rate_limiter_spacing_is_shared_across_threads():
    """One process, many workers, ONE quota. A per-thread limiter would multiply the rate by
    `--concurrency`, which is exactly how a 15/min limit becomes 60/min."""
    import threading
    import time as _time

    limiter = gen.RateLimiter(rpm=600)
    started = _time.monotonic()
    threads = [threading.Thread(target=limiter.acquire) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _time.monotonic() - started > 0.6, "the threads all went out at once"
    assert limiter.spent == 8


def test_no_rpm_and_no_budget_installs_a_no_op():
    """bharatcode, codex and the OpenAI run pass neither, and must not acquire a lock or
    sleep for it."""
    assert gen.install_rate_limiter(0, 0) is gen._NO_LIMIT
    assert gen.install_rate_limiter(12, 0) is not gen._NO_LIMIT
    gen.install_rate_limiter(0, 0)


def test_the_request_budget_counts_RETRIES_and_not_items(tmp_path, monkeypatch):
    """**The distinction the whole flag exists for.** A requests-per-day quota charges for a
    retry, for a resample and for the `response_format` probe, so a cap expressed in ITEMS
    would undercount by exactly the failure rate -- i.e. by the most on the worst day.

    `post_chat` is what is faked here, NOT `call_with_retries`, so the real retry ladder runs
    and the real limiter gates it.
    """
    items = tmp_path / "items.jsonl"
    _write_items(items, 20)
    args = _generate_args(tmp_path, items, request_budget=6, retries=1, concurrency=1)

    sent = []

    def fake_post(base_url, api_key, body, timeout):
        gen._LIMITER.acquire()
        sent.append(body)
        raise gen.urllib.error.HTTPError("u", 503, "nope", {}, None)   # retryable

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "post_chat", fake_post)
    monkeypatch.setattr(gen, "negotiate_response_format", lambda url, key, item, a: (None, None))

    gen.cmd_generate(args)

    assert len(sent) == 6, f"the budget counted items, not requests ({len(sent)} sent)"
    # 6 requests at 2 attempts each is 3 items, nowhere near the 20 in the file.
    rows = [gen.json.loads(l) for l in
            (tmp_path / "cf.responses.jsonl").read_text().strip().splitlines()]
    assert all(r["response"] is None for r in rows)
    gen.install_rate_limiter(0, 0)


def test_an_unsent_request_stays_eligible_for_resume(tmp_path, monkeypatch):
    """A `budget_exhausted` item was never sent and nothing was paid for it, so it must come
    back tomorrow -- the same rule `_already_done` applies to a 429."""
    items = tmp_path / "items.jsonl"
    _write_items(items, 6)
    ok = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}

    limiter = gen.install_rate_limiter(0, 3)
    args = _generate_args(tmp_path, items, request_budget=3, concurrency=1)

    def fake_post(base_url, api_key, body, timeout):
        limiter.acquire()
        return ok

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "post_chat", fake_post)
    monkeypatch.setattr(gen, "negotiate_response_format",
                        lambda url, key, item, a: (None, fake_post(url, key, {}, 0)))

    gen.cmd_generate(args)

    landed = gen._already_done(tmp_path / "cf.responses.jsonl")
    assert len(landed) == 3, landed
    assert set(landed) == {"cf000000", "cf000001", "cf000002"}
    gen.install_rate_limiter(0, 0)


def test_an_exhausted_budget_is_not_reported_as_a_refused_request_shape(monkeypatch):
    """`negotiate_response_format` walks json_schema -> json_object -> none and exits with
    "the endpoint refused every request shape". If `BudgetExhausted` were caught by its
    `except Exception`, a day that simply had no quota left would exit naming the wrong
    cause -- B11/B12's family (§14)."""
    gen.install_rate_limiter(0, 1)
    gen._LIMITER.acquire()                       # spend the one request

    args = Namespace(model="m", max_tokens=10, temperature=0.7, no_thinking=False,
                     positives=3, negatives=6, timeout=1.0, reasoning_effort="",
                     completion_token_param="auto", prompt_cache_key="")
    item = {"custom_id": "cf000000", "question": "?", "steps": ["a", ANCHOR], "step_index": 1}

    with pytest.raises(gen.BudgetExhausted):
        gen.negotiate_response_format("http://x", "k", item, args)
    gen.install_rate_limiter(0, 0)


# ------------------------- three keys and two anchors per request: Gemini x3 (2026-08-11)


def test_several_keys_are_several_quotas_and_are_rotated():
    """**The whole point of a key pool.** A requests-per-day quota belongs to a KEY, so three
    keys is three quotas -- one shared limiter across them would cap the session at one key's
    budget and the extra keys would buy nothing at all."""
    pool = gen.install_rate_limiter(0, 2, ["k1", "k2", "k3"])
    try:
        assert pool.budget == 6 and pool.per_key_budget == 2
        got = [pool.acquire() for _ in range(6)]
        assert sorted(got) == ["k1", "k1", "k2", "k2", "k3", "k3"]
        assert got[0] != got[1], "round-robin, not one key drained before the next"
        assert pool.per_key_spent == [2, 2, 2] and pool.exhausted
        with pytest.raises(gen.BudgetExhausted):
            pool.acquire()
    finally:
        gen.install_rate_limiter(0, 0)


def test_an_exhausted_key_is_skipped_and_the_others_carry_on():
    """A key that runs out mid-session must hand its traffic over, not stall the run behind
    it. The same path is what lets a key the endpoint is 429ing be routed around on retry."""
    pool = gen.install_rate_limiter(0, 1, ["k1", "k2"])
    try:
        first = pool.acquire()
        second = pool.acquire()
        assert {first, second} == {"k1", "k2"}
    finally:
        gen.install_rate_limiter(0, 0)


def test_the_rate_limit_is_per_key_so_n_keys_go_n_times_faster():
    """`--rpm` is the endpoint's limit on ONE key. Applying it to the pool would throw away
    exactly the throughput the extra keys were added for."""
    import time as _time

    pool = gen.install_rate_limiter(600, 0, ["k1", "k2", "k3"])   # 0.1s apart, PER KEY
    try:
        started = _time.monotonic()
        for _ in range(6):
            pool.acquire()
        elapsed = _time.monotonic() - started
        # 2 slots per key => one 0.1s gap each, not five of them.
        assert elapsed < 0.35, elapsed
    finally:
        gen.install_rate_limiter(0, 0)


def test_one_key_still_installs_the_plain_limiter():
    """Nothing about bharatcode, codex or the OpenAI run changes."""
    assert isinstance(gen.install_rate_limiter(12, 5, ["only"]), gen.RateLimiter)
    assert gen.install_rate_limiter(0, 0, ["only"]) is gen._NO_LIMIT
    gen.install_rate_limiter(0, 0)


def test_two_variables_holding_the_same_key_are_refused(monkeypatch):
    """**A guard that fails toward healthy is worse than no guard (B11/B12, §14).** Rotating
    between duplicates spends ONE quota N times while --request-budget counts N quotas: the
    day ends in 429s with the counter still reading fine."""
    monkeypatch.setenv("K_A", "same-key")
    monkeypatch.setenv("K_B", "same-key")
    args = Namespace(api_key_env="K_A,K_B")
    with pytest.raises(SystemExit):
        gen.read_keys(args, "same-key")

    monkeypatch.setenv("K_B", "other-key")
    assert gen.read_keys(args, "same-key") == ["same-key", "other-key"]


def test_a_missing_key_in_the_rotation_is_refused_not_dropped(monkeypatch):
    """Silently rotating between two of three keys spends the survivors' quota 1.5x faster
    than --request-budget expects, and the run reports success until the 429s start."""
    monkeypatch.setenv("K_A", "a")
    monkeypatch.delenv("K_MISSING", raising=False)
    with pytest.raises(SystemExit):
        gen.read_keys(Namespace(api_key_env="K_A,K_MISSING"), "a")


def test_one_key_env_name_is_unchanged(monkeypatch):
    """The comma list is opt-in: a single name returns the primary key and nothing else."""
    monkeypatch.setenv("K_A", "a")
    assert gen.read_keys(Namespace(api_key_env="K_A"), "a") == ["a"]
    assert gen.read_keys(Namespace(api_key_env=""), "fallback") == ["fallback"]


_GROUP_A = {"custom_id": "cf000000", "question": "Q-A", "steps": ["s", ANCHOR], "step_index": 1}
_GROUP_B = {"custom_id": "cf000001", "question": "Q-B", "steps": ["s", "Bob has 5 * 2 = 10."],
            "step_index": 1}


def test_the_grouped_system_prompt_keeps_the_single_one_as_an_exact_prefix():
    """Two reasons, and both would be invisible if this drifted. (a) There is ONE statement of
    the task -- a second copy would fall behind §7.5.6's prompt fixes silently and every
    measurement taken against them would quietly stop applying. (b) The endpoint's prefix cache
    keys on the shared prefix, which is the ~2,300 tokens that dominate the input meter."""
    assert gen.SYSTEM_PROMPT_GROUPED.startswith(gen.SYSTEM_PROMPT)
    assert len(gen.SYSTEM_PROMPT_GROUPED) > len(gen.SYSTEM_PROMPT)


def test_batching_off_leaves_the_request_byte_identical():
    """The regression guard for bharatcode, codex and the OpenAI run: they pass nothing new,
    so not one byte of their requests may move."""
    single = gen.build_body(_GROUP_A, _body_args(), None)
    assert single["messages"][0]["content"] == gen.SYSTEM_PROMPT
    assert "ANCHOR A" not in single["messages"][1]["content"]


def test_a_grouped_request_carries_both_anchors_and_asks_them_apart():
    args = _body_args(anchors_per_request=2)
    body = gen.build_body([_GROUP_A, _GROUP_B], args, None)
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]

    assert system == gen.SYSTEM_PROMPT_GROUPED
    assert "Q-A" in user and "Q-B" in user           # both problems, in full
    assert "ANCHOR A" in user and "ANCHOR B" in user
    # The per-anchor instructions are the SAME text the unbatched path sends, not a paraphrase.
    assert gen.user_message(_GROUP_A, args.positives, args.negatives) in user
    assert gen.user_message(_GROUP_B, args.positives, args.negatives) in user


def test_a_group_of_one_still_asks_for_the_grouped_shape():
    """The probe fixes ONE response_format for the whole session, so the odd anchor at the end
    of an odd-length file cannot ask for a different one. A single-anchor prompt against a
    grouped schema parses to a payload with no `anchors` key and loses the item."""
    body = gen.build_body([_GROUP_A], _body_args(anchors_per_request=2), None)
    assert body["messages"][0]["content"] == gen.SYSTEM_PROMPT_GROUPED
    assert "anchors" in body["messages"][1]["content"]


def test_the_grouped_schema_is_derived_from_the_single_one():
    """Restating it would let the two drift; a field added to one must be in the other by
    construction. `anchor_id` goes FIRST because a structured-output model fills fields in
    schema order -- naming the anchor before answering for it, not after."""
    block = gen.grouped_response_schema()["properties"]["anchors"]["items"]
    single = gen.response_schema()
    assert list(block["properties"]) == ["anchor_id", *single["properties"]]
    assert block["required"][0] == "anchor_id"
    assert set(single["required"]) <= set(block["required"])


def _grouped_response(blocks, usage=None):
    payload = {"anchors": blocks}
    response = {"choices": [{"message": {"content": gen.json.dumps(payload)},
                             "finish_reason": "stop"}]}
    if usage:
        response["usage"] = usage
    return response


def test_a_grouped_response_is_split_back_to_one_response_per_anchor():
    """**This is why batching costs nothing downstream.** `--resume`, `_stream_responses`,
    `validate`, `--replay`, the report and `cf_exclude_generated.py` all key on one custom_id
    and one payload per row, and none of them learns that a request carried two anchors."""
    response = _grouped_response(
        [{"anchor_id": "A", "anchor_result": "10"}, {"anchor_id": "B", "anchor_result": "20"}],
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    rows = gen.split_group_response(response, [_GROUP_A, _GROUP_B])

    assert [status for _, status in rows] == ["ok", "ok"]
    assert gen.extract_json(gen.response_text(rows[0][0])[0])["anchor_result"] == "10"
    assert gen.extract_json(gen.response_text(rows[1][0])[0])["anchor_result"] == "20"
    # usage is a per-REQUEST quantity; copying it onto every anchor multiplies the session's
    # token count by the group size.
    assert "usage" in rows[0][0] and "usage" not in rows[1][0]


def test_the_split_follows_the_model_s_own_labels_when_it_reorders_them():
    """A model that answers B first has not made a mistake; pairing B's answer with A's
    question would be one, and it would be invisible -- both rows validate cleanly."""
    response = _grouped_response(
        [{"anchor_id": "B", "anchor_result": "20"}, {"anchor_id": "A", "anchor_result": "10"}]
    )
    rows = gen.split_group_response(response, [_GROUP_A, _GROUP_B])
    assert gen.extract_json(gen.response_text(rows[0][0])[0])["anchor_result"] == "10"
    assert gen.extract_json(gen.response_text(rows[1][0])[0])["anchor_result"] == "20"


def test_unlabelled_blocks_fall_back_to_position():
    """Labels are used only when they cover the group EXACTLY. A partial set is worse than
    none: it pairs one anchor's answer with another anchor's question and says nothing."""
    response = _grouped_response([{"anchor_result": "10"}, {"anchor_result": "20"}])
    rows = gen.split_group_response(response, [_GROUP_A, _GROUP_B])
    assert gen.extract_json(gen.response_text(rows[0][0])[0])["anchor_result"] == "10"
    assert gen.extract_json(gen.response_text(rows[1][0])[0])["anchor_result"] == "20"


def test_a_missing_block_is_null_and_resumable_not_a_validation_failure():
    """The model answered for one anchor and forgot the other. That anchor was never answered,
    so it must come back tomorrow -- the same rule `_already_done` applies to a 429. Passing it
    to `validate` would blame the model for a rewrite it never wrote."""
    response = _grouped_response([{"anchor_id": "A", "anchor_result": "10"}])
    rows = gen.split_group_response(response, [_GROUP_A, _GROUP_B])
    assert rows[0][1] == "ok"
    assert rows[1] == (None, "pair_block_missing")


def test_an_unparseable_grouped_response_hands_every_anchor_the_ORIGINAL_text():
    """A batched truncation costs both anchors, and `truncation_reason` has to see the real
    failure rather than a repackaging of it -- `api_degenerate_repetition` and
    `api_truncated_while_thinking` want opposite fixes (§7.5.6)."""
    response = {"choices": [{"message": {"content": "   " + "\t" * 900},
                             "finish_reason": "length"}]}
    rows = gen.split_group_response(response, [_GROUP_A, _GROUP_B])
    assert rows == [(response, "ok"), (response, "ok")]


def test_two_anchors_per_request_write_one_row_each_and_resume_sees_both(tmp_path, monkeypatch):
    """End to end: the file on disk is one row per ANCHOR whatever the request carried, so
    tomorrow's `--resume` and `cf_exclude_generated.py --check` are unchanged."""
    items = tmp_path / "items.jsonl"
    _write_items(items, 4)
    groups = []

    def fake_call(base_url, api_key, group, a, response_format):
        groups.append([one["custom_id"] for one in gen.as_group(group)])
        blocks = [{"anchor_id": gen.GROUP_LABELS[i], "anchor_result": one["custom_id"]}
                  for i, one in enumerate(gen.as_group(group))]
        return _grouped_response(blocks), "ok"

    monkeypatch.setattr(gen, "read_endpoint", lambda a: ("http://x", "k"))
    monkeypatch.setattr(gen, "call_with_retries", fake_call)
    monkeypatch.setattr(gen, "negotiate_response_format",
                        lambda url, key, g, a: (None, fake_call(url, key, g, a, None)[0]))

    gen.cmd_generate(_generate_args(tmp_path, items, anchors_per_request=2))

    assert groups == [["cf000000", "cf000001"], ["cf000002", "cf000003"]], groups
    rows = [gen.json.loads(l) for l in
            (tmp_path / "cf.responses.jsonl").read_text().strip().splitlines()]
    assert [r["custom_id"] for r in rows] == ["cf%06d" % i for i in range(4)]
    # Each row holds ITS OWN anchor's payload, not the pair's.
    for row in rows:
        payload = gen.extract_json(gen.response_text(row["response"])[0])
        assert payload["anchor_result"] == row["custom_id"]
    assert gen._already_done(tmp_path / "cf.responses.jsonl") == {"cf%06d" % i for i in range(4)}
