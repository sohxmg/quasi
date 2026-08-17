#!/usr/bin/env python
"""§10.1's goal-head go/no-go gate, on a phase-1 checkpoint (milestone §18.1, PLAN step 5).

    within = mean over q of mean_{j!=k} d(psi(s_T^j), psi(s_T^k))    same question
    across = mean over q!=q' of d(psi(s_T^q), psi(s_T^q'))
    ratio  = within / across

    ratio -> 1   -> STOP AND REDESIGN before spending GPU hours. The goal head cannot work
                    no matter how it is trained; the fallback is the goal-free asymmetry
                    score (§9.4), NOT a reference goal (that is a skyline, §5.1).

**READ `gate/auc`, NOT `gate/ratio`.** §10.1's "< 0.3 -> proceed" was never derived -- it is
the one number in CLAUDE.md with no §17 provenance entry -- and it is far stricter than what
the goal head needs. In 512 dimensions a mean-distance ratio of **0.62 is already 100%
same-question retrieval** (the simulation is tabulated in `losses/goal.py:terminal_separability`).
The ratio's null is what it does pin down: iid terminals with no question structure give
**1.000** to three decimals, so the ratio detects "no signal at all" and nothing finer.
`--untrained` reruns the whole thing on the base model with random heads to show that null on
the real data rather than in simulation.

Cheap: it needs only terminals, and there are 3.36 correct solutions per question on average
(§4.2), which is enough to estimate it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without `pip install -e .`

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import torch

from feynman_prm.data.collate import SequenceRow, collate
from feynman_prm.data.math_shepherd import read_sequences_parquet
from feynman_prm.data.tokenize import EmptyStep, SequenceTooLong, build_sequence, sep_token_id
from feynman_prm.diagnostics.terminal_shortcut import strip_answer_span
from feynman_prm.losses.goal import terminal_separability, terminal_spread_ratio
from feynman_prm.model.backbone import load_backbone_with_adapter, load_tokenizer, read_hidden_size
from feynman_prm.model.wrapper import FeynmanPRM
from feynman_prm.utils.checkpoint import load_config_from_checkpoint, load_heads


def _backbone(cfg, ckpt: Path, args):
    """The trained adapter, or the bare base model for the `--untrained` null."""
    if not args.untrained:
        return load_backbone_with_adapter(cfg, ckpt / "adapter")

    from transformers import AutoModel

    base = AutoModel.from_pretrained(
        cfg.model.name,
        dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        attn_implementation=cfg.model.attn_implementation,
    )
    for p in base.parameters():
        p.requires_grad_(False)
    return base.eval()


def add_gate_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags every probe that reads cached terminals needs. Shared so a second probe
    cannot silently cache a *different* set of terminals than the gate ran on."""
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", type=int, default=200)
    parser.add_argument("--batch-sequences", type=int, default=16)
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="NULL BASELINE: base backbone, no adapter, random-init psi. The gate's numbers "
        "are only interpretable against this -- a ratio is a comparison and the checkpoint "
        "alone gives you one side of it.",
    )
    parser.add_argument(
        "--mask-answer",
        action="store_true",
        help="§7.13.1's SEPARATOR, required of any lambda_term > 0 run (§16.26). Scores the "
        "gate TWICE in one process -- once normally, once with the printed final answer "
        "deleted from the input -- on the IDENTICAL row set, and prints both plus the drop. "
        "Recall that survives masking is structure; recall that collapses under it was the "
        "encoder clustering on a printed string, which transfers to nothing because a PRM "
        "scores UNFINISHED solutions.",
    )
    return parser


def decode_row(row: SequenceRow, tokenizer) -> tuple[str, list[str]]:
    """`(prompt, steps)` recovered from a pre-tokenised row.

    The parquet carries no text, and the alternative -- re-reading `trl-lib/math_shepherd` and
    joining on `qid` -- would make a GPU diagnostic depend on the dataset being cached and on a
    join that can silently mismatch. The row already knows where everything is: `state_pos[0]`
    is the separator after the prompt, and `span_start/span_end` are the steps' own half-open
    token ranges (§6.1). So this is exact by construction, not a reconstruction.
    """
    ids = row.input_ids
    prompt = tokenizer.decode(ids[: int(row.state_pos[0])])
    steps = [
        tokenizer.decode(ids[int(a):int(b)])
        for a, b in zip(row.span_start, row.span_end)
    ]
    return prompt, steps


def mask_answer_row(row: SequenceRow, tokenizer, sep_id: int, cfg) -> SequenceRow | None:
    """`row` with the printed final answer deleted from its LAST step, or None if it cannot be.

    `strip_answer_span` is imported from `diagnostics/terminal_shortcut.py` rather than
    re-written here, so "the answer span" means the same thing to the instrument that MEASURED
    the shortcut's availability (§7.13.1's `answer_match_auc` 0.927, `masked_overlap_auc`) and
    to the one that measures whether the encoder took it. Two definitions would make the two
    numbers incomparable while looking fine.

    Returns None when the last step is nothing BUT the answer, so stripping empties it, or when
    the retokenised sequence trips `max_len`. Those rows are dropped from BOTH passes by the
    caller -- never from one -- and counted, because a comparison across two different
    populations is not a comparison (§14).
    """
    prompt, steps = decode_row(row, tokenizer)
    masked = strip_answer_span(steps[-1])
    if not masked.strip():
        return None
    steps[-1] = masked
    try:
        seq = build_sequence(
            tokenizer, prompt, steps, sep_id,
            prompt_format=cfg.data.prompt_format, max_len=cfg.data.max_len,
        )
    except (EmptyStep, SequenceTooLong):
        return None
    return SequenceRow(
        qid=row.qid,
        input_ids=np.asarray(seq.input_ids, dtype=np.int64),
        state_pos=np.asarray(seq.state_pos, dtype=np.int64),
        span_start=np.asarray([s for s, _ in seq.step_spans], dtype=np.int64),
        span_end=np.asarray([e for _, e in seq.step_spans], dtype=np.int64),
        correct=row.correct,
        z=row.z,
        recovery=row.recovery,
    )


def load_gate_model(args):
    """`(model, tokenizer, cfg, device)` -- the checkpoint, or the `--untrained` null."""
    ckpt = Path(args.checkpoint)
    cfg = load_config_from_checkpoint(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg)
    model = FeynmanPRM(cfg, read_hidden_size(cfg.model.name), backbone=_backbone(cfg, ckpt, args))
    if not args.untrained:
        load_heads(model, ckpt)
    model.to(device).eval()
    return model, tokenizer, cfg, device


def select_pairs(cfg, args) -> list[tuple[int, SequenceRow]]:
    """`[(question_index, row), ...]` -- up to 4 correct trajectories of each of the first
    `--questions` questions that have >= 2 of them. Only such a question can contribute a
    within-question pair, which is what the gate measures."""
    rows = read_sequences_parquet(Path(cfg.data.dir) / "sequences.parquet", split="train")
    by_question = defaultdict(list)
    for row in rows:
        if row.correct:
            by_question[row.qid].append(row)
    qids = [q for q in sorted(by_question) if len(by_question[q]) >= 2][: args.questions]
    return [(qi, row) for qi, qid in enumerate(qids) for row in by_question[qid][:4]]


@torch.no_grad()
def encode_pairs(model, pairs, tokenizer, device, batch_sequences: int):
    """`(psi_terminals (N,D), question_index (N,))` for `pairs`, in order."""
    terminals, question_index = [], []
    pending: list[tuple[int, SequenceRow]] = []

    def flush():
        if not pending:
            return
        batch = collate([r for _, r in pending], pad_id=tokenizer.pad_token_id).to(device)
        reps = model(batch)
        for b, (qi, _) in enumerate(pending):
            terminals.append(reps.psi[int(batch.traj_terminal[b])].float().cpu())
            question_index.append(qi)
        pending.clear()

    for pair in pairs:
        pending.append(pair)
        if len(pending) >= batch_sequences:
            flush()
    flush()

    return (
        torch.stack(terminals).to(device),
        torch.as_tensor(question_index).to(device),
    )


def matched_masked_pairs(pairs, tokenizer, cfg):
    """`(kept_plain, kept_masked, dropped)` -- the same rows in both, masked and not.

    **The matching is the whole point.** A row whose last step is nothing but the answer
    cannot be masked, and a question that falls below 2 correct terminals after those drops
    can no longer contribute a within-question pair. Dropping such rows from the masked pass
    ALONE would change `gate/recall_at_1` because the population moved, not because the
    representation did -- and the delta would be read as shortcut evidence. So both passes get
    the identical row set, and what was dropped is reported rather than absorbed (§14).
    """
    sep_id = sep_token_id(tokenizer, cfg.data.sep_token)
    masked = {i: mask_answer_row(row, tokenizer, sep_id, cfg) for i, (_, row) in enumerate(pairs)}
    usable = [i for i, row in masked.items() if row is not None]

    surviving = Counter(pairs[i][0] for i in usable)
    keep = [i for i in usable if surviving[pairs[i][0]] >= 2]

    kept_plain = [pairs[i] for i in keep]
    kept_masked = [(pairs[i][0], masked[i]) for i in keep]
    dropped = {
        "rows_total": len(pairs),
        "rows_kept": len(keep),
        "rows_unmaskable": len(pairs) - len(usable),
        "rows_lost_with_their_question": len(usable) - len(keep),
        "questions_total": len({qi for qi, _ in pairs}),
        "questions_kept": len({pairs[i][0] for i in keep}),
    }
    return kept_plain, kept_masked, dropped


@torch.no_grad()
def cache_terminals(args):
    """(model, psi_terminals (N,D), question_index (N,), cfg) -- psi(s_T) of every correct
    trajectory of the first `--questions` questions that have >= 2 of them.

    Honours `--mask-answer` if the caller's parser carries it, so a probe sharing
    `add_gate_args` cannot accept the flag and silently ignore it. `goal_gate.main` does NOT
    go through here under that flag -- it needs both passes on one model load.
    """
    model, tokenizer, cfg, device = load_gate_model(args)
    pairs = select_pairs(cfg, args)
    if getattr(args, "mask_answer", False):
        _, pairs, dropped = matched_masked_pairs(pairs, tokenizer, cfg)
        print(json.dumps({"mask_answer": dropped}, indent=2))
    psi_t, qidx = encode_pairs(model, pairs, tokenizer, device, args.batch_sequences)
    return model, psi_t, qidx, cfg


def score(model, psi_t, qidx, args) -> dict:
    gate = terminal_spread_ratio(psi_t, qidx, model.distance)
    gate.update(terminal_separability(psi_t, qidx, model.distance))
    gate["gate/questions"] = len(torch.unique(qidx))
    gate["gate/terminals"] = len(psi_t)
    gate["gate/untrained_baseline"] = args.untrained
    return gate


def _verdict_lines(gate: dict) -> str:
    auc = gate["gate/auc"]
    verdict = "PROCEED" if auc > 0.9 else "INVESTIGATE" if auc > 0.7 else "STOP AND REDESIGN (§10.1)"
    n_q = max(int(gate["gate/questions"]), 1)
    return (
        f"\nauc      = {auc:.3f}  (chance 0.5)   ->  {verdict}"
        f"\nrecall@1 = {gate['gate/recall_at_1']:.3f}  (chance ~{1 / n_q:.3f})"
        f"\nratio    = {gate['gate/ratio']:.3f}  (no-structure null is 1.000; NOT gated on -- see"
        " losses/goal.py:terminal_separability)"
    )


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    parser = add_gate_args(argparse.ArgumentParser(description="the §10.1 goal-head gate"))
    args = parser.parse_args(argv)

    model, tokenizer, cfg, device = load_gate_model(args)
    pairs = select_pairs(cfg, args)

    if not args.mask_answer:
        psi_t, qidx = encode_pairs(model, pairs, tokenizer, device, args.batch_sequences)
        gate = score(model, psi_t, qidx, args)
        print(json.dumps(gate, indent=2))
        print(_verdict_lines(gate))
        return 0 if gate["gate/auc"] > 0.9 else 1

    # ---- §7.13.1's separator: the same rows, scored with and without the printed answer ----
    plain_pairs, masked_pairs, dropped = matched_masked_pairs(pairs, tokenizer, cfg)
    if not plain_pairs:
        print("no rows survived masking -- nothing to compare")
        return 1

    plain = score(model, *encode_pairs(model, plain_pairs, tokenizer, device, args.batch_sequences), args)
    masked = score(model, *encode_pairs(model, masked_pairs, tokenizer, device, args.batch_sequences), args)

    out = {
        "population": dropped,
        "unmasked": plain,
        "masked": masked,
        "delta": {k: masked[k] - plain[k] for k in ("gate/auc", "gate/recall_at_1", "gate/ratio")},
    }
    print(json.dumps(out, indent=2))

    r_plain, r_masked = plain["gate/recall_at_1"], masked["gate/recall_at_1"]
    retained = r_masked / r_plain if r_plain > 0 else float("nan")
    print(
        f"\nrecall@1  unmasked {r_plain:.3f}  ->  masked {r_masked:.3f}"
        f"   ({retained:.1%} retained)"
        f"\nauc       unmasked {plain['gate/auc']:.3f}  ->  masked {masked['gate/auc']:.3f}"
        f"\n\n{dropped['rows_kept']} of {dropped['rows_total']} rows over "
        f"{dropped['questions_kept']} of {dropped['questions_total']} questions; "
        f"{dropped['rows_unmaskable']} unmaskable, "
        f"{dropped['rows_lost_with_their_question']} lost with their question."
        "\n\nWHAT THIS DOES AND DOES NOT SAY (§7.13.1). It measures how much of the "
        "same-question terminal structure survives deleting the printed answer from the "
        "INPUT -- i.e. whether the encoder took the shortcut, not whether the shortcut was "
        "available. Availability is `scripts/terminal_shortcut.py`, it is 0.927 against a "
        "chance of 0.500, and it does not depend on any checkpoint. Report the pair; a "
        "lambda_term > 0 run that improves recall@1 has two explanations and only this "
        "separates them. There is NO threshold here on purpose -- print the ratio and let "
        "the reader judge (§14, B12: a guard that renders a verdict can render the wrong one)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
