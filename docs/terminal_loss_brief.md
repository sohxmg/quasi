# Brief: ⑥ `L_term` — multi-positive counterfactual loss over a question's terminals

Working directory `/Users/macos/machinelearning/feynman_reasoning/feynman-prm`.

## Read first, in this order

- `CLAUDE.md` §0 (the rules — especially rule 5, never fabricate a value), §17 (every number is
  labelled measured / simulated / derived), §14 (the B11–B14 family: a guard that fails toward
  "healthy" or names the wrong diagnosis is worse than no guard).
- §16.26 — **this task is that item.** Read it and the paragraph above it at §9.9's end.
- §9.8.3, §9.9 — `gate/recall_at_1` 0.618 untrained → 0.276 trained, and the ~44% figure.
- §10.1.1 — where it was written down in advance that no loss pulls two correct terminals
  together.
- §7.5, §7.5.1 — the multi-positive L_CF you are about to reuse. Read the loss docstring in
  `feynman_prm/losses/counterfactual.py` in full; the two REQUIREMENT properties there are
  requirements here too.
- §6.5 (`d(from, to)` argument order), §6.1 (the one-sequence-per-trajectory construction),
  §8.1 (`min(4, k_c)` correct + `min(3, k_i)` incorrect per question).

## State of play

`counterfactual_loss(phi, variant_example, variant_kind, distance, temperature)` in
`feynman_prm/losses/counterfactual.py` already implements SupCon `L_out` over an equivalence
class, and it is **agnostic to what a "variant" is**. It takes a flat `(V, D)` tensor of
latents, a group id per row, and a kind per row (0 = anchor, 1 = positive, 2 = negative). It
does not know or care that the current caller means "reworded steps of one example".

So the loss is done. **This task is data plumbing plus one honest measurement.** Do not fork
the loss, do not copy it, do not add a `mode=` flag to it.

## What to build

For each question in a batch: every **correct** trajectory's terminal latent joins one
equivalence class; every **incorrect** trajectory's terminal is a negative. Same loss, grouped
by question instead of by CF example.

The two properties from §7.5.1 carry over unchanged and are still requirements:

- **Positives stay in the denominator.** All correct solutions of one question are ONE class,
  not `|P|` independent pull-together problems.
- **Negatives are never queries.** Two solutions that are wrong in different ways end at
  different wrong answers; nothing may pull them together. Assert this in a test the same way
  `tests/test_counterfactual.py` does — permuting the negatives leaves the loss bit-identical.

### 1. Data

`feynman_prm/data/math_shepherd.py` already groups by question (`build_questions`, `Question`,
`Trajectory` with `.correct`), and `feynman_prm/data/sampler.py` already takes
`min(4, k_c)` + `min(3, k_i)` per question. **The terminals you need are already in every
batch and are already forwarded.** Confirm this by reading `feynman_prm/data/collate.py`
before writing anything — if a `question_index` per sequence is already carried through, use
it; if not, add one, and say in the commit which it was.

The cost claim to verify and then record: **zero extra LM forwards.** If that turns out to be
false — e.g. the collate drops the grouping and you would have to re-forward — STOP and report
it rather than paying for it quietly. It changes the economics of the whole proposal.

### 2. The loss

`feynman_prm/losses/terminal_class.py`, one thin function that builds
`(phi, group, kind)` from the batch and calls `counterfactual_loss`. Kind 0 for the first
correct terminal, 1 for the rest, 2 for the incorrect ones. Skip questions with fewer than 2
correct terminals — the loss already drops a class of size 1 rather than crashing, but do not
rely on that silently; count them and log the count.

Which latent: use the terminal `ψ`, the same tensor `L_step` compares, NOT a fresh head. Check
`feynman_prm/model/` for what is already exposed and reuse it. If ψ is detached anywhere on
that path, find out why before undetaching it — §7.9 records a collapse test that motivated at
least one `.detach()`.

### 3. Config and wiring

`lambda_term: 0.0` in `config/default.yaml`, alongside `lambda_cf`. **It ships at zero.** The
total loss must be bit-identical with and without the term at 0.0 — mirror
`tests/test_smoke.py`'s existing check and
`test_the_shipped_total_is_bit_identical_with_and_without_the_cf_term`.

### 4. Diagnostics

Mirror the `cf/` key set: `term/loss`, `term/questions`, `term/positive_distance` (over ALL
class pairs, both orderings — not one slot), `term/negative_distance`, `term/positives_per_question`,
`term/negatives_per_question`, `term/questions_skipped_single_correct`. The empty path must log
the SAME KEYS as the populated one; `_empty_info` in `counterfactual.py` shows the pattern and
says why.

Also log `within_question_terminal_spread` if it is not already logged — §16.26 names it as the
statistic this term is supposed to move, and `gate/recall_at_1` as the one that decides whether
it worked.

## The measurement that decides whether this is real

**A shortcut exists here and you must measure it before anyone trains on this.**

Every correct trajectory ends with the same printed answer (`The answer is: 60`) and the
incorrect ones end with different numbers. So the encoder can solve this loss by **reading the
final number and clustering on it** — learning to match a printed string, not to judge
reasoning. That transfers to nothing: a PRM scores *unfinished* solutions, where no answer has
been printed yet.

This is the same disease as §7.5.6's lexical shortcut, one level up, so measure it the same
way — a statistic with a fixed chance level, reported in both directions. Suggested: strip or
mask the final answer span and report how much of the class structure survives. **Design the
statistic yourself and justify the chance level in a comment.** Report it; do not gate on it.
Nobody has measured this rate, and §7.5.6 records what gating on an unmeasured rate costs.

## Tests (`tests/test_terminal_class.py`, CPU, no model)

- Two correct terminals of one question are pulled together; a third correct one that is
  already the closest is NOT pulled further (the SupCon property — `∂L_q/∂s_p = softmax_p −
  1/|P|`; `tests/test_counterfactual.py` has this pinned and explains it).
- Every incorrect terminal is pushed away, unconditionally.
- Permuting the incorrect terminals leaves the loss **bit-identical** (float equality, not
  `allclose`); moving two incorrect terminals toward each other changes nothing.
- A question with 1 correct terminal is skipped and counted, not crashed.
- Ragged: questions with different correct/incorrect counts in one batch.
- Grouping is by QUESTION — two trajectories of different questions never share a class. Build
  a batch where they interleave and assert it.
- `lambda_term = 0.0` leaves the shipped total bit-identical.

## Done when

- `pytest -m "not gpu"` green.
- `lambda_term` is 0.0 and the total is bit-identical with and without the term.
- The shortcut statistic is implemented, run on a real batch, and its value written into
  CLAUDE.md labelled **MEASURED** with the date — or, if it could not be run, written in as
  **UNMEASURED** with the reason. Never a fabricated number (§0 rule 5).
- CLAUDE.md §16.26 updated from proposal to implemented-and-not-yet-run, with a new subsection
  giving the loss, the diagnostics, the shortcut statistic, and the zero-extra-forwards claim
  labelled per §17. Cross-reference §7.5.1 rather than restating the loss.

## Out of scope

- Do not raise `lambda_term` above 0.0 or launch a training run.
- Do not touch `feynman_prm/losses/counterfactual.py` except to add a docstring line noting the
  second caller. If you believe it needs a real change, stop and report why.
- Do not touch the L_CF generator (`scripts/generate_counterfactuals.py`) or its data.
- §16.26 says the free NCE masks should be judged first because they are free and unambiguously
  correct. You are building the term, not overriding that order — note it in the docs.

## One honest caveat to carry into the write-up

§16.26 states plainly that if `within_question_terminal_spread` and `gate/recall_at_1` do not
move under the masks, they will not move under this either. **This term is a hypothesis, not a
fix.** Write it up that way.
