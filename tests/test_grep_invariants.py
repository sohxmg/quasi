"""Grep tests. These enforce structural rules that no unit test can, because the failure
they guard against is a line of code EXISTING, not a value being wrong (§15, §7.9, §7.1)."""

from __future__ import annotations

import re

from conftest import REPO_ROOT

PACKAGE = REPO_ROOT / "feynman_prm"
SOURCES = sorted(PACKAGE.rglob("*.py")) + sorted((REPO_ROOT / "scripts").glob("*.py"))


def _code_lines(path):
    """Lines with comments and docstring-only lines stripped, roughly. Good enough to tell a
    mention in prose from a call in code."""
    text = path.read_text()
    text = re.sub(r'"""..*?"""', "", text, flags=re.S)
    text = re.sub(r"#.*", "", text)
    return text


def test_no_value_head_anywhere():
    """Locked #3a / §7.9: one head, one score -- the distance. A value_head is a second
    scoring path, and the old ① BT trained a scalar nothing at eval ever read (root cause C).
    §14's launch guard says a value_head appearing in the trainable set means someone
    reintroduced §7.9; this catches it one step earlier."""
    offenders = [p.name for p in SOURCES if "value_head" in _code_lines(p)]
    assert not offenders, offenders


def test_no_deleted_loss_names_in_code():
    """§18: there is no L_BT, no L_CRM and no L_correct. If any appears in the logs, someone
    reinstated a deleted term (§7.9)."""
    banned = ("lambda_bt", "lambda_crm", "lambda_correct", "L_correct", "survival_loss")
    offenders = {
        p.name: [b for b in banned if b in _code_lines(p)]
        for p in SOURCES
        if any(b in _code_lines(p) for b in banned)
    }
    assert not offenders, offenders


def test_no_torch_diagonal_on_the_shared_matrix():
    """§7.1: Dist is RECTANGULAR and has no diagonal -- source rows from incorrect
    trajectories are negative-only. Every "diagonal" in tmd.py becomes a pos_row gather."""
    offenders = [p.name for p in SOURCES if "torch.diagonal" in _code_lines(p)]
    assert not offenders, offenders


def test_exactly_one_sequence_builder():
    """The old project's root cause I was two prompt templates diverging by whitespace. Only
    data/tokenize.py may assemble a sequence; everything else imports build_sequence."""
    builders = [p for p in SOURCES if "def build_sequence" in _code_lines(p)]
    assert [p.name for p in builders] == ["tokenize.py"]

    # No other module may hand-assemble prompt + separator + steps.
    for path in SOURCES:
        if path.name in ("tokenize.py", "counterfactual.py"):
            continue        # counterfactual.py builds its prefix THROUGH build_sequence
        code = _code_lines(path)
        assert "sep_id]" not in code.replace("[sep_id]", ""), path.name


def test_no_centroid_of_terminals():
    """Root cause D: averaging terminals in latent space collapses onto the population mean
    and the distance becomes an atypicality detector. `mean OF distances`, never `distance to
    a mean` (§7.7)."""
    for path in SOURCES:
        code = _code_lines(path)
        for bad in ("g_mean", "goal_centroid", "terminals.mean(", "psi_term.mean("):
            assert bad not in code, f"{path.name}: {bad}"


def test_no_flash_attn_or_crm_stack_dependency():
    """§13 / PLAN: flash-attn has no cp312/cu128/sm_120 wheel, and CRM's pins (torch 2.6,
    vllm, deepspeed, trl) cannot run on sm_120 at all."""
    banned = ("flash_attn", "import deepspeed", "import vllm", "from trl", "import verl")
    for path in SOURCES:
        code = _code_lines(path)
        for bad in banned:
            assert bad not in code, f"{path.name}: {bad}"
    requirements = (REPO_ROOT / "requirements.txt").read_text()
    for bad in ("flash-attn\n", "deepspeed", "vllm", "trl>", "verl"):
        assert not re.search(rf"^{re.escape(bad)}", requirements, re.M), bad


def test_torch_dtype_kwarg_is_not_used():
    """transformers >= 4.56 takes `dtype=`; `torch_dtype=` is deprecated and goes away in v5.
    CRM uses the old name; we do not."""
    offenders = [p.name for p in SOURCES if "torch_dtype" in _code_lines(p)]
    assert not offenders, offenders


def test_automodel_not_automodelforcausallm():
    """§6.2: token classification over hidden states, no LM head, no vocab-sized logits
    anywhere (Qwen's vocab is ~151k)."""
    for path in SOURCES:
        assert "AutoModelForCausalLM" not in _code_lines(path), path.name
