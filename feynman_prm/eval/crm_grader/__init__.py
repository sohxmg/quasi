"""CRM's Best-of-N answer grader, VENDORED VERBATIM. Do not improve it.

Locked #6 says we do not compare against other papers' numbers; the human overrode that on
2026-08-05 for BoN specifically. The whole point of a BoN comparison is that CRM's published
accuracies and ours are produced by the SAME judge, so the only thing differing between the
two rows of the table is which model reranked the candidates. A "better" grader here is a
confound, not an improvement -- if it marks one more answer correct than CRM's did, the
comparison silently stops being a comparison.

Provenance, copied 2026-08-05 from ../CRM/BoN/eval/:

    eval_grader.py      sha256 0040f5fa1880ccefc8e7373b1ec190e528e1b78a0812361e01cb71986455a119
    eval_normalizer.py  sha256 03d7bc00a215a8dcba379d9935cb228131ecd6f208b5027f2610d48c13b9356d
    eval_utils.py       sha256 a35a5031772bd5c0a4448acce9d1269a2c89f1204d385086453b2f67f06e5b1b

The hashes above are of CRM's originals, and the vendored files are byte-identical to them
except for THREE import lines, each marked at its site:

    eval_normalizer.py:242  `from eval_PQM_grader import extract_answer`
                            -> `from .eval_grader import extract_answer`
                            CRM's tree has no `eval_PQM_grader` module at all, so the shipped
                            file cannot be imported as-is. `extract_answer` lives in
                            `eval_grader.py:265` and is the only symbol taken, so this is a
                            rename of a broken import, not a substitution.
    eval_utils.py:4,5       `from eval_grader ...` / `from eval_normalizer ...`
                            -> relative imports, because these are a package here and a flat
                            directory on sys.path there.

`tests/test_bon.py` re-hashes the vendored files against those three edits and fails if any
other line moves. Re-run it after any `pip` upgrade that touches sympy: `math_equal` calls
`sympy.parsing.latex.parse_latex`, whose behaviour is the one external dependency the
comparison rests on.

Two things the grader does that are worth knowing before reading a BoN number:

  * gsm8k grading is `extract the LAST number in the response` (`eval_utils.py:23-28`) against
    the GSM-Plus gold answer. It is lenient in both directions and it is what CRM reported on.
  * math grading routes through `extract_math_answer_new` -> `math_equal`, which shells out to
    sympy with a `signal.alarm` timeout. It is not thread-safe and it is slow; the BoN harness
    grades only the ~500 SELECTED responses per (file, N, aggregator), never all 128 x 500.
"""

from .eval_utils import eval_gsm8k, eval_math_prm

__all__ = ["eval_gsm8k", "eval_math_prm", "assert_grader_environment", "latex_parser_available"]


def latex_parser_available() -> bool:
    """Is sympy's LaTeX parser actually usable here?

    `symbolic_equal` tries `[parse_expr, parse_latex]` inside a bare `except Exception: pass`
    (`eval_grader.py:236-244`), so a missing `antlr4-python3-runtime` does not raise -- it
    silently removes `parse_latex` from the chain and the grader falls back to `parse_expr`,
    which cannot read `\\frac{1}{2}`. **That is a grader that marks fewer math answers correct
    than CRM's did, with no error anywhere**, and it would land as a modelling result.
    """
    try:
        from sympy.parsing.latex import parse_latex

        parse_latex(r"\frac{1}{2}")
        return True
    except Exception:
        return False


def assert_grader_environment(data_name: str) -> None:
    """Fail before the GPU work, not after it.

    gsm8k grading is a regex over the last number (`eval_utils.py:23-28`) and touches no
    sympy at all, so it is exempt. The math path is the one that can degrade quietly.
    """
    if data_name == "gsm8k" or latex_parser_available():
        return
    raise RuntimeError(
        "sympy's LaTeX parser is not usable in this environment, so `math_equal` will grade "
        "with `parse_expr` alone and mark LaTeX answers wrong that CRM's grader accepted. Any "
        "math BoN number produced now would be lower than CRM's for a reason that has nothing "
        "to do with the reward model.\n"
        "    pip install 'antlr4-python3-runtime==4.11.*'\n"
        "Pass --allow-degraded-grader to proceed anyway; the result JSON records that you did."
    )
