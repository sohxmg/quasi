"""The QRL-only knobs, strict-parsed exactly like `feynman_prm/config.py` (old bug B4: a
config value silently ignored and a library default used instead).

**Everything else comes from `config/default.yaml`, unchanged.** That is the entire point of
this variant: the dataset, the selection, the seed, the batch stream, the tokenisation, the
backbone, the LoRA, the optimiser, the schedule, the CF corpus and the eval protocol are
shared by construction, so the only difference between the QRL row and the Feynman-PRM row is
the objective (and, deliberately, the head -- see README.md §2).

> **Naming trap.** `losses.*` in `config/default.yaml` -- `lambda_cf`, `lambda_step`,
> `lambda_good`, `zeta`, ... -- is Feynman-PRM's fixed-weight loss set and **not one of those
> keys is read by anything in `qrl_prm/`**. QRL's weights are DUAL VARIABLES: they are
> learned, they live on `local.raw` / `path.raw` / `cf.raw` in `lagrange.py`, and their
> initial values are `init_lagrange_local` / `init_lagrange_path` / `init_lagrange_cf` here.
> `train.py` refuses to start if a
> `--set losses.*` override is passed, because that override would be silently inert --
> exactly bug B4's shape.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Reusing the strict parser rather than re-typing it: unknown key -> hard error with the
# closest legal name, wrong type -> hard error with the path. `_resolve_annotations` is
# required because `from __future__ import annotations` makes every `f.type` a string.
from feynman_prm.config import ConfigError, _build, _resolve_annotations, apply_override


@dataclass(frozen=True)
class QRLConfig:
    """QRL's own defaults where QRL has one, and a derivation where it does not.

    Line citations are into `quasimetric-rl/quasimetric_rl/modules/quasimetric_critic/`.
    """

    # `losses/global_push.py:26`. QRL's comment: "should be greater than most GT distances
    # between sampled pairs". Online default 15, offline maze2d 500.
    #
    # 25, and the round trip to 5 and back is the reasoning. The offset was FIRST blamed for
    # `loj243n4`'s runaway on a ratio argument -- 25 is ~10x upstream's offset/horizon and
    # offset/mean-pair calibration -- and dropped to 5. That argument was wrong, because it
    # assumed the offset sets the scale. It does not: `init_lagrange_path` does. Lowering it
    # bought ~26% on the adjacent-step mean (offset 8 vs 25 at matched lambda) and then
    # lambda = 1.5 took that mean from 9.15 to 1.40 by itself.
    #
    # What the offset actually sets is the DYNAMIC RANGE available for separation, and 5 is
    # too little of it. With lambda pinning adjacent steps near 1.4, the push wants to expand
    # every other pair, and the offset is where it gives up. At 5 (`1wbpyf2g`)
    # push_saturated_frac ran 0.09 -> 0.45 between steps 40 and 120 and was tracking to cross
    # 0.9 by ~step 250: the flat-region failure train.sh:63 warns about, where the objective
    # stops maximising anything. At 25 (`loj243n4`) it sat at a steady 0.12-0.22 for 1,460
    # steps and cross-question distances reached 10-16 rather than capping at 6.6.
    #
    # The structure metric agrees. Bucketing `loj243n4` by lambda level, same_traj/cross_q
    # falls monotonically -- 0.875 at lambda < 0.5, 0.834 at 1.5-2.0, 0.810 at 2.5-3.0, with
    # its best individual readings 0.63-0.68 at lambda 2.2-3.2 -- all of that at offset 25. At
    # matched lambda 1.5-2.0 the offset-5 run reads 0.869 against that run's 0.834.
    #
    # Read `qrl/push_saturated_frac` at launch AND at step ~300. Near 1.0 at init means the
    # offset is below the untrained distance scale (head-specific: iqe 2.995, full_mrn 6.723)
    # and the push has no gradient. RISING through the run means it is too small for the scale
    # lambda has settled on.
    softplus_offset: float = 25.0

    # `losses/global_push.py:23`. Smaller => smoother. QRL's default, unchanged.
    softplus_beta: float = 0.1

    # `losses/local_constraint.py:29`. One observed reasoning step is the unit of the metric,
    # so d(psi_z, psi_{z+1}) -> 1.0 is what `qrl/local_dist_mean` should pin to. THAT is the
    # ruler IMPLEMENTATION.md §9 says decays under the fixed-weight losses, and here it is a
    # constraint rather than a target a soft loss trades away.
    #
    # It is also the SLOPE of the path constraint's target: a gap of k observed steps is
    # bounded by `k * step_cost`. k = 1 and k >= 2 are the SAME inequality read at different
    # gaps, but they are enforced as two DISJOINT constraints under two multipliers -- see
    # `init_lagrange_local` below for the measurement that forced the split.
    step_cost: float = 1.0

    # `losses/local_constraint.py:22`, QRL's default, unchanged. The tolerance on the k = 1
    # constraint: `E_{j=i+1}[relu(d(s_i, s_j) - step_cost)^2] <= epsilon_local^2`, so the
    # adjacent-step ruler is allowed to sit at `step_cost + epsilon_local = 1.25`.
    #
    # This is the number `qrl/local_dist_mean` is read against and it is upstream's own.
    epsilon_local: float = 0.25

    # The same tolerance on the k >= 2 constraint:
    # `E_{2 <= j-i <= path_max_gap}[relu(d(s_i, s_j) - (j - i) * step_cost)^2] <= epsilon_path^2`.
    #
    # Deliberately equal to `epsilon_local`, and deliberately a SEPARATE key. The deviation is
    # ABSOLUTE, not per-step (see `path_max_gap` for why, and for the cap that makes that
    # choice safe), so at gap k the budget `k * step_cost + epsilon_path` allows the same
    # 0.25 of total overshoot a single adjacent step gets. Setting the two apart is how you
    # would say "a longer sub-path deserves more slack" -- a claim about the data nobody has
    # measured, so they start equal.
    #
    # The constraint is ONE-SIDED at every k (`(d - k * step_cost).relu()`): a sub-path
    # SHORTER than its observed length is free, because a shortcut is real information about
    # the metric and penalising it would be asserting the observed trace is optimal.
    epsilon_path: float = 0.25

    # No QRL counterpart -- CF invariance is this project's term. It is the radius of the
    # equivalence ball two meaning-preserving wordings of one step may sit in, and it must be
    # far below the ruler the verdict is read against: the old margin ruler is
    # `2 * log 2 ~= 1.386` (§7.6), so 0.2 keeps a paraphrase ~7x too small to flip a step's
    # verdict. Both directions are constrained, so by the triangle inequality the class
    # diameter is bounded by 2 * epsilon_cf.
    epsilon_cf: float = 0.2

    # SPLIT THREE WAYS, and no longer QRL's shared default.
    # `losses/local_constraint.py:31` starts its multipliers at 0.01 because upstream trains
    # for 2e5 steps (`offline/main.py:41`) and can afford the ramp. This run gets ~1,464, and
    # the dual is a rate-limited integrator: AdamW normalises by running gradient magnitude,
    # so `raw` advances at ~lagrange_lr per step no matter how large the violation is.
    # Measured on run `loj243n4`: raw moved 0.0141/step over steps 1-120 and 0.0024/step over
    # 1000-1460, while the violation it was integrating ranged over 172.6 -> 0.16. The
    # multiplier is a CLOCK, not a controller, and the only way to buy back the transient is
    # to start it where it was going to end up.
    #
    # That cost 600 steps -- 41% of the run -- to climb from 0.01 to the ~1.33 where
    # `local_dist_mean` finally turned over, and during those steps the push term ran unopposed
    # and inflated every distance together: the same_traj/cross_question ratio was 0.851 at step
    # 1 (untrained) and 0.876 at step 1460, i.e. training made the structure slightly WORSE
    # while multiplying the scale by 6.
    #
    # ---- 5.0, and why the k = 1 constraint is armed hardest ----
    #
    # THIS IS THE RULER. `qrl/local_dist_mean` -> step_cost is the curve the run is steered by,
    # and it is the one upstream's `local_constraint.py` holds.
    #
    # ANALYTIC: at the epsilon boundary d = 1.25 the push and constraint gradients balance when
    # `2 * lambda * (d - 1) == sigmoid(beta * (offset - d))`, giving
    # `lambda = sigmoid(0.1 * 23.75) / 0.5 = 1.83` at offset 25.
    #
    # EMPIRICAL, and the number that wins: on `loj243n4` (offset 25) lambda_local SETTLED at
    # 3.479 by step 1460 -- 1.90x the analytic estimate -- and `local_dist_mean` was STILL at
    # 1.482 rather than 1.25. 3.479 was therefore a lower bound that had not converged when the
    # run ended, not an equilibrium. On top of that, `pos_neg_push` did not exist when 3.479
    # was measured and now carries ~34% of the objective, all of it expansion pressure on psi.
    # 5.0 is 3.479 with that headroom.
    #
    # Starting HIGH is the safe direction:
    #   * the constraint is ONE-SIDED -- `(d - step_cost).relu().square()` -- so it costs
    #     exactly nothing below the step cost. A large lambda_local cannot crush distances
    #     below 1.0; it can only stop overshoot. There is no collapse failure mode on that side.
    #   * it applies ONLY to same-trajectory adjacent pairs, while the push applies to every
    #     (state, goal) pair. So a high lambda_local pins observed steps near their step count
    #     and leaves the push free to expand everything else -- which IS the
    #     same_traj/cross_question gap the 0.01 run failed to produce.
    #   * overshoot self-corrects: the violation goes negative and the multiplier descends at
    #     the same rate. Undershoot does not -- it drives distances past the offset into the
    #     flat region where the objective has no gradient left to fix them with.
    init_lagrange_local: float = 5.0

    # ---- 3.0, and why k >= 2 is armed LOWER than k = 1 ----
    #
    # The k >= 2 rows were merged into the k = 1 mean once, in run `0lcrduzl`, and the merge
    # is what this split undoes. Both constraints are means of squared deviations, and a mean
    # divides by pairs that a ONE-SIDED constraint leaves at exactly zero. Measured at step 1
    # of `0lcrduzl`, against `1wbpyf2g` on the identical seed-42 batch:
    #
    #     set        pairs   violating       sum(dev^2)   mean
    #     k = 1        489   486  (99.4%)         969.4   1.982
    #     k >= 2     3,119   418  (13.4%)         226.3   0.073
    #     pooled     3,608   904  (25.1%)       1,195.7   0.331
    #
    # The k = 1 rows carried 81% of the violation and received 489/3608 = 13.5% of the weight.
    # `local_dist_mean` went 2.263 -> 3.009 over 20 steps where the same lambda on the k = 1
    # constraint alone had taken it 2.263 -> 1.390. Splitting the two means is what fixes that.
    #
    # (The "N cancels" argument that justified the merge is right at EQUILIBRIUM and wrong in
    # the TRANSIENT: `(2N/S) * 2*lambda*eps / N` cancels only if every pair sits at the
    # boundary. Slack pairs contribute 0 to the numerator and full weight to N.)
    #
    # ANALYTIC equilibrium, offset 25, absolute deviation, at the epsilon boundary:
    #     k = 1, d = 1.25 -> sigmoid(0.1 * 23.75) / 0.5 = 1.83
    #     k = 2, d = 2.25 -> sigmoid(0.1 * 22.75) / 0.5 = 1.81
    #     k = 3, d = 3.25 -> sigmoid(0.1 * 21.75) / 0.5 = 1.80
    # They agree to ~1%, because the deviation at the boundary is `epsilon` in every case and
    # the push gradient barely moves across the gap range. Scaling 1.81 by the 1.90x
    # empirical-over-analytic factor `loj243n4` measured for k = 1 gives 3.44 -> 3.0.
    #
    # It does NOT get lambda_local's 5.0 uplift, and this is the reason: that uplift is for
    # "3.479 had not converged after 1,464 steps", a measurement about k = 1 ONLY. The k >= 2
    # constraint is in the opposite situation -- with `path_max_gap = 3` its step-1 violation
    # is ~+0.202 against the k = 1 constraint's +1.920, i.e. 9.5x less violated. Arming a
    # nearly-satisfied backstop harder than the badly-violated ruler has nothing behind it.
    #
    # And it IS a backstop. `d(s_i, s_j) <= (j - i) * step_cost` is implied by k = 1 through
    # the triangle inequality; the k >= 2 rows exist only because k = 1 is enforced softly and
    # a mean averages a heavy tail away (`loj243n4`: adjacent mean 1.418, adjacent max 6.721,
    # gap-2 pairs at 11.162 where 2.85 was implied). Under-arming a backstop is cheap -- the
    # dual climbs ~0.01/step and reaches 5.0 by ~step 200 on its own. Under-arming the ruler
    # is what cost `loj243n4` 600 steps.
    init_lagrange_path: float = 3.0

    # RAISED from QRL's 0.01, on the same "initialise at the destination" argument and a
    # measurement that reads the opposite way to how it was first reported.
    #
    # `loj243n4` was cited as evidence that the CF constraint "converges on its own inside the
    # budget": it ended `cf_sq_dev` 0.128 against `epsilon_cf ** 2 = 0.04` and `cf_violation`
    # 0.088, from an init violation of 3.62. True -- but its FINAL lambda_cf was 5.336. Getting
    # there from 0.01 means moving `raw` from -4.60 to +5.34 at ~lagrange_lr per step: ~994
    # steps, 68% of the run. The constraint converged BARELY, and the clock ate two thirds of
    # the budget doing it. That is `init_lagrange_local`'s pathology, unfixed.
    #
    # It is now worse, because `cf_pos_neg_push_weight` is new and is by construction the
    # direct negation of this constraint (see that field): it pushes rewrites of one step
    # apart while this pulls them together, and it carries ~34% of the objective. The
    # equilibrium lambda_cf is therefore ABOVE 5.336, still starting from 0.01. Measured on
    # probe `0lcrduzl`, `cf_violation` reached 7.411 by step 20 against `loj243n4`'s 5.924 on
    # the identical batch -- 25% higher and rising 1.6x faster.
    #
    # 3.0 rather than 5.336, because this constraint is NOT one-sided the way the path pair
    # are. `cf_sq_dev` is `mean(relu(d)^2)` against a target of 0 (`loss.py::cf_terms`), so
    # the gradient is always live and a large enough lambda_cf CAN collapse an equivalence
    # class to a point. 5.336 is empirically safe -- `loj243n4` sat there with `cf_dist_mean`
    # 0.236, above `epsilon_cf = 0.2` -- but that was without `pos_neg_push`. 3.0 banks ~800
    # steps of the climb and leaves the dual to finish it, which is the direction that
    # self-corrects.
    #
    # The three multipliers stay SEPARATE: they integrate constraints with different violation
    # scales and different convergence behaviour, and one shared dual would be steered entirely
    # by the largest.
    init_lagrange_cf: float = 3.0

    # `losses/__init__.py:47`: `lagrange_mult_optim: AdamWSpec.Conf(lr=1e-2)`. Constant --
    # the dual variable must be free to track the constraint for the whole run, so it does
    # NOT ride the primal's cosine decay.
    lagrange_lr: float = 1e-2

    # Weight on the CF-NEGATIVE push term: `softplus(offset - d(psi(prefix + neg), psi_g))` over
    # same-question goal columns. Broken states are pushed AWAY from goals, in the direction
    # eval queries (broken state as source). 0.0 makes the term an exact zero with its keys
    # still logged.
    cf_neg_push_weight: float = 1.0

    # Weight on the POSITIVE-vs-NEGATIVE push:
    # `softplus(offset - d)` over both directions of every (class member, negative) pair
    # WITHIN one CF example, where the class members are the anchor and its positives.
    #
    # This is the direct negation of the CF constraint and the reason it exists. The CF
    # constraint says a meaning-preserving rewrite of step i is the SAME point as the
    # original, in both directions. Nothing until now said that a meaning-BREAKING rewrite of
    # the same step is a DIFFERENT point -- the negatives were only pushed away from goal
    # columns, which is a claim about reaching the answer, not about the step itself. A metric
    # can satisfy `d(neg, goal)` large and still place the broken rewrite on top of the
    # correct one, and then a paraphrase-sized perturbation flips the verdict.
    #
    # BOTH directions, exactly as the CF constraint binds both: you cannot reach a broken step
    # from a correct one, and you cannot recover from a broken step back to the correct one.
    # SAME EXAMPLE only -- a negative and a positive from different examples are rewrites of
    # different steps, so their distance is not a statement about anything.
    #
    # 0.0 makes the term an exact zero with `qrl/pos_neg_push_*` still logged.
    cf_pos_neg_push_weight: float = 1.0

    # Largest observed gap `j - i` the PATH constraint is applied at. The path set is
    # `2 <= j - i <= path_max_gap`; 0 = unlimited, the same reading `push_chunk_cols: 0` has.
    # k = 1 is never in this set -- it is `local_violation`, under its own multiplier.
    #
    # `path_max_gap = 1` therefore makes the path set EMPTY, which is the local-only ablation
    # and is handled by the same exact-zero path an empty batch takes.
    #
    # **3, because the deviation is ABSOLUTE and an absolute deviation makes long gaps
    # dominate the mean.** At gap k a metric inflated to `d ~ 3k` deviates by `2k`, so
    # `dev^2 ~ 4k^2` and one gap-20 pair carries 100x what a gap-2 pair does. Left uncapped
    # (`path_gap_max` was 20 on probe `0lcrduzl`), the k >= 2 mean would be steered by the
    # longest sub-paths -- which is exactly the "the constraint controlled the mean, not the
    # tail" failure this constraint exists to fix, relocated one level up. It would also aim
    # the term at the wrong gaps: `loj243n4`'s damage was at `probe16/goal_offset_mean` 2.010,
    # i.e. k ~ 2, not k = 20.
    #
    # The cap also concentrates the violation, which is what makes `init_lagrange_path`
    # meaningful. Step 1 of `0lcrduzl`, k >= 2 uncapped: violation +0.010 -- the term is
    # already satisfied and the multiplier has nothing to integrate. Capped at k <= 3: ~+0.202,
    # 20x more. Same pairs, and the ones that were diluting the signal are the ones dropped.
    #
    # The alternative to a cap is a PER-STEP deviation, `((d - k*c).relu() / k)^2`, which makes
    # every gap contribute in the same units and needs no cap at all. That was considered and
    # not taken: it changes what `epsilon_path` asserts, and the absolute form is upstream's.
    # `qrl/path_ratio_mean` logs the per-step quantity either way.
    path_max_gap: int = 3

    # Padded-token budget for the CF variant forward: `sequences x longest`, the shape the GPU
    # actually pays for. **This is the knob `data.cf_max_per_batch` used to be.** Under
    # `cf_phi` a variant cost an embedding lookup and an MLP; under `cf_encode.py` it is a full
    # sequence through the backbone, so 12 examples x ~8 rewrites is ~96 sequences against a
    # main batch of ~51. 16384 is half `sampling.max_padded_tokens`, which is what the main
    # batch is capped at. Whole EXAMPLES are dropped off the tail of the seeded order once the
    # budget is met -- unbiased in length, and counted in `cf/examples_dropped_budget`.
    cf_encode_max_tokens: int = 16384

    # 0 = build the (S, C) push matrix in one shot. A positive value splits it into chunks of
    # that many GOAL COLUMNS and accumulates the mean exactly (weighted by each chunk's column
    # count). It lowers the TRANSIENT peak of the distance's internals -- IQE materialises
    # several (S, C, D/k, 2k) fp32 tensors -- not the autograd graph, which is the same size
    # either way. Reach for it only if `launch/memory_probe` says the card cannot take the
    # one-shot form; tests pin chunked == unchunked.
    push_chunk_cols: int = 0

    def __post_init__(self) -> None:
        if self.softplus_offset <= 0.0:
            raise ConfigError(
                f"qrl.softplus_offset must be > 0, got {self.softplus_offset}. It is the point "
                "the push term stops rewarding distance; at 0 there is no push at all."
            )
        if self.softplus_beta <= 0.0:
            raise ConfigError(f"qrl.softplus_beta must be > 0, got {self.softplus_beta}")
        if self.step_cost <= 0.0:
            raise ConfigError(
                f"qrl.step_cost must be > 0, got {self.step_cost}. It is the unit the whole "
                "metric is expressed in -- `qrl/local_dist_mean` is read against it, and it "
                "is the slope of the path constraint's per-gap target."
            )
        if self.epsilon_local <= 0.0:
            raise ConfigError(
                f"qrl.epsilon_local must be > 0, got {self.epsilon_local}. At 0 the constraint "
                "is unsatisfiable (the violation can never go negative) and the multiplier "
                "diverges instead of stabilising."
            )
        if self.epsilon_path <= 0.0:
            raise ConfigError(
                f"qrl.epsilon_path must be > 0, got {self.epsilon_path}. Same failure as "
                "epsilon_local, on the k >= 2 constraint."
            )
        if self.epsilon_cf <= 0.0:
            raise ConfigError(
                f"qrl.epsilon_cf must be > 0, got {self.epsilon_cf}. Same failure as "
                "epsilon_local: a zero-radius ball is unreachable and lambda_cf would climb "
                "for the whole run with nothing to report."
            )
        for _name, _v in (
            ("init_lagrange_local", self.init_lagrange_local),
            ("init_lagrange_path", self.init_lagrange_path),
            ("init_lagrange_cf", self.init_lagrange_cf),
        ):
            if _v <= 0.0:
                raise ConfigError(
                    f"qrl.{_name} must be > 0, got {_v}. softplus_inv is undefined at 0, and "
                    "a multiplier that starts at exactly 0 has no gradient path back into the "
                    "primal on the first step."
                )
        if self.lagrange_lr <= 0.0:
            raise ConfigError(f"qrl.lagrange_lr must be > 0, got {self.lagrange_lr}")
        if self.cf_neg_push_weight < 0.0:
            raise ConfigError(
                f"qrl.cf_neg_push_weight must be >= 0, got {self.cf_neg_push_weight}. A "
                "negative weight would PULL broken states towards goals."
            )
        if self.cf_pos_neg_push_weight < 0.0:
            raise ConfigError(
                f"qrl.cf_pos_neg_push_weight must be >= 0, got {self.cf_pos_neg_push_weight}. "
                "A negative weight would PULL a meaning-breaking rewrite of a step onto the "
                "correct one, which is the exact failure the term exists to prevent."
            )
        if self.path_max_gap < 0:
            raise ConfigError(
                f"qrl.path_max_gap must be >= 0 (0 = unlimited), got {self.path_max_gap}. "
                "The path set is `2 <= gap <= path_max_gap`, so 1 makes it EMPTY -- that is "
                "the local-only ablation and it is legal."
            )
        if self.cf_encode_max_tokens <= 0:
            raise ConfigError(
                f"qrl.cf_encode_max_tokens must be > 0, got {self.cf_encode_max_tokens}. At 0 "
                "the budget would drop every CF example after the first and the CF CONSTRAINT "
                "-- one of the two constraints this objective is defined by -- would train on "
                "one class per batch."
            )
        if self.push_chunk_cols < 0:
            raise ConfigError(
                f"qrl.push_chunk_cols must be >= 0 (0 = one shot), got {self.push_chunk_cols}"
            )
        if self.epsilon_cf >= 2.0 * 0.6931471805599453:
            raise ConfigError(
                f"qrl.epsilon_cf = {self.epsilon_cf} is at or above the old margin ruler "
                "2*log 2 = 1.3863 (§7.6). An equivalence ball that wide lets a paraphrase move "
                "a step across the decision boundary, which is the exact property (4) L_CF "
                "exists to prevent. If this is deliberate, say so in the run's README row."
            )

    # ---- derived ----

    @property
    def local_target(self) -> float:
        """The k = 1 constraint's right-hand side, `epsilon_local^2`. `qrl/local_violation` is
        `mean(relu(d - step_cost)^2) - this`, so its SIGN is the readable quantity: negative
        means the constraint is satisfied and lambda_local should be falling."""
        return self.epsilon_local**2

    @property
    def path_target(self) -> float:
        """The k >= 2 constraint's right-hand side, `epsilon_path^2`. Same reading as
        `local_target`, for `qrl/path_violation`."""
        return self.epsilon_path**2

    @property
    def cf_target(self) -> float:
        """`epsilon_cf^2`. Same reading as `path_target`, for `qrl/cf_violation`."""
        return self.epsilon_cf**2

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["_derived"] = {
            "local_target": self.local_target,
            "path_target": self.path_target,
            "cf_target": self.cf_target,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False))


_resolve_annotations(QRLConfig)


def load_qrl_config(path: str | Path, overrides: list[str] | None = None) -> QRLConfig:
    """Parse `qrl_prm/config/qrl.yaml` strictly. `overrides` are `key=value` strings with the
    `qrl.` prefix ALREADY STRIPPED -- `train.py` partitions `--set` so that
    `--set qrl.cf_encode_max_tokens=8192` lands here and `--set run.name=x` lands on the
    Feynman config."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for assignment in overrides or []:
        apply_override(raw, assignment)
    return _build(QRLConfig, raw, "qrl")


def qrl_config_from_dict(data: dict[str, Any]) -> QRLConfig:
    """Same strictness, from an already-parsed mapping (checkpoints, tests)."""
    return _build(QRLConfig, {k: v for k, v in data.items() if k != "_derived"}, "qrl")


def load_qrl_config_from_checkpoint(checkpoint_dir: str | Path) -> QRLConfig:
    return qrl_config_from_dict(yaml.safe_load((Path(checkpoint_dir) / "qrl.yaml").read_text()))


def split_overrides(overrides: list[str]) -> tuple[list[str], list[str]]:
    """Partition `--set` assignments into (feynman, qrl). A `qrl.`-prefixed key has the prefix
    stripped; everything else passes through untouched.

    This is what keeps `config/default.yaml` free of a `qrl:` block it would have to declare
    (it is strict-parsed, so an undeclared key there is a hard error) while still giving the
    one-line ablations -- `--set qrl.cf_encode_max_tokens=8192`,
    `--set qrl.cf_neg_push_weight=0` -- a
    place to land.
    """
    feynman, qrl = [], []
    for assignment in overrides:
        if "=" not in assignment:
            raise ConfigError(f"override {assignment!r} is not of the form key.path=value")
        key = assignment.split("=", 1)[0].strip()
        if key == "qrl" or key.startswith("qrl."):
            qrl.append(assignment.split(".", 1)[1] if "." in key else assignment)
        else:
            feynman.append(assignment)
    return feynman, qrl


QRL_YAML = Path(__file__).resolve().parent / "config" / "qrl.yaml"


def field_names() -> tuple[str, ...]:
    return tuple(f.name for f in fields(QRLConfig))
