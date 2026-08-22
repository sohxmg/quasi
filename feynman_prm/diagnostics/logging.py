"""JSONL + console logging, with wandb wired but optional (PLAN 'Decided in this planning
session').

Every loss curve is logged SEPARATELY from step 1 (§10 #11): the terms were never designed
to be additive and their gradient scales are uncharacterised.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional


class Progress:
    """A rate + ETA line for long silent stretches, printed on a TIME budget not a count.

    Phase 2's `build_cache` pushes ~34M tokens through the backbone -- 11 to 38 minutes --
    and printed nothing at all while it did, which is indistinguishable from a hang and was
    read as one. Anything between two log lines that can outlast a coffee needs one of these.

    Ticking on elapsed time rather than every N items means the line stays readable whether
    the loop does 3 items a second or 3,000, with no per-loop tuning.
    """

    def __init__(self, label: str, total: int, every_seconds: float = 15.0):
        self.label, self.total, self.every = label, max(total, 0), every_seconds
        self.done = 0
        self.t0 = self.last = time.time()
        print(f"[{label}] starting, {self.total:,} to go", flush=True)

    def advance(self, n: int = 1) -> None:
        self.done += n
        now = time.time()
        # Always print the final line, however recently the last one went out.
        if now - self.last < self.every and self.done < self.total:
            return
        self.last = now
        elapsed = now - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        pct = 100.0 * self.done / self.total if self.total else 100.0
        # A caller whose `total` is slightly off should get a pinned bar, not "101.1%" and a
        # negative ETA -- the display must never be the thing that looks broken.
        remaining = max(self.total - self.done, 0)
        eta = remaining / rate if rate > 0 else float("nan")
        pct = min(pct, 100.0)
        print(
            f"[{self.label}] {self.done:,}/{self.total:,} ({pct:5.1f}%)  "
            f"{rate:8.1f}/s  elapsed {elapsed / 60:6.1f}m  eta {eta / 60:6.1f}m",
            flush=True,
        )


class RunLogger:
    def __init__(
        self,
        out_dir: str | Path,
        run_name: str,
        use_wandb: bool = False,
        wandb_project: str = "feynman-prm",
        config: Optional[Mapping[str, Any]] = None,
    ):
        self.dir = Path(out_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.dir / "metrics.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self._metrics = self.metrics_path.open("a")
        self._events = self.events_path.open("a")
        self._t0 = time.time()
        self._wandb = None
        if use_wandb:
            # `log.wandb: true` was asked for explicitly, so a missing package is a hard error.
            # It used to warn on stderr and train on regardless -- three hours of GPU time with
            # nothing in the dashboard and one line scrolled past in tmux. Same reasoning as the
            # strict config parse: a silently ignored config value is old bug B4.
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError(
                    "log.wandb is true but wandb is not installed. Either "
                    "`pip install wandb && wandb login`, or run with --set log.wandb=false "
                    "(metrics.jsonl is written either way)."
                ) from exc

            self._wandb = wandb
            run = wandb.init(project=wandb_project, name=run_name, config=dict(config or {}))
            # The dashboard URL is printed by wandb itself and then scrolls past. On a rented
            # box the console is a log stream you may not be watching, so persist it: it is
            # the only pointer from an artifact directory back to the live curves.
            url = getattr(run, "url", None)
            if url:
                self._events.write(json.dumps({"event": "wandb/run", "url": url}) + "\n")
                self._events.flush()
                print(f"[wandb/run] {url}", flush=True)

    # ---- writing ----

    def log(self, step: int, metrics: Mapping[str, float], console: bool = False) -> None:
        record = {"step": step, "elapsed_s": round(time.time() - self._t0, 2), **dict(metrics)}
        self._metrics.write(json.dumps(record, default=float) + "\n")
        self._metrics.flush()
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=step)
        if console:
            print(self.format_console(step, metrics), flush=True)

    def event(self, name: str, payload: Mapping[str, Any]) -> None:
        """One-off structured records: launch asserts, probes, gates, checkpoints."""
        record = {"event": name, "elapsed_s": round(time.time() - self._t0, 2), **dict(payload)}
        self._events.write(json.dumps(record, default=_jsonable) + "\n")
        self._events.flush()
        print(f"[{name}] " + json.dumps(dict(payload), default=_jsonable), flush=True)

    @staticmethod
    def format_console(step: int, metrics: Mapping[str, float]) -> str:
        keys = [
            ("loss/total", "L"),
            ("nce/loss", "nce"),
            ("invariance/loss", "inv"),
            ("backup/loss", "bkp"),
            ("step/loss", "step"),
            ("good/loss", "good"),
            ("probe03/gap", "gap"),
            # The console line is what gets watched in tmux for hours, so the two numbers
            # §7.12 exists to move sit on it: the good-step tail, and the gap it must not
            # cost. `above` is the fraction of good steps over target -- read it, not the
            # mean, which stayed at +0.240 through a whole run that capped F1 at 0.456.
            ("good/above_target_fraction", "above"),
            # §16.26's own gauge, on the console because it is the statistic (7) L_term (§7.13)
            # exists to move and it is watched at lambda_term = 0.0 to decide whether the term
            # is worth turning on at all. `gate/within_question_terminal_spread` is the same
            # quantity but only runs in phase 2; this one is per step.
            ("term/within_question_terminal_spread", "spread"),
            ("probe01/questions_in_batch", "Q"),
            ("step/distinct_z", "z"),
            # ---- phase 2 (§7.7). It logs per EPOCH, and 20 epochs is the whole curve, so
            # `var` earns its place on the line: diagnostic #6 fires when the head gives up on
            # conditioning and learns one global anchor, and near-zero variance is how that
            # looks while `goal/loss` falls perfectly smoothly.
            ("goal/loss", "goal"),
            ("goal/pred_variance", "var"),
            ("goal/seconds", "s"),
        ]
        parts = [f"step {step:>6}"]
        if "goal/epoch" in metrics:
            # Phase 2's x-axis is the epoch, so labelling it "step" would misread by
            # `batches_per_epoch`. Both numbers go on the line.
            parts = [f"epoch {int(metrics['goal/epoch']):>4}"]
            if "goal/optimizer_step" in metrics:
                parts.append(f"step={int(metrics['goal/optimizer_step'])}")
        for key, label in keys:
            if key in metrics:
                parts.append(f"{label}={metrics[key]:+.4f}")
        return "  ".join(parts)

    def close(self) -> None:
        self._metrics.close()
        self._events.close()
        if self._wandb is not None:
            self._wandb.finish()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def read_metrics(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
