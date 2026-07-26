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
            wandb.init(project=wandb_project, name=run_name, config=dict(config or {}))

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
            ("probe03/gap", "gap"),
            ("probe01/questions_in_batch", "Q"),
            ("step/distinct_z", "z"),
        ]
        parts = [f"step {step:>6}"]
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
