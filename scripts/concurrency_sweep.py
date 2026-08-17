"""Find the concurrency at which the endpoint stops giving more throughput.

Deliberately does NOT use call_with_retries: a retry would hide the 429/5xx we are looking
for. Every failure is recorded raw. Latency is per-request wall clock, so a server that
queues shows up as rising p50 with flat throughput -- that is the saturation signature.
"""
import argparse, importlib.util, json, statistics, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gen_cf", ROOT / "scripts" / "generate_counterfactuals.py")
gen = importlib.util.module_from_spec(spec)
sys.modules["gen_cf"] = gen
spec.loader.exec_module(gen)

# The endpoint is a PARAMETER, not a constant. It was `secret.txt` + DEFAULT_MODEL_OPENAI
# until 2026-08-09, which meant the only way to sweep a second provider was to edit the file
# -- and a sweep is worth exactly as much as the configuration it ran under, so it has to
# name that configuration in its own output.
_cli = argparse.ArgumentParser(description=__doc__)
_cli.add_argument("--base-url", default="")
_cli.add_argument("--api-key", default="")
_cli.add_argument("--secret", default=str(ROOT / "secret.txt"))
_cli.add_argument("--model", default=gen.DEFAULT_MODEL_OPENAI)
_cli.add_argument("--max-tokens", type=int, default=24000)
_cli.add_argument("--temperature", type=float, default=0.7)
_cli.add_argument("--no-thinking", action="store_true")
_cli.add_argument("--timeout", type=float, default=1800.0)
_cli.add_argument("--levels", default="1,2,4,6,8", help="comma-separated concurrencies")
_cli.add_argument("--reqs-per-level", type=int, default=0, help="0 = 2 x concurrency")
_cli.add_argument(
    "--cooldown", type=float, default=0.0,
    help="seconds to idle BETWEEN levels. Needed on a TOKENS-PER-MINUTE-limited endpoint "
         "(Groq's free tier is 6,000 TPM): without it, level N+1 inherits level N's spent "
         "budget and measures the bucket rather than the concurrency. 70 is one refill.",
)
_cli.add_argument("--budget", type=float, default=5400.0, help="total wall-clock seconds")
_cli_args = _cli.parse_args()

args = argparse.Namespace(
    base_url=_cli_args.base_url, api_key=_cli_args.api_key, secret=_cli_args.secret,
    model=_cli_args.model, max_tokens=_cli_args.max_tokens,
    temperature=_cli_args.temperature, positives=3, negatives=6,
    no_thinking=_cli_args.no_thinking, timeout=_cli_args.timeout, retries=0,
)
base_url, api_key = gen.read_endpoint(args)
items = [json.loads(l) for l in (ROOT / "data/cf/cf_items.jsonl").read_text().splitlines() if l.strip()]
print(f"# endpoint = {base_url}  model = {args.model}  max_tokens = {args.max_tokens}", flush=True)


# Use the SAME response_format the real pipeline negotiates. Without it the first sweep got
# bad_stop on 10/16 -- guided decoding is part of what we are timing, so measuring without it
# measures a configuration nobody runs.
RESPONSE_FORMAT, _ = gen.negotiate_response_format(base_url, api_key, items[0], args)
print(f"# response_format = {json.dumps(RESPONSE_FORMAT)[:80]}", flush=True)


def one(item, timeout):
    t0 = time.time()
    try:
        response = gen.post_chat(base_url, api_key, gen.build_body(item, args, RESPONSE_FORMAT), timeout)
        text, finish = gen.response_text(response)
        usage = response.get("usage", {}) or {}
        ok = "ok" if gen.extract_json(text) is not None else f"bad_{finish}"
        return ok, time.time() - t0, usage.get("completion_tokens", 0)
    except Exception as exc:                                    # noqa: BLE001
        code = getattr(exc, "code", None)
        return f"http_{code}" if code else type(exc).__name__, time.time() - t0, 0


def level(c, n, timeout):
    """Latency is kept PER STATUS. A RemoteDisconnected has two candidate causes -- the
    server shedding load, or a fixed proxy idle timeout that long generations cross -- and
    they want opposite fixes. Tightly clustered disconnect latencies mean a timeout wall;
    scattered ones mean overload. One aggregate p50 cannot tell them apart."""
    work = [items[i % len(items)] for i in range(n)]
    status, lat, by_status, toks = Counter(), [], {}, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as pool:
        futures = [pool.submit(one, it, timeout) for it in work]
        for f in as_completed(futures):
            s, dt, tk = f.result()
            status[s] += 1
            lat.append(dt)
            by_status.setdefault(s, []).append(round(dt, 1))
            toks += tk
    wall = time.time() - t0
    lat.sort()
    # GOODPUT, not throughput. `items_per_min` counts a 429 as a completed request, so on a
    # rate-limited endpoint it RISES with concurrency while the useful work falls -- the sweep
    # of 2026-08-08 already carried that caveat on its 16 row as a footnote. `ok_per_min` is
    # the number to read; the two agree exactly when nothing fails.
    n_ok = status.get("ok", 0)
    return {
        "concurrency": c, "requests": n, "wall_s": round(wall, 1),
        "ok": n_ok,
        "ok_per_min": round(n_ok / wall * 60, 2),
        "items_per_min": round(n / wall * 60, 2),
        "tok_per_s": round(toks / wall, 1),
        "p50_s": round(statistics.median(lat), 1),
        "p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
        "status": dict(status),
        "latency_by_status": {k: sorted(v) for k, v in by_status.items()},
    }


if __name__ == "__main__":
    budget = _cli_args.budget
    levels = [int(x) for x in _cli_args.levels.split(",") if x.strip()]
    started = time.time()
    rows = []
    for i, c in enumerate(levels):
        if time.time() - started > budget:
            print(f"# budget spent, stopping before concurrency={c}", flush=True)
            break
        if i and _cli_args.cooldown:
            print(f"# cooldown {_cli_args.cooldown:.0f}s before concurrency={c}", flush=True)
            time.sleep(_cli_args.cooldown)
        n = _cli_args.reqs_per_level or 2 * c
        row = level(c, n, timeout=_cli_args.timeout)
        rows.append(row)
        print(json.dumps(row), flush=True)
    print("\n concurrency  reqs    ok  wall_s  ok/min  items/min  tok/s   p50_s  p95_s  status")
    for r in rows:
        print(f"{r['concurrency']:>11}  {r['requests']:>4}  {r['ok']:>4}  {r['wall_s']:>6}  "
              f"{r['ok_per_min']:>6}  {r['items_per_min']:>9}  {r['tok_per_s']:>6}  "
              f"{r['p50_s']:>6}  {r['p95_s']:>6}  {r['status']}")
