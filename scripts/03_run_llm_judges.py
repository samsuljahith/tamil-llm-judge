"""Step 03 (objective O2): run the four LLM judges over the PARIKSHA pairs and direct responses.

What this script does, following the rules in AGENTS.md:
  * four judges, defined in config/judges.json (frontier, low_cost, sea_lion, base_model);
  * one FIXED rubric prompt per task type, so the only thing that changes between judges is the model;
  * every pairwise pair is judged in BOTH answer orders (A/B and B/A), and the swapped verdict is
    mapped back to the original A/B, so position bias can be measured later and cancelled;
  * temperature 0 where the model supports it (SEA-LION, Gemma base); for OpenAI reasoning models,
    which reject a custom temperature, it is omitted and logged as "unsupported", with the reasoning
    effort set to the lowest level the model accepts;
  * short outputs: the model must return JSON {"verdict": "A"|"B"|"tie", "reason": "<=30 words"},
    and a max output-token limit is set;
  * a 10 requests/minute limiter for the SEA-LION judge (its documented rate limit);
  * fully resumable: every call is appended to results/judgements/<task>__<judge>.jsonl, and on a
    restart the pairs already completed are skipped;
  * every call is logged with the prompt, the raw output, the parsed verdict, the model snapshot,
    a timestamp, input / reasoning / visible-output tokens, and the cost the API reports;
  * a budget guard: if the projected full-run cost exceeds MAX_BUDGET_USD it stops (never during a
    pilot, which is tiny by design).

Usage:
  python scripts/03_run_llm_judges.py --task tamil_pairwise --pilot 50
  python scripts/03_run_llm_judges.py --task hindi_pairwise
  python scripts/03_run_llm_judges.py --task tamil_direct

No API key is ever printed, logged or committed. Keys live only in .env (git-ignored) and are read
by the variable name given in config/judges.json. No annotator IDs or justification texts are read
or written here (GDPR data minimisation); only the merged, de-identified processed tables are used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "judges.json"
ENV_FILE = ROOT / ".env"
LOG_DIR = ROOT / "results" / "judgements"

# A verdict given when A and B are swapped maps back to the original orientation like this.
SWAP = {"A": "B", "B": "A", "tie": "tie"}
# The tasks the script can run, and the processed table each one reads.
TASKS = {
    "tamil_pairwise": ("pairs_tamil.csv", "tamil"),
    "hindi_pairwise": ("pairs_hindi_sample.csv", "hindi"),
    "tamil_direct": ("direct_tamil.csv", "tamil"),
}
# The full experiment judges every pair in both orders plus every direct response, per judge:
#   tamil_pairwise 2,002 x 2 + hindi_pairwise 1,000 x 2 + tamil_direct 280 = 6,284 calls per judge.
FULL_CALLS_PER_JUDGE = 2002 * 2 + 1000 * 2 + 280


# ---------------------------------------------------------------------------------------------
# Configuration, keys and input data
# ---------------------------------------------------------------------------------------------
def load_config() -> dict:
    """Read the fixed judge slate and family map written before any results were seen."""
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_keys() -> dict:
    """Read API keys from .env without putting them in the environment or printing them."""
    if not ENV_FILE.is_file():
        sys.exit(f"{ENV_FILE} not found. Copy .env.example to .env and fill in the keys.")
    return dotenv_values(ENV_FILE)


def load_prompts_and_responses() -> tuple[dict, dict]:
    """Return two lookups: (language, prompt_id) -> prompt text, and (language, prompt_id, model) -> response."""
    prompts = pd.read_csv(PROCESSED / "prompts.csv")
    responses = pd.read_csv(PROCESSED / "responses.csv")
    prompt_by_id = {(r.language, r.prompt_id): r.prompt for r in prompts.itertuples(index=False)}
    response_by_key = {(r.language, r.prompt_id, r.model): r.response
                       for r in responses.itertuples(index=False)}
    return prompt_by_id, response_by_key


def build_items(task: str, prompt_by_id: dict, response_by_key: dict) -> list[dict]:
    """Turn one processed table into a list of work items the judges will score."""
    table, lang = TASKS[task]
    df = pd.read_csv(PROCESSED / table)
    items = []
    if task.endswith("pairwise"):
        for r in df.itertuples(index=False):
            items.append({
                "item_id": r.pairing_id,                      # the merge key, also the checkpoint key
                "language": lang,
                "prompt_id": r.prompt_id,
                "prompt": prompt_by_id[(lang, r.prompt_id)],
                "response_a": response_by_key[(lang, r.prompt_id, r.model_a)],
                "response_b": response_by_key[(lang, r.prompt_id, r.model_b)],
            })
    else:  # tamil_direct: one response scored on its own (no A/B, so no order swap)
        for r in df.itertuples(index=False):
            items.append({
                "item_id": f"{r.prompt_id}|{r.model}",
                "language": lang,
                "prompt_id": r.prompt_id,
                "prompt": prompt_by_id[(lang, r.prompt_id)],
                "response": response_by_key[(lang, r.prompt_id, r.model)],
            })
    return items


# ---------------------------------------------------------------------------------------------
# The fixed rubric prompts
# ---------------------------------------------------------------------------------------------
LANG_NAME = {"tamil": "Tamil", "hindi": "Hindi"}


def pairwise_messages(item: dict, answer_a: str, answer_b: str) -> list[dict]:
    """The single fixed pairwise rubric. Identical for every judge and every order; only A/B swap."""
    language = LANG_NAME[item["language"]]
    system = (
        f"You are an impartial expert evaluator of answers written in {language}. "
        "You are shown a question and two candidate answers, A and B. Decide which answer is better "
        "overall, weighing: factual accuracy and correctness; helpfulness and completeness; fluency "
        f"and grammatical correctness in {language}; and cultural appropriateness. "
        "Ignore the order in which the answers are shown, their length, and any formatting. "
        "If the two answers are about equally good or equally bad, choose \"tie\". "
        "Reply with ONLY a JSON object and nothing else, in the form "
        "{\"verdict\": \"A\" | \"B\" | \"tie\", \"reason\": \"<at most 30 words>\"}."
    )
    user = (
        f"Question:\n{item['prompt']}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Which answer is better? Respond with the JSON object only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def direct_messages(item: dict) -> list[dict]:
    """The single fixed direct-assessment rubric, mirroring PARIKSHA's three dimensions (0-2, 0-2, 0/1)."""
    language = LANG_NAME[item["language"]]
    system = (
        f"You are an impartial expert evaluator of a single answer written in {language}. "
        "Score the answer on three dimensions: linguistic_acceptability (0, 1 or 2 for its "
        f"{language} grammar and fluency); task_quality (0, 1 or 2 for how well it answers the "
        "question); hallucination (1 if the answer contains any fabricated or incorrect information, "
        "else 0). Reply with ONLY a JSON object and nothing else, in the form "
        "{\"linguistic_acceptability\": 0, \"task_quality\": 0, \"hallucination\": 0, "
        "\"reason\": \"<at most 30 words>\"}."
    )
    user = f"Question:\n{item['prompt']}\n\nAnswer:\n{item['response']}\n\nScore the answer. Respond with the JSON object only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------------------------
# Parsing the model output
# ---------------------------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of the model's text, tolerating code fences or stray prose."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):  # strip a ```json ... ``` fence if present
        cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)  # fall back to the first {...} block
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def parse_pairwise(text: str) -> str | None:
    """Return "A", "B" or "tie" from the model's JSON, or None if it cannot be parsed."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).strip()
    upper = verdict.upper()
    if upper == "A":
        return "A"
    if upper == "B":
        return "B"
    if verdict.lower() == "tie":
        return "tie"
    return None


def parse_direct(text: str) -> dict | None:
    """Return {LA, TQ, H} ints from the model's JSON, or None if they cannot be parsed / are out of range."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return None
    try:
        la = int(obj["linguistic_acceptability"])
        tq = int(obj["task_quality"])
        h = int(obj["hallucination"])
    except (KeyError, ValueError, TypeError):
        return None
    if la in (0, 1, 2) and tq in (0, 1, 2) and h in (0, 1):
        return {"LA": la, "TQ": tq, "H": h}
    return None


# ---------------------------------------------------------------------------------------------
# Rate limiting and API clients
# ---------------------------------------------------------------------------------------------
class RateLimiter:
    """Enforce a minimum gap between calls for a judge that has a requests-per-minute limit."""

    def __init__(self, rpm: int | None):
        self.min_interval = 60.0 / rpm if rpm else 0.0  # e.g. 10 rpm -> at least 6 s between calls
        self.last = 0.0

    def wait(self) -> None:
        if self.min_interval:
            gap = time.monotonic() - self.last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
        self.last = time.monotonic()


def make_client(judge_cfg: dict, keys: dict) -> OpenAI:
    """Build an OpenAI-compatible client for a judge (both OpenRouter and SEA-LION speak this protocol)."""
    key = keys.get(judge_cfg["api_key_env"])
    if not key:
        sys.exit(f"Missing {judge_cfg['api_key_env']} in .env for judge model {judge_cfg['model']}.")
    return OpenAI(api_key=key, base_url=judge_cfg["base_url"])


# ---------------------------------------------------------------------------------------------
# One API call
# ---------------------------------------------------------------------------------------------
def call_judge(client: OpenAI, judge_cfg: dict, seed: int, messages: list[dict]) -> dict:
    """Make one chat-completion call and return the raw text, usage, cost and the settings actually used.

    Handles the temperature / reasoning-effort rules and asks OpenRouter to report the dollar cost.
    """
    provider = judge_cfg["provider"]
    is_openai = provider == "openrouter" and judge_cfg["model"].startswith("openai/")
    kwargs: dict = {
        "model": judge_cfg["model"],
        "messages": messages,
        "max_tokens": judge_cfg["max_output_tokens"],  # the "short output" cap
    }
    extra: dict = {}
    # Temperature 0 only where the model supports it; otherwise omit and record "unsupported".
    if judge_cfg.get("supports_temperature"):
        kwargs["temperature"] = judge_cfg.get("temperature", 0)
        temperature_logged: object = kwargs["temperature"]
    else:
        temperature_logged = "unsupported"
    # Reasoning models: ask for the lowest effort to keep reasoning tokens (and cost) small.
    effort = judge_cfg.get("reasoning_effort")
    if effort:
        extra["reasoning"] = {"effort": effort}
    # OpenAI models accept a fixed seed and a JSON response mode, which make parsing more reliable.
    if is_openai:
        kwargs["seed"] = seed
        kwargs["response_format"] = {"type": "json_object"}
    # OpenRouter returns the exact dollar cost of the call when usage accounting is switched on.
    if provider == "openrouter":
        extra["usage"] = {"include": True}
    if extra:
        kwargs["extra_body"] = extra

    response = client.chat.completions.create(**kwargs)
    data = response.model_dump()  # a plain dict, so provider-specific fields (cost) are reachable
    text = (data["choices"][0]["message"].get("content") or "").strip()
    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    visible_tokens = max(completion_tokens - reasoning_tokens, 0)  # output the user would actually see
    # Prefer the cost the API reports; otherwise price it from the config table (SEA-LION is free).
    cost = usage.get("cost")
    if cost is None:
        price = judge_cfg["price_per_1m"]
        cost = prompt_tokens / 1e6 * price["input"] + completion_tokens / 1e6 * price["output"]
    return {
        "model_returned": data.get("model", judge_cfg["model"]),
        "raw_output": text,
        "input_tokens": prompt_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": visible_tokens,
        "total_completion_tokens": completion_tokens,
        "cost_usd": float(cost),
        "temperature": temperature_logged,
        "reasoning_effort": effort,
    }


# ---------------------------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------------------------
def log_path(task: str, judge: str) -> Path:
    return LOG_DIR / f"{task}__{judge}.jsonl"


def done_keys(task: str, judge: str) -> set[tuple[str, str]]:
    """Read a judge's log and return the (item_id, order) pairs already finished without error."""
    path = log_path(task, judge)
    done: set[tuple[str, str]] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") is None:  # errored calls are left out so they are retried
                done.add((rec["item_id"], rec["order"]))
    return done


def prior_cost_and_count(task: str, judge: str) -> tuple[float, int]:
    """Sum the cost and count the successful calls already logged, so budgeting survives restarts."""
    path = log_path(task, judge)
    total, n = 0.0, 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error") is None:
                total += float(rec.get("cost_usd") or 0.0)
                n += 1
    return total, n


def append_log(task: str, judge: str, record: dict) -> None:
    """Append one JSON line. Append-only writes are what make the run safely resumable."""
    with log_path(task, judge).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------
# Running one task
# ---------------------------------------------------------------------------------------------
def run(task: str, pilot: int | None, consistency: int | None = None) -> None:
    config = load_config()
    keys = load_keys()
    seed = int(config.get("seed", 42))
    max_budget = float(config.get("max_budget_usd", 100))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Self-consistency re-runs a subset of Tamil pairs in the A/B order only, for the two OpenAI
    # judges, and writes a SEPARATE log so the original pilot records are not overwritten.
    log_task = "consistency" if consistency else task
    prompt_by_id, response_by_key = load_prompts_and_responses()
    items = build_items(task, prompt_by_id, response_by_key)
    n_keep = consistency if consistency is not None else pilot
    if n_keep is not None:
        items = items[:n_keep]  # the first N items, deterministic because the tables are sorted
    if consistency is not None:
        orders = ("AB",)  # original presentation order only
        judges = {k: v for k, v in config["judges"].items() if k in ("frontier", "low_cost")}
    else:
        orders = ("AB", "BA") if task.endswith("pairwise") else ("NA",)
        judges = config["judges"]

    clients = {name: make_client(cfg, keys) for name, cfg in judges.items()}
    limiters = {name: RateLimiter(cfg.get("rpm_limit")) for name, cfg in judges.items()}
    # Seed the running cost/count from any previous run of this log stem, so the budget guard is accurate.
    cost_sum = {name: prior_cost_and_count(log_task, name)[0] for name in judges}
    call_count = {name: prior_cost_and_count(log_task, name)[1] for name in judges}

    print(f"Task {task} (log={log_task}): {len(items)} items x {len(orders)} order(s) x {len(judges)} judges"
          f"{' (PILOT)' if pilot is not None else ''}{' (CONSISTENCY)' if consistency is not None else ''}")

    for name, cfg in judges.items():
        already = done_keys(log_task, name)
        todo = [(it, o) for it in items for o in orders if (it["item_id"], o) not in already]
        print(f"  judge {name} ({cfg['model']}): {len(already)} done, {len(todo)} to do")
        for item, order in todo:
            # Budget guard: outside a pilot/consistency check, stop before a call if the projected
            # full cost is too high.
            if pilot is None and consistency is None and call_count[name] > 0:
                projected = project_full_cost(cost_sum, call_count, judges)
                if projected > max_budget:
                    sys.exit(f"BUDGET STOP: projected full-run cost ${projected:,.2f} exceeds "
                             f"MAX_BUDGET_USD ${max_budget:,.2f}. Nothing further was run.")
            # Build the messages for this task and order.
            if task.endswith("pairwise"):
                a, b = (item["response_a"], item["response_b"]) if order == "AB" \
                    else (item["response_b"], item["response_a"])
                messages = pairwise_messages(item, a, b)
            else:
                messages = direct_messages(item)
            limiters[name].wait()  # SEA-LION 10 rpm; LOW_COST 20 rpm (OpenRouter new-account cap)
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            record = {
                "schema": 1, "task": log_task, "judge": name, "provider": cfg["provider"],
                "model": cfg["model"], "snapshot": cfg["snapshot"],
                "language": item["language"], "item_id": item["item_id"],
                "prompt_id": item["prompt_id"], "order": order, "timestamp": timestamp,
                "request_messages": messages,
            }
            try:
                result = call_judge(clients[name], cfg, seed, messages)
            except Exception as exc:  # network/API error: log it and move on; it will be retried next run
                record.update({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                append_log(log_task, name, record)
                print(f"    ! {name} {item['item_id']} {order}: {record['error']}")
                continue
            record.update(result)
            record["error"] = None
            # Parse the verdict / scores and, for the swapped order, map the verdict back to A/B.
            if task.endswith("pairwise"):
                presented = parse_pairwise(result["raw_output"])
                record["verdict_presented"] = presented
                record["verdict_canonical"] = (
                    None if presented is None else (presented if order == "AB" else SWAP[presented]))
                record["parse_ok"] = presented is not None
            else:
                scores = parse_direct(result["raw_output"])
                record["direct_scores"] = scores
                record["parse_ok"] = scores is not None
            append_log(log_task, name, record)
            cost_sum[name] += result["cost_usd"]
            call_count[name] += 1

    if pilot is not None and consistency is None:
        report_pilot(task, judges)


def project_full_cost(cost_sum: dict, call_count: dict, judges: dict) -> float:
    """Project the full-run cost: each judge's mean cost per call so far times its 6,284 full calls."""
    total = 0.0
    for name in judges:
        if call_count[name]:
            total += (cost_sum[name] / call_count[name]) * FULL_CALLS_PER_JUDGE
    return total


# ---------------------------------------------------------------------------------------------
# Pilot report
# ---------------------------------------------------------------------------------------------
def report_pilot(task: str, judges: dict) -> None:
    """Print the pilot cost/quality table and the projected full-run cost per judge."""
    human = {}
    if task.endswith("pairwise"):
        df = pd.read_csv(PROCESSED / TASKS[task][0])
        human = {r.pairing_id: r.human_majority for r in df.itertuples(index=False)}

    print("\n=== PILOT REPORT ===")
    header = ("judge", "calls", "in/call", "reason/call", "out/call",
              "parse_fail", "$/1k", "proj_$ (6,284)")
    print("{:<10} {:>6} {:>8} {:>11} {:>8} {:>10} {:>9} {:>14}".format(*header))

    projected_total = 0.0
    frontier_projection = 0.0
    for name in judges:
        recs = []
        path = log_path(task, name)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    recs.append(json.loads(line))
        ok = [r for r in recs if r.get("error") is None]
        n = len(ok)
        if n == 0:
            print(f"{name:<10} {0:>6}  (no successful calls)")
            continue
        in_tok = sum(r.get("input_tokens", 0) for r in ok) / n
        re_tok = sum(r.get("reasoning_tokens", 0) for r in ok) / n
        out_tok = sum(r.get("output_tokens", 0) for r in ok) / n
        fails = sum(0 if r.get("parse_ok") else 1 for r in ok)
        fail_rate = fails / n
        cost = sum(r.get("cost_usd", 0.0) for r in ok)
        cost_per_1k = cost / n * 1000
        proj = cost / n * FULL_CALLS_PER_JUDGE
        projected_total += proj
        if name == "frontier":
            frontier_projection = proj
        print("{:<10} {:>6} {:>8.0f} {:>11.0f} {:>8.0f} {:>9.1%} {:>9.3f} {:>14.2f}".format(
            name, n, in_tok, re_tok, out_tok, fail_rate, cost_per_1k, proj))

        # Agreement with the human majority (diagnostic only; the pilot is far too small for conclusions).
        if task.endswith("pairwise"):
            judged = [(r["verdict_canonical"], human.get(r["item_id"]))
                      for r in ok if r.get("parse_ok") and r["item_id"] in human]
            if judged:
                agree = sum(1 for v, h in judged if v == h) / len(judged)
                print(f"           agreement with human_majority: {agree:.1%} over {len(judged)} verdicts")

    print(f"\nProjected full-run total (~{FULL_CALLS_PER_JUDGE * len(judges):,} calls): "
          f"${projected_total:,.2f}  (budget ${load_config().get('max_budget_usd')})")
    # Amendment 6: if the frontier projection is high, surface the cheaper Batch option (do not switch).
    if frontier_projection > 60:
        print(f"NOTE: frontier projection ${frontier_projection:,.2f} > $60. OpenAI's Batch API is about "
              "50% cheaper for the same model; it could roughly halve the frontier cost. Not switched "
              "without your go-ahead.")
    print("Diagnostic only. The full run has NOT been started.")


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four LLM judges over the PARIKSHA data.")
    parser.add_argument("--task", choices=list(TASKS),
                        help="which table to judge (not needed with --consistency)")
    parser.add_argument("--pilot", type=int, default=None, metavar="N",
                        help="judge only the first N items (for a cost/quality pilot)")
    parser.add_argument("--consistency", type=int, default=None, metavar="N",
                        help="re-run the first N Tamil pairs in A/B order only, for FRONTIER and "
                             "LOW_COST, logging to results/judgements/consistency__*.jsonl")
    args = parser.parse_args()
    if args.consistency is not None:
        run("tamil_pairwise", pilot=None, consistency=args.consistency)
    elif not args.task:
        parser.error("--task is required unless --consistency is set")
    else:
        run(args.task, args.pilot)


if __name__ == "__main__":
    main()
