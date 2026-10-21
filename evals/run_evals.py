"""Offline evaluation with regression gate.

    python -m evals.run_evals                 # agent + retrieval evals
    python -m evals.run_evals --judge         # adds LLM-as-a-judge groundedness
    python -m evals.run_evals --save-baseline # store current scores as the baseline

Exits with code 1 when a metric drops more than --tolerance below the baseline,
so it can block a prompt or model change in CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from pydantic import BaseModel, Field

from netplan.agents import build_planner
from netplan.config import get_settings
from netplan.schemas import NetworkRequest, PlanResponse

HERE = Path(__file__).parent
BASELINE = HERE / "baseline.json"


class JudgeVerdict(BaseModel):
    groundedness: int = Field(..., ge=1, le=5, description="5 = every claim is supported by tool data")
    hallucinated_claims: list[str]


JUDGE_PROMPT = """You audit a telecom planning report for hallucinations.
Tool data (ground truth) and the report are given as JSON. Score groundedness 1 to 5 and list
any claim in the summary, risks or next steps that contradicts or is not supported by the
tool data (numbers, bands, vendors). Be strict."""


def score_case(case: dict, res: PlanResponse) -> dict:
    p = res.report.protocol
    params = p.parameters or []
    hw = res.report.hardware.options
    errors = [f for f in res.findings if f.level == "error"]
    return {
        "band_correct": float(p.nr_band in case["expected_bands"]),
        "duplex_correct": float(p.duplex == case["expected_duplex"]),
        "scs_valid": float(not any(f.check == "scs" and f.level == "error" for f in res.findings)),
        "citation_coverage": sum(bool(x.citations) for x in params) / len(params) if params else 0.0,
        "hardware_sourced": sum(bool(h.source_url) for h in hw) / len(hw) if hw else 0.0,
        "guardrail_errors": len(errors),
        "latency_s": res.latency_s,
        "total_tokens": sum(u.get("total_tokens", 0) for u in res.usage.values()),
    }


async def judge(planner, res: PlanResponse) -> JudgeVerdict:
    model = planner.coord_model.with_structured_output(JudgeVerdict)
    payload = {
        "tool_data": {
            "rf": res.report.rf.model_dump(exclude={"assumptions"}),
            "protocol": res.report.protocol.model_dump(),
            "hardware": res.report.hardware.model_dump(),
        },
        "report": {
            "summary": res.report.executive_summary,
            "risks": res.report.risks,
            "next_steps": res.report.next_steps,
        },
    }
    return await model.ainvoke(f"{JUDGE_PROMPT}\n\n{json.dumps(payload, ensure_ascii=False)}")


def retrieval_eval(planner, k: int) -> dict:
    cases = json.loads((HERE / "retrieval_set.json").read_text())
    hits = 0
    for c in cases:
        docs = planner.retriever.search(c["query"], k=k)
        text = " ".join(d.page_content.lower() for d in docs)
        hits += any(kw.lower() in text for kw in c["expected_keywords"])
    return {f"retrieval_hit@{k}": hits / len(cases)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--tolerance", type=float, default=0.1)
    args = ap.parse_args()

    settings = get_settings()
    planner = build_planner(settings)
    cases = json.loads((HERE / "dataset.json").read_text())[: args.limit]

    rows = []
    for case in cases:
        t = time.perf_counter()
        try:
            res = await planner.plan(NetworkRequest(**case["request"]))
            row = {"id": case["id"], "ok": 1.0, **score_case(case, res)}
            if args.judge:
                v = await judge(planner, res)
                row["groundedness"] = v.groundedness / 5
                row["hallucinated_claims"] = v.hallucinated_claims
        except Exception as e:  # a crash is a failed case, not a failed eval run
            row = {"id": case["id"], "ok": 0.0, "error": repr(e), "latency_s": time.perf_counter() - t}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    metrics = ["ok", "band_correct", "duplex_correct", "scs_valid", "citation_coverage", "hardware_sourced"]
    if args.judge:
        metrics.append("groundedness")
    summary = {m: round(statistics.mean(r.get(m, 0.0) for r in rows), 3) for m in metrics}
    lat = sorted(r["latency_s"] for r in rows)
    summary["latency_p50_s"] = round(statistics.median(lat), 2)
    summary["latency_max_s"] = round(lat[-1], 2)
    summary["avg_tokens"] = round(statistics.mean(r.get("total_tokens", 0) for r in rows))
    summary.update(retrieval_eval(planner, settings.retrieval_k))

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    result = {"summary": summary, "cases": rows}
    (out_dir / f"{stamp}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nSUMMARY", json.dumps(summary, indent=2))

    if args.save_baseline:
        BASELINE.write_text(json.dumps(summary, indent=2))
        print(f"Baseline saved to {BASELINE}")
        return 0

    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
        regressions = [
            f"{m}: {base[m]} -> {summary[m]}"
            for m in summary
            if m in base and not m.startswith(("latency", "avg_tokens")) and summary[m] < base[m] - args.tolerance
        ]
        if regressions:
            print("REGRESSION\n" + "\n".join(regressions))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
