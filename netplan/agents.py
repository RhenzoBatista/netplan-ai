"""Multi-agent planner (supervisor pattern).

Coordinator (LLM)
  ├── analyze_rf_link           deterministic link budget, no LLM
  ├── consult_hardware_architect  sub-agent with web search (call limited)
  └── consult_protocol_specialist sub-agent with hybrid RAG over 3GPP + read-only SQL

Every sub-agent returns a Pydantic object (structured output). The coordinator only
writes what needs judgment (summary, risks, next steps). The final report is assembled
in code and then passes the output guardrails.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from netplan.config import Settings
from netplan.guardrails import check_request, check_text_input, validate_report
from netplan.rag import HybridRetriever, format_hits
from netplan.rf import analyze_link
from netplan.schemas import (
    HardwareShortlist,
    NetPlanReport,
    NetworkRequest,
    ParsedRequest,
    PlanResponse,
    ProtocolRecommendation,
    RFAnalysis,
)
from netplan.telecom_db import ReadOnlyTelecomDB

HARDWARE_PROMPT = """You are a cellular RAN hardware architect.
Find at most 3 real, currently sold products (radio units, basebands or end to end private
5G solutions) that support the requested band and network type. Prefer vendors with public
datasheets. Fill source_url with the page you used. Never invent a product or a URL. If the
search does not confirm band support, leave supports_band empty."""

PROTOCOL_PROMPT = """You are a 3GPP RAN protocol specialist.
1. Call lookup_band to get the NR band, duplex mode and allowed subcarrier spacings.
2. Use search_3gpp_specs to ground each parameter in the specifications.
3. Recommend subcarrier spacing, TDD pattern (if TDD), SSB periodicity and key RRC timers
   for the use case. Each parameter must carry citations copied from the retrieved
   [source=... page=...] headers. If the specs do not cover a parameter, say so in the
   justification and leave citations empty. Do not guess values that are not supported."""

COORDINATOR_PROMPT = """You are the chief network deployment engineer.
Call analyze_rf_link, consult_hardware_architect and consult_protocol_specialist
(call them all, in any order). Then write an executive summary, the main risks and the
next steps. Use only numbers returned by the tools. Write in the language of the region
field if it is clearly Portuguese, otherwise English."""


class CoordinatorOutput(BaseModel):
    executive_summary: str = Field(..., max_length=1500)
    risks: list[str] = Field(..., max_length=8)
    next_steps: list[str] = Field(..., max_length=8)


SearchFn = Callable[[str], dict[str, Any]]


def tavily_search() -> SearchFn:
    from tavily import TavilyClient

    client = TavilyClient()
    return lambda q: client.search(q, max_results=5)


class NetPlanner:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        db: ReadOnlyTelecomDB,
        search_fn: SearchFn | None = None,
        coordinator_model: Any | None = None,
        specialist_model: Any | None = None,
    ):
        self.s = settings
        self.retriever = retriever
        self.db = db
        self.search_fn = search_fn or tavily_search()
        self.coord_model = coordinator_model or init_chat_model(settings.coordinator_model)
        self.spec_model = specialist_model or init_chat_model(settings.specialist_model)

    # ---------- shared middleware ----------
    def _middleware(self, extra: list | None = None) -> list:
        mw = [
            ModelRetryMiddleware(max_retries=2),
            ModelCallLimitMiddleware(run_limit=self.s.max_model_calls, exit_behavior="end"),
        ]
        if self.s.fallback_model:
            mw.append(ModelFallbackMiddleware(self.s.fallback_model))
        return mw + (extra or [])

    # ---------- free text to structured request ----------
    async def parse_request(self, text: str) -> NetworkRequest:
        check_text_input(text)
        parser = self.spec_model.with_structured_output(ParsedRequest)
        parsed: ParsedRequest = await parser.ainvoke(f"Extract the network deployment request.\n\nRequest:\n{text}")
        return parsed.to_request()

    # ---------- sub-agents ----------
    def _hardware_agent(self):
        search = self.search_fn

        @tool
        def web_search(query: str) -> str:
            """Search the web for RAN hardware datasheets and product pages."""
            try:
                res = search(query)
                hits = [
                    {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:500]}
                    for r in res.get("results", [])
                ]
                return json.dumps(hits, ensure_ascii=False)
            except Exception as e:  # tool errors go back to the model, not up the stack
                return f"search error: {e}"

        return create_agent(
            self.spec_model,
            tools=[web_search],
            system_prompt=HARDWARE_PROMPT,
            response_format=ToolStrategy(HardwareShortlist),
            middleware=self._middleware(
                [ToolCallLimitMiddleware(tool_name="web_search", run_limit=self.s.max_web_searches)]
            ),
            name="hardware_architect",
        )

    def _protocol_agent(self):
        retriever, db, k = self.retriever, self.db, self.s.retrieval_k

        @tool
        def search_3gpp_specs(query: str) -> str:
            """Hybrid search (semantic + keyword) over the indexed 3GPP specifications."""
            return format_hits(retriever.search(query, k=k))

        @tool
        def lookup_band(frequency_mhz: float) -> str:
            """Return NR bands, duplex and allowed subcarrier spacings for a frequency."""
            return json.dumps(db.band_for_frequency(frequency_mhz))

        @tool
        def query_telecom_db(sql: str) -> str:
            """Read-only SELECT over tables nr_bands, numerology and brazil_spectrum."""
            try:
                return json.dumps(db.query(sql))
            except Exception as e:
                return f"query rejected: {e}. Schema:\n{db.schema()}"

        return create_agent(
            self.spec_model,
            tools=[search_3gpp_specs, lookup_band, query_telecom_db],
            system_prompt=PROTOCOL_PROMPT,
            response_format=ToolStrategy(ProtocolRecommendation),
            middleware=self._middleware(),
            name="protocol_specialist",
        )

    # ---------- orchestration ----------
    async def plan(self, req: NetworkRequest) -> PlanResponse:
        t0 = time.perf_counter()
        findings = check_request(req, self.db)
        results: dict[str, Any] = {}
        hw_agent, proto_agent = self._hardware_agent(), self._protocol_agent()
        brief = req.model_dump_json(exclude={"link_budget"})

        async def run_hardware() -> HardwareShortlist:
            out = await hw_agent.ainvoke({"messages": [HumanMessage(f"Shortlist hardware for: {brief}")]})
            results["hardware"] = out["structured_response"]
            return results["hardware"]

        async def run_protocol() -> ProtocolRecommendation:
            out = await proto_agent.ainvoke({"messages": [HumanMessage(f"Recommend protocol config for: {brief}")]})
            results["protocol"] = out["structured_response"]
            return results["protocol"]

        @tool
        def analyze_rf_link() -> str:
            """Deterministic DL/UL link budget, cell edge margin, max radius and site count."""
            results["rf"] = analyze_link(req.frequency_mhz, req.coverage_radius_km, req.link_budget)
            return results["rf"].model_dump_json(exclude={"assumptions"})

        @tool
        async def consult_hardware_architect() -> str:
            """Ask the hardware sub-agent for a sourced equipment shortlist."""
            return (await run_hardware()).model_dump_json()

        @tool
        async def consult_protocol_specialist() -> str:
            """Ask the 3GPP sub-agent for cited protocol parameters."""
            return (await run_protocol()).model_dump_json()

        coordinator = create_agent(
            self.coord_model,
            tools=[analyze_rf_link, consult_hardware_architect, consult_protocol_specialist],
            system_prompt=COORDINATOR_PROMPT,
            response_format=ToolStrategy(CoordinatorOutput),
            middleware=self._middleware(),
            name="coordinator",
        )

        with get_usage_metadata_callback() as usage_cb:
            out = await coordinator.ainvoke({"messages": [HumanMessage(f"Plan this deployment: {brief}")]})
            # If the coordinator skipped a specialist, complete the plan deterministically.
            pending = []
            if "hardware" not in results:
                pending.append(run_hardware())
            if "protocol" not in results:
                pending.append(run_protocol())
            if pending:
                await asyncio.gather(*pending)
            usage = {
                m: {k: int(v.get(k, 0)) for k in ("input_tokens", "output_tokens", "total_tokens")}
                for m, v in usage_cb.usage_metadata.items()
            }

        rf: RFAnalysis = results.get("rf") or analyze_link(req.frequency_mhz, req.coverage_radius_km, req.link_budget)
        summary: CoordinatorOutput = out["structured_response"]
        report = NetPlanReport(
            executive_summary=summary.executive_summary,
            request=req,
            rf=rf,
            protocol=results["protocol"],
            hardware=results["hardware"],
            risks=summary.risks,
            next_steps=summary.next_steps,
        )
        report, out_findings = validate_report(report, req, self.db)
        return PlanResponse(
            report=report,
            findings=findings + out_findings,
            latency_s=round(time.perf_counter() - t0, 2),
            usage=usage,
        )


def build_planner(settings: Settings) -> NetPlanner:
    from netplan.config import build_embeddings

    retriever = HybridRetriever.from_folder(settings.docs_dir, build_embeddings(settings), settings.index_dir)
    return NetPlanner(settings, retriever, ReadOnlyTelecomDB(settings.db_path))
