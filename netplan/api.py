"""REST API. Run with: uvicorn netplan.api:app --reload"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from netplan.agents import NetPlanner, build_planner
from netplan.config import get_settings
from netplan.guardrails import GuardrailError
from netplan.rf import analyze_link
from netplan.schemas import LinkBudgetParams, NetworkRequest, PlanResponse, RFAnalysis

log = logging.getLogger("netplan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.planner = None  # built lazily so /health and /rf work without API keys
    yield


app = FastAPI(title="NetPlan AI", version="2.0.0", lifespan=lifespan)


def get_planner(request: Request) -> NetPlanner:
    if request.app.state.planner is None:
        request.app.state.planner = build_planner(get_settings())
    return request.app.state.planner


class TextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=2000)


class RFRequest(BaseModel):
    frequency_mhz: float = Field(..., gt=400, lt=52_600)
    radius_km: float = Field(..., gt=0, le=20)
    link_budget: LinkBudgetParams = Field(default_factory=LinkBudgetParams)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/rf", response_model=RFAnalysis)
def rf(body: RFRequest) -> RFAnalysis:
    """Deterministic link budget only. No LLM call, no cost."""
    return analyze_link(body.frequency_mhz, body.radius_km, body.link_budget)


@app.post("/plan", response_model=PlanResponse)
async def plan(body: NetworkRequest, planner: NetPlanner = Depends(get_planner)) -> PlanResponse:
    try:
        res = await planner.plan(body)
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    log.info("plan ok latency=%ss usage=%s", res.latency_s, res.usage)
    return res


@app.post("/plan/text", response_model=PlanResponse)
async def plan_text(body: TextRequest, planner: NetPlanner = Depends(get_planner)) -> PlanResponse:
    try:
        req = await planner.parse_request(body.text)
        return await planner.plan(req)
    except GuardrailError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
