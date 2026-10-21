"""Typed contracts for every agent input and output (structured outputs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

NetworkType = Literal["5G Private", "5G Macro", "5G Small Cell", "4G LTE Private", "4G LTE Macro"]


class LinkBudgetParams(BaseModel):
    """Link budget assumptions. Defaults describe a typical FR1 macro site."""

    scenario: Literal["FSPL", "UMa_LOS", "UMa_NLOS", "UMi_LOS", "UMi_NLOS"] = "UMa_NLOS"
    gnb_tx_power_dbm: float = Field(46.0, description="Total gNB transmit power")
    gnb_antenna_gain_dbi: float = 17.0
    gnb_cable_loss_db: float = 2.0
    gnb_noise_figure_db: float = 5.0
    gnb_height_m: float = Field(25.0, ge=10, le=150)
    ue_tx_power_dbm: float = 23.0
    ue_antenna_gain_dbi: float = 0.0
    ue_noise_figure_db: float = 7.0
    ue_height_m: float = Field(1.5, ge=1, le=22.5)
    dl_bandwidth_mhz: float = 100.0
    ul_bandwidth_mhz: float = Field(5.0, description="UL bandwidth allocated to a cell edge UE")
    dl_required_sinr_db: float = -5.0
    ul_required_sinr_db: float = -5.0
    shadow_fading_margin_db: float = 8.0
    penetration_loss_db: float = 0.0
    interference_margin_db: float = 3.0


class NetworkRequest(BaseModel):
    region: str = Field(..., min_length=2, max_length=200)
    network_type: NetworkType
    frequency_mhz: float = Field(..., gt=400, lt=52_600, description="Carrier frequency")
    coverage_radius_km: float = Field(..., gt=0, le=20)
    use_case: str | None = Field(None, max_length=500, description="e.g. AGVs, video, sensors")
    link_budget: LinkBudgetParams = Field(default_factory=LinkBudgetParams)


class ParsedRequest(BaseModel):
    """What the LLM may extract from free text. Link budget is deliberately absent:
    a small model filled those optional numbers with zeros, so engineering defaults
    are applied in code instead."""

    region: str = Field(..., min_length=2, max_length=200)
    network_type: NetworkType
    frequency_mhz: float = Field(..., gt=400, lt=52_600)
    coverage_radius_km: float = Field(..., gt=0, le=20)
    use_case: str | None = Field(None, max_length=500)

    def to_request(self) -> NetworkRequest:
        return NetworkRequest(**self.model_dump())


class LinkDirectionResult(BaseModel):
    direction: Literal["DL", "UL"]
    eirp_dbm: float
    rx_sensitivity_dbm: float
    max_allowed_path_loss_db: float
    received_power_dbm: float
    margin_db: float


class RFAnalysis(BaseModel):
    frequency_mhz: float
    radius_km: float
    scenario: str
    fspl_db: float
    path_loss_db: float
    downlink: LinkDirectionResult
    uplink: LinkDirectionResult
    limiting_link: Literal["DL", "UL"]
    feasible: bool
    max_cell_radius_km: float
    estimated_sites: int | None
    assumptions: LinkBudgetParams


class Citation(BaseModel):
    source: str = Field(..., description="File name of the retrieved document")
    page: int | None = None
    quote: str = Field(..., max_length=400, description="Short excerpt that supports the claim")


class HardwareOption(BaseModel):
    vendor: str
    product: str
    category: Literal["Radio Unit", "Baseband", "Antenna", "Core", "End to end solution"]
    supports_band: str | None = None
    rationale: str
    source_url: str | None = None


class HardwareShortlist(BaseModel):
    options: list[HardwareOption] = Field(..., min_length=1, max_length=4)


class ProtocolParameter(BaseModel):
    name: str = Field(..., description="e.g. subcarrier spacing, TDD pattern, RRC T310")
    value: str
    justification: str
    citations: list[Citation] = Field(default_factory=list)


class ProtocolRecommendation(BaseModel):
    nr_band: str = Field(..., description="3GPP band, e.g. n78")
    duplex: Literal["TDD", "FDD", "SDL"]
    subcarrier_spacing_khz: int
    parameters: list[ProtocolParameter]

    @field_validator("nr_band")
    @classmethod
    def _norm_band(cls, v: str) -> str:
        return v.strip().lower()


class NetPlanReport(BaseModel):
    executive_summary: str
    request: NetworkRequest
    rf: RFAnalysis
    protocol: ProtocolRecommendation
    hardware: HardwareShortlist
    risks: list[str]
    next_steps: list[str]


class GuardrailFinding(BaseModel):
    level: Literal["info", "warning", "error"]
    check: str
    message: str


class PlanResponse(BaseModel):
    report: NetPlanReport
    findings: list[GuardrailFinding]
    latency_s: float
    usage: dict[str, dict[str, int]] = Field(default_factory=dict, description="Tokens per model")
