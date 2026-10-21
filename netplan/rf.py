"""Deterministic RF engineering: path loss models and a downlink/uplink link budget.

The LLM never computes RF numbers. Agents call these functions as tools, and the
output guardrail recomputes them, so the report cannot carry hallucinated dB values.

Models follow 3GPP TR 38.901 (Table 7.4.1-1), simplified:
LOS/NLOS chosen explicitly, no LOS probability, no shadow fading draw.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

from netplan.schemas import LinkBudgetParams, LinkDirectionResult, RFAnalysis

C = 299_792_458.0  # speed of light, m/s

Scenario = Literal["FSPL", "UMa_LOS", "UMa_NLOS", "UMi_LOS", "UMi_NLOS"]


def fspl_db(freq_mhz: float, dist_km: float) -> float:
    """Free space path loss. FSPL = 32.44 + 20 log10(f_MHz) + 20 log10(d_km)."""
    if freq_mhz <= 0 or dist_km <= 0:
        raise ValueError("frequency and distance must be positive")
    return 32.44 + 20 * math.log10(freq_mhz) + 20 * math.log10(dist_km)


def _d3d(d2d_m: float, h_bs: float, h_ut: float) -> float:
    return math.sqrt(d2d_m**2 + (h_bs - h_ut) ** 2)


def _breakpoint_m(h_bs: float, h_ut: float, fc_ghz: float) -> float:
    # d'_BP = 4 h'_BS h'_UT fc / c, with effective environment height h_E = 1 m
    return 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc_ghz * 1e9 / C


def uma_los_db(d2d_m: float, fc_ghz: float, h_bs: float = 25.0, h_ut: float = 1.5) -> float:
    d3d = _d3d(d2d_m, h_bs, h_ut)
    d_bp = _breakpoint_m(h_bs, h_ut, fc_ghz)
    if d2d_m <= d_bp:
        return 28.0 + 22 * math.log10(d3d) + 20 * math.log10(fc_ghz)
    return 28.0 + 40 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 9 * math.log10(d_bp**2 + (h_bs - h_ut) ** 2)


def uma_nlos_db(d2d_m: float, fc_ghz: float, h_bs: float = 25.0, h_ut: float = 1.5) -> float:
    d3d = _d3d(d2d_m, h_bs, h_ut)
    pl_nlos = 13.54 + 39.08 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return max(uma_los_db(d2d_m, fc_ghz, h_bs, h_ut), pl_nlos)


def umi_los_db(d2d_m: float, fc_ghz: float, h_bs: float = 10.0, h_ut: float = 1.5) -> float:
    d3d = _d3d(d2d_m, h_bs, h_ut)
    d_bp = _breakpoint_m(h_bs, h_ut, fc_ghz)
    if d2d_m <= d_bp:
        return 32.4 + 21 * math.log10(d3d) + 20 * math.log10(fc_ghz)
    return 32.4 + 40 * math.log10(d3d) + 20 * math.log10(fc_ghz) - 9.5 * math.log10(d_bp**2 + (h_bs - h_ut) ** 2)


def umi_nlos_db(d2d_m: float, fc_ghz: float, h_bs: float = 10.0, h_ut: float = 1.5) -> float:
    d3d = _d3d(d2d_m, h_bs, h_ut)
    pl_nlos = 35.3 * math.log10(d3d) + 22.4 + 21.3 * math.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return max(umi_los_db(d2d_m, fc_ghz, h_bs, h_ut), pl_nlos)


def path_loss_db(scenario: Scenario, freq_mhz: float, dist_km: float, h_bs: float, h_ut: float) -> float:
    if scenario == "FSPL":
        return fspl_db(freq_mhz, dist_km)
    fc_ghz, d2d_m = freq_mhz / 1000.0, dist_km * 1000.0
    fn: dict[str, Callable[..., float]] = {
        "UMa_LOS": uma_los_db,
        "UMa_NLOS": uma_nlos_db,
        "UMi_LOS": umi_los_db,
        "UMi_NLOS": umi_nlos_db,
    }
    return fn[scenario](max(d2d_m, 10.0), fc_ghz, h_bs, h_ut)


def noise_floor_dbm(bandwidth_mhz: float, noise_figure_db: float) -> float:
    """Thermal noise kTB at 290 K plus receiver noise figure."""
    return -174.0 + 10 * math.log10(bandwidth_mhz * 1e6) + noise_figure_db


def _direction(
    name: Literal["DL", "UL"],
    eirp_dbm: float,
    rx_gain_dbi: float,
    rx_nf_db: float,
    bw_mhz: float,
    sinr_db: float,
    p: LinkBudgetParams,
    pl_db: float,
) -> LinkDirectionResult:
    sensitivity = noise_floor_dbm(bw_mhz, rx_nf_db) + sinr_db
    margins = p.shadow_fading_margin_db + p.penetration_loss_db + p.interference_margin_db
    mapl = eirp_dbm + rx_gain_dbi - sensitivity - margins
    return LinkDirectionResult(
        direction=name,
        eirp_dbm=round(eirp_dbm, 2),
        rx_sensitivity_dbm=round(sensitivity, 2),
        max_allowed_path_loss_db=round(mapl, 2),
        received_power_dbm=round(eirp_dbm + rx_gain_dbi - pl_db - margins, 2),
        margin_db=round(mapl - pl_db, 2),
    )


def _max_radius_km(mapl_db: float, p: LinkBudgetParams, freq_mhz: float) -> float:
    """Largest distance where path loss <= MAPL, found by bisection (10 m to 20 km)."""
    lo, hi = 0.01, 20.0

    def pl(d: float) -> float:
        return path_loss_db(p.scenario, freq_mhz, d, p.gnb_height_m, p.ue_height_m)

    if pl(lo) > mapl_db:
        return 0.0
    if pl(hi) <= mapl_db:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pl(mid) <= mapl_db else (lo, mid)
    return lo


def analyze_link(freq_mhz: float, radius_km: float, p: LinkBudgetParams | None = None) -> RFAnalysis:
    """Full link budget at the cell edge, DL and UL, with max cell radius and site estimate."""
    p = p or LinkBudgetParams()
    pl = path_loss_db(p.scenario, freq_mhz, radius_km, p.gnb_height_m, p.ue_height_m)

    dl = _direction(
        "DL",
        p.gnb_tx_power_dbm + p.gnb_antenna_gain_dbi - p.gnb_cable_loss_db,
        p.ue_antenna_gain_dbi,
        p.ue_noise_figure_db,
        p.dl_bandwidth_mhz,
        p.dl_required_sinr_db,
        p,
        pl,
    )
    ul = _direction(
        "UL",
        p.ue_tx_power_dbm + p.ue_antenna_gain_dbi,
        p.gnb_antenna_gain_dbi - p.gnb_cable_loss_db,
        p.gnb_noise_figure_db,
        p.ul_bandwidth_mhz,
        p.ul_required_sinr_db,
        p,
        pl,
    )
    limiting = min(dl, ul, key=lambda r: r.margin_db)
    max_r = _max_radius_km(limiting.max_allowed_path_loss_db, p, freq_mhz)

    # Hexagonal cell area = 3*sqrt(3)/2 * r^2 versus the circular target area
    if max_r <= 0:
        sites = None
    else:
        sites = max(1, math.ceil((math.pi * radius_km**2) / (2.598 * max_r**2)))

    return RFAnalysis(
        frequency_mhz=freq_mhz,
        radius_km=radius_km,
        scenario=p.scenario,
        fspl_db=round(fspl_db(freq_mhz, radius_km), 2),
        path_loss_db=round(pl, 2),
        downlink=dl,
        uplink=ul,
        limiting_link=limiting.direction,
        feasible=limiting.margin_db >= 0,
        max_cell_radius_km=round(max_r, 3),
        estimated_sites=sites,
        assumptions=p,
    )
