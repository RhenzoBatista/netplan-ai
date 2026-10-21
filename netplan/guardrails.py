"""Input and output guardrails.

Input: reject prompt injection attempts and requests outside any known NR band.
Output: recompute RF numbers deterministically, check band, duplex and subcarrier
spacing against the reference DB, and flag uncited protocol claims. The LLM report
is corrected where a ground truth exists and annotated where it does not.
"""

from __future__ import annotations

import re

from netplan.rf import analyze_link
from netplan.schemas import GuardrailFinding, NetPlanReport, NetworkRequest
from netplan.telecom_db import ReadOnlyTelecomDB

_INJECTION = re.compile(
    r"(ignore (all|any|the|previous|prior) (instructions|rules)|system prompt|"
    r"you are now|disregard .{0,30}instructions|developer mode|reveal .{0,20}prompt)",
    re.IGNORECASE,
)


class GuardrailError(ValueError):
    pass


def check_text_input(text: str, max_len: int = 2000) -> None:
    if len(text) > max_len:
        raise GuardrailError(f"Input too long ({len(text)} > {max_len} chars)")
    if _INJECTION.search(text):
        raise GuardrailError("Input rejected by prompt injection filter")


def check_request(req: NetworkRequest, db: ReadOnlyTelecomDB) -> list[GuardrailFinding]:
    for field in (req.region, req.use_case or ""):
        check_text_input(field)
    bands = db.band_for_frequency(req.frequency_mhz)
    if not bands:
        raise GuardrailError(f"{req.frequency_mhz} MHz is not inside any NR band in the reference DB")
    findings = [
        GuardrailFinding(
            level="info",
            check="band_lookup",
            message=f"{req.frequency_mhz} MHz maps to {', '.join(b['band'] for b in bands)}",
        )
    ]
    lb = req.link_budget
    if lb.gnb_tx_power_dbm < 20 or lb.gnb_antenna_gain_dbi <= 0 or lb.dl_bandwidth_mhz < 5:
        findings.append(
            GuardrailFinding(
                level="warning",
                check="link_budget_sanity",
                message=(
                    f"Unusual link budget (gNB {lb.gnb_tx_power_dbm} dBm, {lb.gnb_antenna_gain_dbi} dBi, "
                    f"{lb.dl_bandwidth_mhz} MHz). Check the assumptions."
                ),
            )
        )
    if req.network_type.startswith("4G"):
        findings.append(
            GuardrailFinding(
                level="warning",
                check="technology",
                message="Reference DB covers NR bands. LTE numerology is fixed at 15 kHz.",
            )
        )
    return findings


def validate_report(
    report: NetPlanReport, req: NetworkRequest, db: ReadOnlyTelecomDB
) -> tuple[NetPlanReport, list[GuardrailFinding]]:
    findings: list[GuardrailFinding] = []

    # 1. RF ground truth. Any drift from the deterministic engine is corrected.
    truth = analyze_link(req.frequency_mhz, req.coverage_radius_km, req.link_budget)
    if abs(report.rf.path_loss_db - truth.path_loss_db) > 0.1 or report.rf.feasible != truth.feasible:
        findings.append(
            GuardrailFinding(
                level="warning",
                check="rf_consistency",
                message=(
                    f"Report RF values differed from the link budget engine "
                    f"(PL {report.rf.path_loss_db} vs {truth.path_loss_db} dB). Corrected."
                ),
            )
        )
    report = report.model_copy(update={"rf": truth, "request": req})

    # 2. Band, duplex and SCS against the reference DB
    bands = {b["band"]: b for b in db.band_for_frequency(req.frequency_mhz)}
    proto = report.protocol
    band = bands.get(proto.nr_band)
    if band is None:
        findings.append(
            GuardrailFinding(
                level="error",
                check="band",
                message=f"{proto.nr_band} does not contain {req.frequency_mhz} MHz. Valid: {', '.join(bands)}",
            )
        )
    else:
        if band["duplex"] != proto.duplex:
            findings.append(
                GuardrailFinding(
                    level="error",
                    check="duplex",
                    message=f"{proto.nr_band} is {band['duplex']}, report says {proto.duplex}",
                )
            )
        allowed = [int(s) for s in band["scs_allowed_khz"].split(",")]
        if proto.subcarrier_spacing_khz not in allowed:
            findings.append(
                GuardrailFinding(
                    level="error",
                    check="scs",
                    message=f"{proto.subcarrier_spacing_khz} kHz not allowed for {proto.nr_band} ({allowed})",
                )
            )
        elif proto.subcarrier_spacing_khz != band["scs_typical_khz"]:
            findings.append(
                GuardrailFinding(
                    level="warning",
                    check="scs",
                    message=(
                        f"{proto.subcarrier_spacing_khz} kHz is valid but {band['scs_typical_khz']} kHz "
                        f"is the usual choice for {proto.nr_band}"
                    ),
                )
            )

    # 3. Grounding. Protocol claims must cite the specs, hardware must cite a URL.
    uncited = [p.name for p in proto.parameters if not p.citations]
    if uncited:
        findings.append(
            GuardrailFinding(level="warning", check="citations", message=f"Uncited parameters: {', '.join(uncited)}")
        )
    no_url = [h.product for h in report.hardware.options if not h.source_url]
    if no_url:
        findings.append(
            GuardrailFinding(
                level="warning", check="hardware_sources", message=f"No source URL for: {', '.join(no_url)}"
            )
        )

    if not findings:
        findings.append(GuardrailFinding(level="info", check="all", message="All output checks passed"))
    return report, findings
