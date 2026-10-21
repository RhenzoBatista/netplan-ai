import math

import pytest

from netplan.rf import analyze_link, fspl_db, noise_floor_dbm, path_loss_db, uma_los_db, uma_nlos_db
from netplan.schemas import LinkBudgetParams


def test_fspl_matches_reference_value():
    # 3500 MHz at 2.5 km, the case from the original notebook
    assert fspl_db(3500, 2.5) == pytest.approx(111.28, abs=0.01)


def test_fspl_rejects_invalid_input():
    with pytest.raises(ValueError):
        fspl_db(0, 1)


def test_noise_floor_100mhz():
    # -174 + 10log10(100e6) = -94 dBm, plus 7 dB NF
    assert noise_floor_dbm(100, 7) == pytest.approx(-87.0, abs=0.01)


def test_uma_breakpoint_continuity():
    # d'_BP for 25 m / 1.5 m at 3.5 GHz is about 560 m; PL must be continuous there
    d_bp = 4 * 24 * 0.5 * 3.5e9 / 299_792_458
    before, after = uma_los_db(d_bp - 0.01, 3.5), uma_los_db(d_bp + 0.01, 3.5)
    assert abs(after - before) < 0.05


def test_nlos_never_below_los():
    for d in (50, 300, 1000, 3000):
        assert uma_nlos_db(d, 3.5) >= uma_los_db(d, 3.5)


def test_path_loss_increases_with_distance_and_frequency():
    p = LinkBudgetParams()
    assert path_loss_db("UMa_NLOS", 3500, 2.0, p.gnb_height_m, p.ue_height_m) > path_loss_db(
        "UMa_NLOS", 3500, 1.0, p.gnb_height_m, p.ue_height_m
    )
    assert path_loss_db("UMa_NLOS", 3500, 1.0, 25, 1.5) > path_loss_db("UMa_NLOS", 700, 1.0, 25, 1.5)


def test_link_budget_is_self_consistent():
    r = analyze_link(3500, 2.5)
    for d in (r.downlink, r.uplink):
        assert d.margin_db == pytest.approx(d.max_allowed_path_loss_db - r.path_loss_db, abs=0.02)
    assert r.limiting_link == min((r.downlink, r.uplink), key=lambda x: x.margin_db).direction
    assert r.feasible == (min(r.downlink.margin_db, r.uplink.margin_db) >= 0)


def test_uplink_limits_urban_macro_at_3500():
    # Classic result: 23 dBm UE uplink limits 3.5 GHz macro coverage
    r = analyze_link(3500, 2.5)
    assert r.limiting_link == "UL"
    assert not r.feasible
    assert r.max_cell_radius_km < 2.5
    assert r.estimated_sites and r.estimated_sites > 1


def test_max_radius_is_where_margin_hits_zero():
    r = analyze_link(3500, 1.0)
    at_edge = analyze_link(3500, r.max_cell_radius_km)
    limiting = min(at_edge.downlink.margin_db, at_edge.uplink.margin_db)
    assert math.isclose(limiting, 0, abs_tol=0.05)


def test_fspl_scenario_is_optimistic():
    fs = analyze_link(3500, 2.5, LinkBudgetParams(scenario="FSPL"))
    nlos = analyze_link(3500, 2.5, LinkBudgetParams(scenario="UMa_NLOS"))
    assert fs.path_loss_db < nlos.path_loss_db
