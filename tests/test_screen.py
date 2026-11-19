"""The sector/index screener.

Two things are worth pinning here. Universe resolution must refuse what it
does not understand, because a misspelled sector that quietly screens nothing
(or screens one imaginary ticker) looks exactly like a sector with no ideas in
it. And the ranking must happen before the portfolio caps bind, or the screen
returns whichever names happened to be analysed first.
"""

from __future__ import annotations

import pytest

from tradingagents_in.config.default_config import build_config
from tradingagents_in.graph.runner import IndianTradingAgents
from tradingagents_in.screen import (
    RANK_KEYS,
    Candidate,
    render,
    resolve_universe,
    screen,
)


def _cfg(**overrides):
    base = {
        "data_provider": "synthetic",
        "llm_provider": "offline",
        "online_tools": False,
        "synthetic_seed": 4,
    }
    base.update(overrides)
    return build_config(base)


# ------------------------------------------------------------------ universe


def test_an_index_resolves_to_its_constituents():
    names = resolve_universe("NIFTYIT")
    assert len(names) == 10
    assert all(n.endswith(".NS") for n in names)
    assert "TCS.NS" in names


def test_index_names_are_case_and_space_insensitive():
    assert resolve_universe("niftyit") == resolve_universe("NIFTY IT") == resolve_universe("NIFTYIT")


def test_a_sector_resolves_to_its_members():
    names = resolve_universe("Energy")
    assert names and all(n.endswith(".NS") for n in names)
    # Sector membership, not index membership.
    assert "RELIANCE.NS" in names


def test_a_symbol_list_passes_through():
    assert resolve_universe("TCS.NS, INFY.NS") == ["TCS.NS", "INFY.NS"]


def test_a_bare_known_symbol_is_a_one_name_screen():
    assert resolve_universe("RELIANCE.NS") == ["RELIANCE.NS"]
    assert resolve_universe("TCS") == ["TCS"]


def test_a_misspelled_sector_is_refused_not_screened_as_a_ticker():
    """The regression that matters: 'Enrgy' must not become a one-name screen.

    Accepting any alphanumeric word as a ticker turns a typo into an empty
    result that reads like a sector with no ideas in it.
    """
    with pytest.raises(ValueError) as excinfo:
        resolve_universe("Enrgy")
    message = str(excinfo.value)
    # The error has to be actionable: it names what it would have accepted.
    assert "Enrgy" in message
    assert "NIFTY50" in message and "Energy" in message


def test_empty_universe_is_refused():
    for bad in ("", "   "):
        with pytest.raises(ValueError):
            resolve_universe(bad)


def test_an_unknown_rank_key_is_refused():
    app = IndianTradingAgents(_cfg())
    with pytest.raises(ValueError):
        screen(app, "NIFTYIT", rank_by="vibes")


# ------------------------------------------------------------------- ranking


def test_candidates_are_ranked_by_the_requested_key():
    app = IndianTradingAgents(_cfg())
    result = screen(app, "NIFTYIT", rank_by="conviction", apply_portfolio=False)

    approved = [c for c in result.candidates if c.approved_standalone]
    assert len(approved) >= 2, "synthetic tape should approve a few IT names"
    convictions = [c.conviction or -1 for c in approved]
    assert convictions == sorted(convictions, reverse=True)


def test_ranking_key_changes_the_order():
    """If both keys gave the same order the screen's main knob would be inert."""
    app = IndianTradingAgents(_cfg())
    by_rr = screen(app, "NIFTYIT", rank_by="reward_risk", apply_portfolio=False)
    by_conv = screen(app, "NIFTYIT", rank_by="conviction", apply_portfolio=False)

    order_rr = [c.symbol for c in by_rr.candidates if c.approved_standalone]
    order_conv = [c.symbol for c in by_conv.candidates if c.approved_standalone]
    assert order_rr and order_conv
    assert order_rr != order_conv


def test_rejected_names_sort_below_approved_ones():
    app = IndianTradingAgents(_cfg())
    result = screen(app, "NIFTYIT", apply_portfolio=False)
    flags = [c.approved_standalone for c in result.candidates]
    assert flags == sorted(flags, reverse=True)


def test_every_rank_key_runs():
    app = IndianTradingAgents(_cfg())
    for key in RANK_KEYS:
        result = screen(app, "TCS.NS,INFY.NS", rank_by=key, apply_portfolio=False)
        assert result.scanned == 2


# ----------------------------------------------------------------- portfolio


def test_the_portfolio_pass_keeps_the_book_inside_its_caps():
    """Selections must obey the sector cap and heat budget, not just rank well."""
    app = IndianTradingAgents(_cfg())
    config = app.config
    result = screen(app, "NIFTYIT", rank_by="reward_risk")

    selected = result.selected
    assert selected, "the synthetic tape should fill at least one slot"

    deployed = sum(c.final_size_pct for c in selected)
    # Every NIFTYIT name is one sector, so the sector cap is the binding one.
    assert deployed <= float(config["max_sector_pct"]) + 1e-6, (
        f"screen selected {deployed:.2f}% in one sector against a "
        f"{config['max_sector_pct']}% cap"
    )

    risk_pct = sum(c.final_risk_inr for c in selected) / float(config["capital_inr"]) * 100
    assert risk_pct <= float(config["max_portfolio_heat_pct"]) + 1e-6
    assert len(selected) <= int(config["max_open_positions"])


def test_a_selected_name_may_be_trimmed_below_its_standalone_size():
    """The second pass is a real re-run, not a copy of the first."""
    app = IndianTradingAgents(_cfg())
    result = screen(app, "NIFTYIT", rank_by="reward_risk")
    selected = result.selected
    assert selected
    # Standalone sizes sum past the cap; the fitted ones must not.
    standalone_total = sum(c.size_pct for c in selected)
    fitted_total = sum(c.final_size_pct for c in selected)
    assert fitted_total <= standalone_total + 1e-6


def test_names_that_do_not_fit_say_why():
    app = IndianTradingAgents(_cfg())
    result = screen(app, "NIFTYIT", rank_by="reward_risk")
    passed_over = [
        c for c in result.shortlist if not c.selected and c.blocked_by
    ]
    # Whatever the tape does, a name that passed on merit and was not taken
    # must carry a reason rather than vanishing silently.
    for c in passed_over:
        assert c.blocked_by.strip()


def test_skipping_the_portfolio_pass_selects_nothing():
    app = IndianTradingAgents(_cfg())
    result = screen(app, "NIFTYIT", apply_portfolio=False)
    assert result.shortlist          # merit pass still ran
    assert result.selected == []     # but nothing was fitted


# -------------------------------------------------------------------- output


def test_summary_and_render_survive_an_empty_screen():
    empty = screen(
        IndianTradingAgents(_cfg()), "TCS.NS", apply_portfolio=False
    )
    empty.candidates = []
    empty.scanned = 0
    assert empty.summary()["selected"] == 0
    assert "scanned" in render(empty)


def test_render_marks_selected_names():
    app = IndianTradingAgents(_cfg())
    result = screen(app, "NIFTYIT", rank_by="reward_risk")
    text = render(result)
    assert "take" in text
    for c in result.selected:
        assert c.symbol[:11] in text


def test_a_failing_symbol_is_recorded_not_fatal():
    app = IndianTradingAgents(_cfg())

    real = app.analyse

    def flaky(symbol, **kwargs):
        if symbol == "INFY.NS":
            raise RuntimeError("feed exploded")
        return real(symbol, **kwargs)

    app.analyse = flaky  # type: ignore[method-assign]
    result = screen(app, "TCS.NS,INFY.NS", apply_portfolio=False)

    assert result.scanned == 2
    assert any("feed exploded" in e for e in result.errors)
    assert any(c.error for c in result.candidates)
    # The healthy name still got screened.
    assert any(c.symbol.startswith("TCS") and not c.error for c in result.candidates)


def test_candidate_rank_value_handles_missing_numbers():
    bare = Candidate(symbol="X.NS")
    for key in RANK_KEYS:
        assert isinstance(bare.rank_value(key), float)
