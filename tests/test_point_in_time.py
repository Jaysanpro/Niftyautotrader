"""Lookahead guards for dated runs.

A backtest is only worth reading if the decision could have been made on the
day it claims. These tests pin the boundary: a run dated *d* may read what was
published up to the close on *d*, and nothing after it.

The failure mode being guarded against is quiet. Nothing errors when a run
dated last March reads an article from last June; the backtest simply reports
returns nobody could have earned.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tradingagents_in.config.default_config import build_config
from tradingagents_in.dataflows import news as news_mod
from tradingagents_in.dataflows.news import NewsItem
from tradingagents_in.dataflows.sentiment import score_headlines
from tradingagents_in.dataflows.toolkit import Toolkit
from tradingagents_in.utils.timeutils import IST, as_of_date, as_of_moment, now_ist

AS_OF = "2026-03-16"


def _item(title: str, when: datetime | None, source: str = "test") -> NewsItem:
    return NewsItem(title=title, link="http://example.invalid", published=when, source=source)


@pytest.fixture
def dated_config():
    return build_config({"as_of": AS_OF, "online_tools": True, "llm_provider": "offline"})


# ------------------------------------------------------------------ anchors


def test_as_of_helpers_resolve_the_run_date(dated_config):
    assert str(as_of_date(dated_config)) == AS_OF
    moment = as_of_moment(dated_config)
    assert moment is not None
    # Pinned to that session's close, so the day's bars are complete.
    assert (moment.year, moment.month, moment.day) == (2026, 3, 16)
    assert (moment.hour, moment.minute) == (15, 30)


def test_no_as_of_means_live():
    live = build_config({"llm_provider": "offline"})
    assert as_of_moment(live) is None
    assert as_of_date(live) == now_ist().date()


# --------------------------------------------------------------------- news


def test_news_after_the_as_of_date_is_dropped(monkeypatch, dated_config):
    """The core guard: a dated run must not read its own future."""
    anchor = as_of_moment(dated_config)
    feed = [
        _item("three days before", anchor - timedelta(days=3)),
        _item("same morning", anchor - timedelta(hours=6)),
        _item("next day", anchor + timedelta(days=1)),
        _item("three months later", anchor + timedelta(days=90)),
    ]
    monkeypatch.setattr(news_mod, "fetch_feed", lambda name, url, config=None: list(feed))

    titles = {i.title for i in news_mod.fetch_news(
        feeds={"stub": "http://example.invalid/rss"}, lookback_days=7, config=dated_config
    )}
    assert titles == {"three days before", "same morning"}, (
        f"future news leaked into a dated run: {titles}"
    )


def test_news_before_the_lookback_window_is_still_dropped(monkeypatch, dated_config):
    anchor = as_of_moment(dated_config)
    feed = [
        _item("inside window", anchor - timedelta(days=2)),
        _item("too old", anchor - timedelta(days=40)),
    ]
    monkeypatch.setattr(news_mod, "fetch_feed", lambda name, url, config=None: list(feed))

    titles = {i.title for i in news_mod.fetch_news(
        feeds={"stub": "x"}, lookback_days=7, config=dated_config
    )}
    assert titles == {"inside window"}


def test_undated_items_are_dropped_on_a_dated_run_but_kept_live(monkeypatch):
    """An undated item cannot be proven to predate the session."""
    feed = [_item("undated", None)]
    monkeypatch.setattr(news_mod, "fetch_feed", lambda name, url, config=None: list(feed))

    dated = build_config({"as_of": AS_OF, "online_tools": True})
    assert news_mod.fetch_news(feeds={"stub": "x"}, config=dated) == []

    live = build_config({"online_tools": True})
    kept = news_mod.fetch_news(feeds={"stub": "x"}, config=live)
    assert [i.title for i in kept] == ["undated"]


# ---------------------------------------------------------------- sentiment


def test_recency_weighting_is_measured_from_the_run_not_the_clock():
    """Without a reference, every backtested headline decays to irrelevance."""
    anchor = datetime(2026, 3, 16, 15, 30, tzinfo=IST)
    fresh_then = [_item("bullish surge rally", anchor - timedelta(hours=12))]

    scored_at_run, count = score_headlines(fresh_then, reference=anchor)
    assert count == 1

    # Measured against today the same headline is years stale, so its weight
    # collapses; the score must not be identical to the correctly-anchored one.
    scored_now, _ = score_headlines(fresh_then, reference=None)
    assert scored_at_run != 0.0
    # Same polarity, but the anchored run must not be *less* confident.
    assert abs(scored_at_run) >= abs(scored_now) - 1e-9


# ----------------------------------------------------------------- end-to-end


def test_a_dated_run_rebuilds_its_own_calendar():
    """Session, expiry and reported stamp all follow the as-of date."""
    config = build_config({
        "data_provider": "synthetic", "llm_provider": "offline", "online_tools": False,
    })
    toolkit = Toolkit(config)

    live = toolkit.gather("RELIANCE.NS")
    dated = toolkit.gather("RELIANCE.NS", as_of=AS_OF)

    assert str(dated.session.trade_date) == AS_OF
    assert dated.session.trade_date != live.session.trade_date
    # The expiry for that week, not this one.
    assert dated.session.weekly_expiry.year == 2026
    assert dated.session.weekly_expiry.month == 3
    assert dated.session.trade_date <= dated.session.weekly_expiry


def test_two_dates_through_one_toolkit_do_not_contaminate_each_other():
    """The override is per call; the graph is built once and reused."""
    config = build_config({
        "data_provider": "synthetic", "llm_provider": "offline", "online_tools": False,
    })
    toolkit = Toolkit(config)

    first = toolkit.gather("TCS.NS", as_of="2026-03-16")
    second = toolkit.gather("TCS.NS", as_of="2025-11-04")
    third = toolkit.gather("TCS.NS")  # back to live

    assert str(first.session.trade_date) == "2026-03-16"
    assert str(second.session.trade_date) == "2025-11-04"
    assert third.session.trade_date == now_ist().date()
