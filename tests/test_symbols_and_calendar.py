"""Symbol resolution and the NSE/BSE trading calendar."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from tradingagents_in.dataflows import calendar_in as cal
from tradingagents_in.dataflows.symbols import SymbolError, base_symbol, resolve, to_yf
from tradingagents_in.models.enums import Exchange, MarketPhase
from tradingagents_in.utils.timeutils import IST


class TestSymbolResolution:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("reliance", "RELIANCE.NS"),
            ("RELIANCE.NS", "RELIANCE.NS"),
            ("NSE:RELIANCE", "RELIANCE.NS"),
            ("tcs.bo", "TCS.BO"),
            ("BSE:TCS", "TCS.BO"),
        ],
    )
    def test_equity_symbols(self, raw, expected):
        assert resolve(raw).yf_symbol == expected

    def test_explicit_suffix_beats_default_exchange(self):
        # A .BO suffix must win even when NSE is requested as the default.
        assert resolve("TCS.BO", default_exchange=Exchange.NSE).exchange == Exchange.BSE

    @pytest.mark.parametrize("alias", ["nifty", "NIFTY 50", "^NSEI"])
    def test_index_aliases(self, alias):
        ref = resolve(alias)
        assert ref.is_index and ref.yf_symbol == "^NSEI"

    def test_sensex_resolves_to_bse(self):
        ref = resolve("SENSEX")
        assert ref.exchange == Exchange.BSE and ref.is_index

    def test_fno_metadata_attached(self):
        ref = resolve("RELIANCE")
        assert ref.fno_eligible and ref.lot_size and ref.sector

    def test_index_lot_size_uses_contract_code(self):
        # The index is NIFTY 50; the contract is NIFTY. Lot size keys off the latter.
        assert resolve("NIFTY").lot_size == 75

    def test_empty_symbol_rejected(self):
        with pytest.raises(SymbolError):
            resolve("   ")

    def test_helpers(self):
        assert base_symbol("nse:reliance.ns") == "RELIANCE"
        assert to_yf("TCS", Exchange.BSE) == "TCS.BO"


class TestCalendar:
    def test_weekend_is_not_a_trading_day(self):
        assert not cal.is_trading_day(date(2026, 1, 3))  # Saturday

    def test_republic_day_is_a_holiday(self):
        assert cal.is_holiday(date(2026, 1, 26))
        assert not cal.is_trading_day(date(2026, 1, 26))

    @pytest.mark.parametrize(
        ("clock", "phase"),
        [
            ((9, 5), MarketPhase.PRE_OPEN),
            ((9, 15), MarketPhase.REGULAR),
            ((12, 0), MarketPhase.REGULAR),
            ((15, 30), MarketPhase.REGULAR),
            ((15, 45), MarketPhase.CLOSING),
            ((18, 0), MarketPhase.CLOSED),
        ],
    )
    def test_session_phases(self, clock, phase):
        # 2026-07-29 is a Wednesday and not a listed holiday.
        when = datetime(2026, 7, 29, *clock, tzinfo=IST)
        assert cal.market_phase(when) == phase

    def test_minutes_to_close(self):
        when = datetime(2026, 7, 29, 15, 0, tzinfo=IST)
        assert cal.minutes_to_close(when) == 30
        assert cal.minutes_to_close(datetime(2026, 7, 29, 18, 0, tzinfo=IST)) is None

    def test_nse_weekly_expiry_is_thursday(self):
        expiry = cal.weekly_expiry(date(2026, 7, 27), exchange="NSE")
        assert expiry.weekday() == 3

    def test_bse_weekly_expiry_is_tuesday(self):
        expiry = cal.weekly_expiry(date(2026, 7, 27), exchange="BSE")
        assert expiry.weekday() == 1

    def test_expiry_rolls_back_off_a_holiday(self):
        # 2026-05-01 (Friday) is a holiday; the April monthly must not land on it.
        expiry = cal.monthly_expiry(2026, 4, "NSE")
        assert cal.is_trading_day(expiry)

    def test_monthly_expiry_is_the_last_weekly_of_the_month(self):
        expiry = cal.monthly_expiry(2026, 7, "NSE")
        assert expiry.month == 7
        assert (expiry.day + 7) > 31 or not cal.is_trading_day(expiry.replace(day=expiry.day))

    def test_session_context_is_self_consistent(self):
        ctx = cal.session_context(datetime(2026, 7, 29, 11, 0, tzinfo=IST))
        assert ctx.phase == MarketPhase.REGULAR
        assert ctx.is_trading_day
        # Mid-session, the newest *complete* daily bar is still yesterday's.
        assert ctx.last_trading_day < ctx.trade_date
        assert ctx.weekly_expiry >= ctx.trade_date
