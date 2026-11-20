"""The web side of the autonomous loop.

The pieces are tested elsewhere - the screener in test_screen, the broker in
test_paper_broker, the pass in test_autopilot. What is pinned here is the
part that only exists in the web layer: the book is one file shared by every
request, and the payloads are what the console draws from.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An app whose paper book is a fresh file per test."""
    monkeypatch.setenv("ITA_LLM_PROVIDER", "offline")
    monkeypatch.setenv("ITA_DATA_PROVIDER", "synthetic")
    monkeypatch.setenv("ITA_ONLINE_TOOLS", "false")
    import web.app as app_module

    monkeypatch.setattr(app_module, "_BOOK_PATH", tmp_path / "book.json")
    return TestClient(app_module.app)


def _run_to_completion(client, payload: dict) -> dict:
    started = client.post("/api/autopilot", json=payload)
    assert started.status_code == 200, started.text
    run_id = started.json()["run_id"]
    with client.stream("GET", f"/api/runs/{run_id}/stream") as stream:
        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            import json

            event = json.loads(line[6:])
            if event.get("type") == "autopilot":
                result = event["autopilot"]
            if event.get("done") is True:
                break
    return result


def test_an_empty_book_says_so_rather_than_failing(client):
    body = client.get("/api/book").json()
    assert body["exists"] is False
    assert body["positions"] == []
    assert body["simulated"] is True


def test_a_pass_opens_positions_and_leaves_stops_resting(client):
    """An entry without its stop is the one outcome that must never happen."""
    result = _run_to_completion(client, {"universe": "NIFTYIT", "max_new_positions": 3})
    assert result["scanned"] == 10
    assert result["orders_placed"] > 0

    book = client.get("/api/book").json()
    assert book["exists"] is True
    assert book["positions"], "a pass that placed orders should hold something"
    held = {p["symbol"] for p in book["positions"]}
    stops = {o["symbol"] for o in book["open_orders"] if o["order_type"] == "SL"}
    assert held <= stops, f"positions with no resting stop: {held - stops}"


def test_a_second_pass_does_not_double_up(client):
    """The saved book is the state, so a re-run must see what it already holds."""
    first = _run_to_completion(client, {"universe": "NIFTYIT", "max_new_positions": 3})
    held = {s.split(".")[0] for s in first["selected"]}
    assert held

    second = _run_to_completion(client, {"universe": "NIFTYIT", "max_new_positions": 3})
    skipped = {s.split(".")[0] for s in second["skipped"]}
    assert held <= skipped, f"re-bought names already held: {held - skipped}"

    book = client.get("/api/book").json()
    symbols = [p["symbol"] for p in book["positions"]]
    assert len(symbols) == len(set(symbols)), "a symbol is held twice"


def test_the_book_can_be_reset(client):
    _run_to_completion(client, {"universe": "NIFTYIT", "max_new_positions": 2})
    assert client.get("/api/book").json()["exists"] is True
    assert client.delete("/api/book").json() == {"reset": True, "existed": True}
    assert client.get("/api/book").json()["exists"] is False


def test_a_misspelled_universe_is_refused_before_any_work(client):
    assert client.post("/api/autopilot", json={"universe": "Enrgy"}).status_code == 422
    assert client.post(
        "/api/autopilot", json={"universe": "NIFTYIT", "rank_by": "vibes"}
    ).status_code == 422
    assert client.get("/api/book").json()["exists"] is False


def test_two_passes_at_once_cannot_interleave(client):
    """The book is read-modify-written, so passes must serialise.

    Without the lock two concurrent passes both read the same starting cash
    and both spend it, and the book ends holding more than it can afford.
    """
    results: list[dict] = []
    errors: list[Exception] = []

    def go():
        try:
            results.append(
                _run_to_completion(client, {"universe": "NIFTYIT", "max_new_positions": 3})
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(600)

    # One may be refused for want of a slot; whichever ran must leave a sane book.
    assert not errors, errors
    book = client.get("/api/book").json()
    symbols = [p["symbol"] for p in book["positions"]]
    assert len(symbols) == len(set(symbols)), f"interleaved passes duplicated: {symbols}"
    assert book["cash_inr"] >= 0, "the book spent cash it did not have"
