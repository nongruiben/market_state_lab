"""The fault-injection matrix mutates recorded payloads. If the recorder does not
write them, or cannot tell a failed call from an empty answer, there is nothing
to inject into and replay is a word."""

from __future__ import annotations

import json

import pytest

from market_state_lab.data.ibkr import ArchiveSink


def _sink(tmp_path) -> ArchiveSink:
    return ArchiveSink(tmp_path, session_tag="c917-20260908T100000Z")


def test_request_ids_are_monotonic_within_a_generation(tmp_path) -> None:
    sink = _sink(tmp_path)
    ids = [sink.begin("reqMktData").request_id for _ in range(3)]
    assert ids[0] == "c917-20260908T100000Z-0001"
    assert ids[1].endswith("-0002") and ids[2].endswith("-0003")


def test_complete_writes_the_raw_payload_exactly_once(tmp_path) -> None:
    sink = _sink(tmp_path)
    record = sink.begin("reqMktData", contract={"con_id": 1}, parameters={"wait_seconds": 15})
    payload = [{"bid": 7.65, "ask": 7.68}]
    sink.complete(record, payload, rows=1)
    # Filed under the fetch date, whatever day the test runs on.
    dates = [p.name for p in (tmp_path / "raw" / "ibkr").iterdir()]
    assert len(dates) == 1
    stored = json.loads(
        (tmp_path / "raw" / "ibkr" / dates[0] / f"{record.request_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["payload"] == payload
    assert stored["record"]["endpoint"] == "reqMktData"
    assert stored["record"]["state"] == "complete"
    assert stored["record"]["rows"] == 1
    assert stored["record"]["contract"] == {"con_id": 1}
    # The hash of the file cannot be embedded in the file it hashes; it lives on
    # the live record, where it points at the archive.
    assert sink.frame().iloc[0]["raw_sha256"]


def test_failed_requests_are_recorded_but_write_no_raw(tmp_path) -> None:
    sink = _sink(tmp_path)
    record = sink.begin("reqHistoricalData")
    sink.fail(record, "Error 162: no data")
    assert sink.frame().iloc[0]["state"] == "failed"
    assert "Error 162" in sink.frame().iloc[0]["error"]
    assert not (tmp_path / "raw").exists()


def test_an_empty_answer_is_complete_not_failed(tmp_path) -> None:
    sink = _sink(tmp_path)
    record = sink.begin("reqHistoricalData")
    sink.complete(record, [], rows=0)
    row = sink.frame().iloc[0]
    assert row["state"] == "complete"
    assert row["rows"] == 0


def test_two_connections_are_two_generations(tmp_path) -> None:
    first = _sink(tmp_path).begin("reqMktData")
    second = ArchiveSink(tmp_path, session_tag="c917-20260908T100500Z").begin("reqMktData")
    assert first.request_id != second.request_id
    assert first.session_tag != second.session_tag


def test_rewriting_the_same_request_with_different_bytes_raises(tmp_path) -> None:
    # The recorder surfaces RawArchive's refusal: a provider revision is a new
    # version, never a silent overwrite.
    sink = _sink(tmp_path)
    record = sink.begin("reqCurrentTime")
    sink.complete(record, {"skew_seconds": 0.2})
    with pytest.raises(FileExistsError, match="revision"):
        sink.complete(record, {"skew_seconds": -0.4})


class _FakeTicker:
    """Only what the retry path reads."""

    def __init__(self, put_oi=float("nan"), call_oi=float("nan")):
        self.putOpenInterest = put_oi  # noqa: N803 - mirrors ib_async
        self.callOpenInterest = call_oi  # noqa: N803


class _FakeIB:
    """Records what was re-subscribed, and hands back what it was told to."""

    def __init__(self, answers):
        self.answers = answers
        self.requested: list[tuple] = []
        self.cancelled: list = []

    def reqMktData(self, contract, ticks, *_):  # noqa: N802 - mirrors ib_async
        self.requested.append((contract, ticks))
        return self.answers[contract]

    def sleep(self, _seconds):
        return None

    def cancelMktData(self, contract):  # noqa: N802 - mirrors ib_async
        self.cancelled.append(contract)


def _client_with(fake, tmp_path):
    from market_state_lab.data.ibkr import ArchiveSink, ReadOnlyIBKRClient

    client = ReadOnlyIBKRClient.__new__(ReadOnlyIBKRClient)
    client.ib = fake
    client._require = lambda: fake
    client.archive = ArchiveSink(tmp_path, session_tag="g")
    return client


def test_a_missing_open_interest_is_re_asked_for_and_filled(tmp_path) -> None:
    fake = _FakeIB({"c1": _FakeTicker(put_oi=9695.0)})
    client = _client_with(fake, tmp_path)
    rows = [{"sec_type": "OPT", "right": "P", "con_id": 1, "open_interest": None}]
    client._retry_open_interest([(0, "c1")], rows, 0.0, attempt=1)
    assert rows[0]["open_interest"] == 9695.0
    # Chasing is visible: a value that needed a second ask says so.
    assert rows[0]["open_interest_attempt"] == 1
    # Only tick 101 is re-requested, and the line is closed again.
    assert fake.requested == [("c1", "101")]
    assert fake.cancelled == ["c1"]


def test_what_never_arrives_stays_absent_rather_than_becoming_zero(tmp_path) -> None:
    fake = _FakeIB({"c1": _FakeTicker()})  # still nothing
    client = _client_with(fake, tmp_path)
    rows = [{"sec_type": "OPT", "right": "P", "con_id": 1, "open_interest": None}]
    client._retry_open_interest([(0, "c1")], rows, 0.0, attempt=1)
    assert rows[0]["open_interest"] is None
    assert "open_interest_attempt" not in rows[0]


def test_the_retry_reads_the_side_matching_the_contract(tmp_path) -> None:
    # A put's callOpenInterest is 0 by construction, not a fact about the market.
    fake = _FakeIB({"c1": _FakeTicker(put_oi=float("nan"), call_oi=0.0)})
    client = _client_with(fake, tmp_path)
    rows = [{"sec_type": "OPT", "right": "P", "con_id": 1, "open_interest": None}]
    client._retry_open_interest([(0, "c1")], rows, 0.0, attempt=1)
    assert rows[0]["open_interest"] is None


def test_the_retry_is_archived_as_its_own_request(tmp_path) -> None:
    fake = _FakeIB({"c1": _FakeTicker(put_oi=120.0)})
    client = _client_with(fake, tmp_path)
    rows = [{"sec_type": "OPT", "right": "P", "con_id": 1, "open_interest": None}]
    client._retry_open_interest([(0, "c1")], rows, 0.0, attempt=1)
    record = client.archive.records[-1]
    assert record.parameters["purpose"] == "open_interest_retry"
    assert record.parameters["attempt"] == 1
    assert record.state == "complete" and record.rows == 1
