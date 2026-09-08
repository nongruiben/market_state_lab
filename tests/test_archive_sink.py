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
