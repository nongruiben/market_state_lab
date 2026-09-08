"""Replay is the gate P2 has to pass, and a snapshot that cannot be reproduced
or that quietly absorbs a revision makes replay meaningless."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from market_state_lab.data.snapshots import (
    PURPOSES,
    RawArchive,
    RequestRecord,
    build_snapshot_id,
    frame_hash,
    latest_sessions,
    list_snapshots,
    read_snapshot,
    verify_snapshot,
    write_snapshot,
)

CONFIG = {"ibkr": {"market_data_type": 4}, "project": {"market_calendar": "XNYS"}}


def _quotes(ask: float = 7.68) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"con_id": 882140303, "strike": 730.0, "bid": 7.65, "ask": ask},
            {"con_id": 882140488, "strike": 755.0, "bid": 12.57, "ask": 12.62},
        ]
    )


def _record(request_id: str = "req-1", **over) -> RequestRecord:
    fields = {
        "request_id": request_id,
        "provider": "ibkr",
        "session_tag": "2026-09-08T09:48Z",
        "endpoint": "reqMktData",
        "contract": {"conId": 882140303, "symbol": "SPY"},
        "parameters": {"genericTicks": "100,101,106", "snapshot": False},
        "state": "complete",
    }
    fields.update(over)
    return RequestRecord(**fields)


def test_a_state_machine_not_a_boolean() -> None:
    # An `end` callback proves the transfer stopped, not that the data arrived.
    assert _record(state="partial").state == "partial"
    with pytest.raises(ValueError, match="state must be one of"):
        _record(state="done")


def test_raw_is_written_once_and_rewriting_the_same_bytes_is_a_no_op(tmp_path) -> None:
    archive = RawArchive(tmp_path)
    first, digest = archive.write(_record(), {"bid": 7.65}, "2026-09-04")
    again, digest_again = archive.write(_record(), {"bid": 7.65}, "2026-09-04")
    assert first == again and digest == digest_again
    assert archive.read("ibkr", "2026-09-04", "req-1")["payload"] == {"bid": 7.65}


def test_different_bytes_under_the_same_id_is_a_revision_not_an_update(tmp_path) -> None:
    archive = RawArchive(tmp_path)
    archive.write(_record(), {"bid": 7.65}, "2026-09-04")
    # Last-write-wins is how a quiet correction erases the evidence that the
    # first answer was different.
    with pytest.raises(FileExistsError, match="a provider revision is a new version"):
        archive.write(_record(), {"bid": 7.20}, "2026-09-04")


def test_the_snapshot_id_is_the_content_not_the_clock() -> None:
    tables = {"quotes": _quotes()}
    first = build_snapshot_id("2026-09-04", tables, "cfg")
    second = build_snapshot_id("2026-09-04", {"quotes": _quotes()}, "cfg")
    assert first == second, "the same frozen book must land on the same id"
    assert first != build_snapshot_id("2026-09-04", {"quotes": _quotes(7.70)}, "cfg")
    assert first != build_snapshot_id("2026-09-04", tables, "other-config")
    assert first.startswith("2026-09-04-")


def test_a_rerun_over_unchanged_inputs_writes_nothing_new(tmp_path) -> None:
    first = write_snapshot(tmp_path, "2026-09-04", {"quotes": _quotes()}, CONFIG)
    written_at = first.manifest["written_at_utc"]
    second = write_snapshot(tmp_path, "2026-09-04", {"quotes": _quotes()}, CONFIG)
    assert second.snapshot_id == first.snapshot_id
    # The original manifest is returned untouched; a daily job is idempotent.
    assert second.manifest["written_at_utc"] == written_at
    assert len(list(( tmp_path / "manifests").glob("*.json"))) == 1


def test_a_snapshot_round_trips_and_verifies(tmp_path) -> None:
    written = write_snapshot(
        tmp_path,
        "2026-09-04",
        {"quotes": _quotes(), "plan": pd.DataFrame([{"bucket": "30-60d", "strike": 730.0}])},
        CONFIG,
        requests=[_record()],
        eligible_for=("day_end_analysis", "instrument_quotes"),
    )
    loaded = read_snapshot(tmp_path, written.snapshot_id)
    assert verify_snapshot(loaded) == []
    assert loaded.table("quotes").equals(_quotes())
    assert loaded.manifest["requests"][0]["endpoint"] == "reqMktData"
    assert loaded.eligible_for == ("day_end_analysis", "instrument_quotes")


def test_a_tampered_table_fails_verification_instead_of_loading_quietly(tmp_path) -> None:
    written = write_snapshot(tmp_path, "2026-09-04", {"quotes": _quotes()}, CONFIG)
    loaded = read_snapshot(tmp_path, written.snapshot_id)
    loaded.tables["quotes"].loc[0, "ask"] = 99.0
    problems = verify_snapshot(loaded)
    assert len(problems) == 1 and "does not match" in problems[0]


def test_a_use_the_snapshot_was_not_approved_for_is_refused_with_the_reason(tmp_path) -> None:
    written = write_snapshot(
        tmp_path,
        "2026-09-04",
        {"quotes": _quotes()},
        CONFIG,
        eligible_for=("day_end_analysis",),
        ineligibility={"instrument_quotes": "delayed frozen book, four sessions old"},
    )
    written.require("day_end_analysis")
    with pytest.raises(PermissionError, match="four sessions old"):
        written.require("instrument_quotes")
    with pytest.raises(ValueError, match="unknown purpose"):
        written.require("trading")


def test_approval_is_granted_not_inherited_from_arriving(tmp_path) -> None:
    bare = write_snapshot(tmp_path, "2026-09-04", {"quotes": _quotes()}, CONFIG)
    assert bare.eligible_for == ()
    for purpose in PURPOSES:
        with pytest.raises(PermissionError):
            bare.require(purpose)


def test_code_and_data_hashes_are_separate_fields(tmp_path) -> None:
    written = write_snapshot(
        tmp_path, "2026-09-04", {"quotes": _quotes()}, CONFIG, project_root=tmp_path
    )
    manifest = json.loads(
        (tmp_path / "manifests" / f"{written.snapshot_id}.json").read_text(encoding="utf-8")
    )
    # Same data with changed code is a real situation, and a manifest that could
    # not tell them apart would let a behaviour change look like a data change.
    assert manifest["config_sha256"] != manifest["tables"]["quotes"]["sha256"]
    assert "code_sha256" in manifest


def test_the_frame_hash_ignores_how_parquet_chose_to_write_it(tmp_path) -> None:
    frame = _quotes()
    frame.to_parquet(tmp_path / "a.parquet")
    assert frame_hash(pd.read_parquet(tmp_path / "a.parquet")) == frame_hash(frame)


def test_the_snapshot_list_is_the_history_the_feed_will_not_sell(tmp_path) -> None:
    assert list_snapshots(tmp_path).empty
    write_snapshot(tmp_path, "2026-09-03", {"quotes": _quotes(7.40)}, CONFIG)
    write_snapshot(tmp_path, "2026-09-04", {"quotes": _quotes()}, CONFIG)
    listed = list_snapshots(tmp_path)
    assert list(listed["session_date"]) == ["2026-09-04", "2026-09-03"]
    assert set(listed["rows"]) == {2}


def test_transport_lag_does_not_make_a_new_session() -> None:
    # The same frozen close fetched twice: only the wire time differs. If that
    # changed the id, one session would be filed under several snapshots and the
    # history would count re-runs as days.
    first = _quotes().assign(tick_lag_seconds=9.03, quote_age_seconds=None)
    second = _quotes().assign(tick_lag_seconds=8.71, quote_age_seconds=None)
    assert frame_hash(first) == frame_hash(second)
    assert build_snapshot_id("2026-09-04", {"q": first}, "cfg") == build_snapshot_id(
        "2026-09-04", {"q": second}, "cfg"
    )


def test_a_changed_price_still_makes_a_new_snapshot() -> None:
    base = _quotes().assign(tick_lag_seconds=9.03)
    moved = _quotes(7.70).assign(tick_lag_seconds=9.03)
    assert frame_hash(base) != frame_hash(moved)


def test_a_rerun_with_a_different_wire_time_writes_no_second_snapshot(tmp_path) -> None:
    first = write_snapshot(
        tmp_path, "2026-09-04", {"q": _quotes().assign(tick_lag_seconds=9.03)}, CONFIG
    )
    second = write_snapshot(
        tmp_path, "2026-09-04", {"q": _quotes().assign(tick_lag_seconds=8.71)}, CONFIG
    )
    assert first.snapshot_id == second.snapshot_id
    assert len(list_snapshots(tmp_path)) == 1


def test_a_second_look_at_one_session_is_a_revision_not_a_second_session(tmp_path) -> None:
    # Open interest is a best-effort tick: present on one fetch, absent on the
    # next, flipping a row to DEGRADED. Hashing that away would lie about the
    # data; filing it as a new session would lie about the history.
    first = write_snapshot(
        tmp_path, "2026-09-04",
        {"q": _quotes().assign(open_interest=[None, 8054.0], qualification="DEGRADED")},
        CONFIG,
    )
    second = write_snapshot(
        tmp_path, "2026-09-04",
        {"q": _quotes().assign(open_interest=[9695.0, 8054.0], qualification="VALID")},
        CONFIG,
    )
    assert second.snapshot_id != first.snapshot_id
    assert second.manifest["revision"] == 2
    assert second.manifest["supersedes"] == first.snapshot_id
    assert second.manifest["changed_tables"] == ["q"]
    # Two revisions, one session.
    assert len(list_snapshots(tmp_path)) == 2
    assert len(latest_sessions(tmp_path)) == 1


def test_the_history_counts_sessions_and_says_how_often_each_was_looked_at(tmp_path) -> None:
    write_snapshot(tmp_path, "2026-09-03", {"q": _quotes(7.40)}, CONFIG)
    write_snapshot(tmp_path, "2026-09-04", {"q": _quotes()}, CONFIG)
    write_snapshot(tmp_path, "2026-09-04", {"q": _quotes(7.69)}, CONFIG)
    sessions = latest_sessions(tmp_path)
    assert list(sessions["session_date"]) == ["2026-09-04", "2026-09-03"]
    assert list(sessions["revisions"]) == [2, 1]
    # The newest revision is the one that represents the session.
    assert sessions.loc[0, "revision"] == 2


def test_an_identical_refetch_adds_no_revision(tmp_path) -> None:
    first = write_snapshot(tmp_path, "2026-09-04", {"q": _quotes()}, CONFIG)
    again = write_snapshot(tmp_path, "2026-09-04", {"q": _quotes()}, CONFIG)
    assert again.snapshot_id == first.snapshot_id
    assert again.manifest["revision"] == 1
    assert len(list_snapshots(tmp_path)) == 1


def test_the_first_revision_supersedes_nothing(tmp_path) -> None:
    only = write_snapshot(tmp_path, "2026-09-04", {"q": _quotes()}, CONFIG)
    assert only.manifest["revision"] == 1
    assert only.manifest["supersedes"] is None
    assert only.manifest["changed_tables"] is None


def test_a_superseded_revision_is_kept_not_rewritten(tmp_path) -> None:
    first = write_snapshot(tmp_path, "2026-09-04", {"q": _quotes()}, CONFIG)
    write_snapshot(tmp_path, "2026-09-04", {"q": _quotes(7.99)}, CONFIG)
    # The earlier conclusion stays readable; a correction never erases it.
    recovered = read_snapshot(tmp_path, first.snapshot_id)
    assert verify_snapshot(recovered) == []
    assert recovered.table("q").loc[0, "ask"] == 7.68
