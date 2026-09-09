"""Immutable raw archive, content-addressed snapshots, and the lineage between them.

Nothing downstream can be replayed without this, and replay is the whole point:
the fault-injection matrix injects faults into a recorded run, not into a live
connection, and "the same snapshot must produce the same evidence" is only a
testable claim once a snapshot is a thing with an identity.

Three rules the layout enforces rather than documents.

*Raw is never overwritten.* A response is written once under its request id and
a second write with different bytes is a revision conflict, not an update. The
alternative - last-write-wins - is how a quiet correction erases the evidence
that the first answer was different, which is exactly the case a revision log
exists to catch.

*A snapshot is named by its content.* `snapshot_id` hashes the tables and the
config, never the wall clock. A changed id means changed content, which is the
only way "reproducible" can be checked instead of asserted.

*A session can be observed more than once, and the second look is a revision.*
Two fetches of the same frozen close are not guaranteed to be identical - open
interest is a best-effort tick that arrives perhaps half the time, and its
absence flips a row to DEGRADED. Hashing that away would be lying about the
data; filing it as a second session would be lying about the history. So the
same `session_date` accumulates numbered revisions that name what they
superseded and which tables moved, and the session count stays a count of
sessions.

*Code is hashed separately from data.* Same data and different code is a real
and common situation, and a manifest that could not tell the two apart would
let a behaviour change look like a data change.

This module stores and retrieves. It does not judge: quality rules and purpose
eligibility live in the validation layer, and a snapshot carries their verdicts
rather than forming its own.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Bumped when the identity rule changes, because the rule is part of the
# identity. v2 stopped hashing the transport-timing columns; without a bump the
# same close would have silently re-issued itself under a new id and looked like
# a new session.
CONTRACT_VERSION = "2"

# The four uses the plan keeps separately eligible. A snapshot approved for one
# is not thereby approved for another: a delayed book can describe the day's
# close and still be unfit to price a contract someone is about to buy.
PURPOSES = ("training", "day_end_analysis", "intraday_observation", "instrument_quotes")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")


# Fields that describe the fetch rather than the market. They are stored, because
# provenance is the point of storing anything, but they do not form identity: a
# tick that took 9.03 seconds to arrive instead of 8.71 is not a different close,
# and hashing it would give the same frozen book a new id on every run and file
# one session under several.
VOLATILE_COLUMNS = frozenset(
    {
        "tick_lag_seconds",
        "quote_age_seconds",
        "received_at_utc",
        "logged_at_utc",
        "retrieved_at_utc",
        "exchange_time_utc",
    }
)


def frame_hash(frame: pd.DataFrame, ignore: frozenset[str] = VOLATILE_COLUMNS) -> str:
    """A hash of what the table says about the market.

    Not of how parquet chose to write it - parquet embeds writer metadata, so
    identical data can produce different files - and not of how long the wire
    took, which is why `ignore` exists.
    """
    content = frame.drop(columns=[c for c in frame.columns if c in ignore], errors="ignore")
    digest = hashlib.sha256()
    digest.update(_canonical_json([list(content.columns), [str(d) for d in content.dtypes]]))
    if len(content):
        digest.update(pd.util.hash_pandas_object(content, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def code_hash(root: Path) -> str:
    """Fingerprint of the package source, so a behaviour change is not read as a data change."""
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "market_state_lab").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RequestRecord:
    """One call to a provider, with everything needed to ask it again.

    `state` is a lifecycle, not a boolean. An `end` callback proves the transfer
    stopped; it does not prove the expected dates and fields arrived, so
    `partial` exists and must never be read as `complete`.
    """

    request_id: str
    provider: str
    session_tag: str
    endpoint: str
    contract: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    requested_at_utc: str | None = None
    completed_at_utc: str | None = None
    state: str = "pending"
    error: str | None = None
    sdk_version: str | None = None
    server_version: str | None = None
    raw_sha256: str | None = None
    rows: int | None = None

    def __post_init__(self) -> None:
        allowed = {"pending", "partial", "complete", "failed", "cancelled"}
        if self.state not in allowed:
            raise ValueError(f"{self.request_id}: state must be one of {sorted(allowed)}")


class RawArchive:
    """`data/raw/{provider}/{session_date}/{request_id}.json`, written once.

    Re-writing identical bytes is a no-op, because a re-run that genuinely got
    the same answer should not be an error. Different bytes under the same id
    raise: that is a provider revision, and the caller has to record it as one
    rather than let it land silently.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, provider: str, session_date: str, request_id: str) -> Path:
        return self.root / "raw" / provider / session_date / f"{request_id}.json"

    def write(
        self,
        record: RequestRecord,
        payload: Any,
        session_date: str,
    ) -> tuple[Path, str]:
        body = _canonical_json({"record": asdict(record), "payload": payload})
        digest = _sha256(body)
        target = self.path_for(record.provider, session_date, record.request_id)
        if target.exists():
            existing = target.read_bytes()
            if _sha256(existing) == digest:
                return target, digest
            raise FileExistsError(
                f"{target} already holds different bytes for {record.request_id}; "
                "a provider revision is a new version, not an overwrite"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return target, digest

    def read(self, provider: str, session_date: str, request_id: str) -> dict[str, Any]:
        path = self.path_for(provider, session_date, request_id)
        return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ValidatedSnapshot:
    """Normalised tables plus the manifest that says what they may be used for."""

    snapshot_id: str
    session_date: str
    tables: dict[str, pd.DataFrame]
    manifest: dict[str, Any]

    def table(self, name: str) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"{self.snapshot_id}: no table {name!r}; has {sorted(self.tables)}")
        return self.tables[name]

    @property
    def eligible_for(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("eligible_for", ()))

    def require(self, purpose: str) -> None:
        """Refuse a use the snapshot was not approved for, naming the reason."""
        if purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {purpose!r}; expected one of {list(PURPOSES)}")
        if purpose not in self.eligible_for:
            raise PermissionError(
                f"{self.snapshot_id} is not eligible for {purpose}: "
                f"{self.manifest.get('ineligibility', {}).get(purpose, 'no reason recorded')}"
            )


def build_snapshot_id(
    session_date: str,
    tables: dict[str, pd.DataFrame],
    config_fingerprint: str,
) -> str:
    """Content-addressed, and deliberately blind to the clock.

    Identical content lands on the same id, which is what makes a re-run a
    no-op. Different content does not, even for the same session - that case is
    a revision, and `write_snapshot` records it as one.
    """
    digest = hashlib.sha256()
    digest.update(_canonical_json([CONTRACT_VERSION, session_date, config_fingerprint]))
    for name in sorted(tables):
        digest.update(name.encode("utf-8"))
        digest.update(frame_hash(tables[name]).encode("utf-8"))
    return f"{session_date}-{digest.hexdigest()[:12]}"


def default_eligibility(qualifications: set[str] | None) -> tuple[tuple[str, ...], dict[str, str]]:
    """What a post-close run may grant its snapshot, from the rows' qualifications.

    The question is whether the data could be judged, not whether the
    instruments passed. DEGRADED and UNAVAILABLE mean a row could not be
    assessed - a missing open interest, a frozen book while the session trades,
    an ask that never came - and a single one of those demotes the whole
    snapshot, because a comparison quietly missing one leg reads as "these are
    the choices".

    QUARANTINED does not demote it. A quarantined row is the screen succeeding:
    every field arrived and the contract was confidently rejected as too thin or
    too wide. Treating a working screen as a data fault would mean the better
    the screen got, the less the snapshot was trusted - and the plan is explicit
    that unfit *option data* stops contract ranking, which an unfit *contract*
    is not.

    `training` is never granted by one run, and `intraday_observation` never by
    a post-close one. Both are matters of series and schedule, not of data.
    """
    ineligible = {
        "intraday_observation": "post-close run on a frozen book",
        "training": "one session; a training set is granted over a series, not a run",
    }
    unjudged = (qualifications or set()) & {"DEGRADED", "UNAVAILABLE"}
    eligible: list[str] = ["day_end_analysis"]
    if qualifications and not unjudged:
        eligible.append("instrument_quotes")
    else:
        ineligible["instrument_quotes"] = (
            ("rows that could not be judged: " + ", ".join(sorted(unjudged)))
            if unjudged
            else "no data"
        )
    return tuple(eligible), ineligible


def write_snapshot(
    root: Path,
    session_date: str,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    requests: list[RequestRecord] | None = None,
    issues: list[dict[str, Any]] | None = None,
    eligible_for: tuple[str, ...] = (),
    ineligibility: dict[str, str] | None = None,
    project_root: Path | None = None,
    run_parameters: dict[str, Any] | None = None,
) -> ValidatedSnapshot:
    """Write the tables and their manifest, and never a second time.

    Re-running over unchanged inputs resolves to the same `snapshot_id` and
    leaves the existing files alone, so a daily job is idempotent and a
    genuinely new day is a genuinely new directory.
    """
    root = Path(root)
    # Run parameters are part of the config for identity purposes. A horizon or
    # a notional that changes the output but leaves no trace turns a deliberate
    # second view into an unexplained revision - which happened here, and could
    # not be diagnosed from the archive because the argument was never recorded.
    config_fingerprint = _sha256(_canonical_json({"config": config, "run": run_parameters or {}}))
    snapshot_id = build_snapshot_id(session_date, tables, config_fingerprint)
    directory = root / "validated" / snapshot_id
    manifest_path = root / "manifests" / f"{snapshot_id}.json"

    # An earlier look at the same session makes this a revision of it, not a
    # separate session. The plan keeps revision_id in the lineage for exactly
    # this: a corrected or fuller observation supersedes, it does not replace,
    # and it must never silently rewrite the earlier conclusion.
    prior = _session_revisions(root, session_date, run_parameters or {})
    if any(p["snapshot_id"] == snapshot_id for p in prior):
        existing = next(p for p in prior if p["snapshot_id"] == snapshot_id)
        # The data is unchanged, so nothing is rewritten - but the verdict about
        # it may have moved since, and leaving that frozen is its own kind of
        # lie. Amending is the one thing this path does.
        existing = _amend_verdict(
            manifest_path,
            existing,
            list(eligible_for),
            ineligibility or {},
            code_hash(project_root) if project_root else None,
        )
        return ValidatedSnapshot(snapshot_id, session_date, tables, existing)
    superseded = prior[-1] if prior else None

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "snapshot_id": snapshot_id,
        "session_date": session_date,
        "revision": len(prior) + 1,
        "supersedes": superseded["snapshot_id"] if superseded else None,
        "changed_tables": _changed_tables(superseded, tables) if superseded else None,
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_fingerprint,
        "run_parameters": run_parameters or {},
        "code_sha256": code_hash(project_root) if project_root else None,
        "tables": {name: {"rows": len(f), "sha256": frame_hash(f)} for name, f in tables.items()},
        "requests": [asdict(r) for r in (requests or [])],
        "quality_issues": issues or [],
        # An empty list is the safe default: a snapshot is approved for a use by
        # someone deciding so, never by arriving.
        "eligible_for": list(eligible_for),
        "ineligibility": ineligibility or {},
    }

    if not manifest_path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_parquet(directory / f"{name}.parquet")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
    else:  # pragma: no cover - the revision check above already returned
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return ValidatedSnapshot(snapshot_id, session_date, tables, manifest)


def _amend_verdict(
    path: Path,
    manifest: dict[str, Any],
    eligible_for: list[str],
    ineligibility: dict[str, str],
    code_sha256: str | None,
) -> dict[str, Any]:
    """Record a changed verdict on unchanged data, keeping the one it replaced.

    Identity is the data, so a rule change re-runs to the same snapshot id and
    the write is skipped - which would leave the stored verdict frozen at
    whatever the code said the first time. That happened: a snapshot kept
    "not every screened row qualified VALID: QUARANTINED" after the rule stopped
    treating a working screen as a data fault.

    The verdict is therefore amended rather than left or overwritten. The
    superseded one is appended with the code hash that produced it, so a
    correction is dated and visible instead of silently rewriting a past
    conclusion.
    """
    if (
        manifest.get("eligible_for") == eligible_for
        and manifest.get("ineligibility") == ineligibility
    ):
        return manifest
    manifest.setdefault("amendments", []).append(
        {
            "amended_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_sha256,
            "superseded_eligible_for": manifest.get("eligible_for"),
            "superseded_ineligibility": manifest.get("ineligibility"),
        }
    )
    manifest["eligible_for"] = eligible_for
    manifest["ineligibility"] = ineligibility
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def read_snapshot(root: Path, snapshot_id: str) -> ValidatedSnapshot:
    """Load a snapshot back for replay. Touches the filesystem and nothing else."""
    root = Path(root)
    manifest_path = root / "manifests" / f"{snapshot_id}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest for {snapshot_id} at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    directory = root / "validated" / snapshot_id
    tables = {
        name: pd.read_parquet(directory / f"{name}.parquet") for name in manifest["tables"]
    }
    return ValidatedSnapshot(snapshot_id, manifest["session_date"], tables, manifest)


def verify_snapshot(snapshot: ValidatedSnapshot) -> list[str]:
    """Re-hash what was loaded against what the manifest claims.

    Returns the mismatches rather than raising, so a caller can report every
    corrupted table instead of only the first.
    """
    problems: list[str] = []
    for name, expected in snapshot.manifest["tables"].items():
        if name not in snapshot.tables:
            problems.append(f"{name}: in the manifest but not loaded")
            continue
        actual = frame_hash(snapshot.tables[name])
        if actual != expected["sha256"]:
            problems.append(
                f"{name}: content hash {actual[:12]} does not match "
                f"the manifest's {expected['sha256'][:12]}"
            )
    for name in snapshot.tables:
        if name not in snapshot.manifest["tables"]:
            problems.append(f"{name}: loaded but absent from the manifest")
    return problems


def list_snapshots(root: Path) -> pd.DataFrame:
    """Every snapshot on disk, newest session first - the history TWS will not sell."""
    manifests = sorted((Path(root) / "manifests").glob("*.json"))
    rows = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "snapshot_id": manifest["snapshot_id"],
                "session_date": manifest["session_date"],
                "revision": manifest.get("revision", 1),
                "supersedes": manifest.get("supersedes"),
                "changed_tables": ",".join(manifest.get("changed_tables") or []) or None,
                "written_at_utc": manifest.get("written_at_utc"),
                "tables": len(manifest.get("tables", {})),
                "rows": sum(t["rows"] for t in manifest.get("tables", {}).values()),
                "requests": len(manifest.get("requests", [])),
                "quality_issues": len(manifest.get("quality_issues", [])),
                "eligible_for": ",".join(manifest.get("eligible_for", [])) or None,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["snapshot_id", "session_date", "revision", "supersedes",
                     "changed_tables", "written_at_utc", "tables", "rows",
                     "requests", "quality_issues", "eligible_for"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["session_date", "revision"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _session_revisions(
    root: Path,
    session_date: str,
    run_parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Prior manifests for this session *under the same run parameters*.

    Revisions are scoped to the parameters as well as the day, because a
    30-60d view and a 60-90d view of one close are not corrections of each
    other - they are two questions. Chaining them would have the second
    "supersede" the first, which reads as the earlier answer having been wrong.
    """
    wanted = run_parameters or {}
    found = []
    for path in sorted((Path(root) / "manifests").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("session_date") != session_date:
            continue
        if manifest.get("run_parameters", {}) != wanted:
            continue
        found.append(manifest)
    return sorted(found, key=lambda m: m.get("revision", 1))


def _changed_tables(previous: dict[str, Any], tables: dict[str, pd.DataFrame]) -> list[str]:
    """Which tables actually moved, so a revision says what it revised."""
    before = {name: meta["sha256"] for name, meta in previous.get("tables", {}).items()}
    after = {name: frame_hash(frame) for name, frame in tables.items()}
    return sorted(
        {name for name in before | after if before.get(name) != after.get(name)}
    )


def latest_sessions(root: Path) -> pd.DataFrame:
    """One row per session - the newest revision of each.

    `list_snapshots` shows every revision; this is the history, and a session
    looked at twice is still one session.
    """
    every = list_snapshots(root)
    if every.empty:
        return every
    newest = (
        every.sort_values(["session_date", "revision"])
        .groupby("session_date", as_index=False)
        .last()
    )
    counts = every.groupby("session_date").size().rename("revisions")
    return (
        newest.merge(counts, on="session_date")
        .sort_values("session_date", ascending=False)
        .reset_index(drop=True)
    )
