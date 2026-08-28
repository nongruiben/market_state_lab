from __future__ import annotations

import io
import json
import random
import re
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from market_state_lab.config import project_path
from market_state_lab.data.http import CachedHttpClient


@dataclass
class PublicDataBundle:
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)
    vix: pd.DataFrame = field(default_factory=pd.DataFrame)
    ofr: pd.DataFrame = field(default_factory=pd.DataFrame)
    french: pd.DataFrame = field(default_factory=pd.DataFrame)
    etf_close: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: pd.DataFrame = field(default_factory=pd.DataFrame)


class PublicDataLoader:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        data_config = config["data"]
        self.client = CachedHttpClient(
            project_path(config, "data", "raw", "http_cache"),
            timeout_seconds=int(data_config.get("request_timeout_seconds", 30)),
            retries=int(data_config.get("request_retries", 3)),
            max_age_hours=float(data_config.get("cache_max_age_hours", 24)),
        )
        self.start_date = pd.Timestamp(config["project"]["start_date"])
        self._manifest: list[dict[str, Any]] = []

    @staticmethod
    def _clean_index(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result.index = pd.to_datetime(result.index, errors="coerce")
        result = result.loc[~result.index.isna()]
        result.index = result.index.tz_localize(None).normalize()
        result = result[~result.index.duplicated(keep="last")].sort_index()
        return result

    def _record(
        self,
        dataset: str,
        provider: str,
        frame: pd.DataFrame | None,
        status: str,
        error: str = "",
    ) -> None:
        rows = 0 if frame is None else len(frame)
        latest = "" if frame is None or frame.empty else str(frame.index.max().date())
        earliest = "" if frame is None or frame.empty else str(frame.index.min().date())
        self._manifest.append(
            {
                "dataset": dataset,
                "provider": provider,
                "status": status,
                "rows": rows,
                "earliest_date": earliest,
                "latest_date": latest,
                "error": error,
            }
        )

    def _fred(self) -> pd.DataFrame:
        section = self.config["data"]["fred"]
        frames: list[pd.Series] = []
        for name, series_id in section.get("series", {}).items():
            # FRED's graph endpoint responds much faster without a server-side
            # start-date filter; the small full history is filtered locally.
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode(
                {"id": series_id}
            )
            try:
                result = self.client.get(url, f"fred_{series_id}.csv")
                raw = pd.read_csv(io.BytesIO(result.content))
                raw.columns = [str(column).strip() for column in raw.columns]
                date_column = raw.columns[0]
                value_column = raw.columns[-1]
                series = pd.to_numeric(raw[value_column], errors="coerce")
                series.index = pd.to_datetime(raw[date_column], errors="coerce")
                series.name = name
                frame = self._clean_index(series.to_frame()).loc[self.start_date :]
                frames.append(frame[name])
                self._record(name, "FRED", frame, result.status)
            except Exception as exc:  # One missing macro series must not stop the run.
                self._record(name, "FRED", None, "failed", str(exc))
        return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    def _vix(self) -> pd.DataFrame:
        url = self.config["data"]["cboe"]["vix_url"]
        try:
            result = self.client.get(url, "cboe_vix.csv")
            raw = pd.read_csv(io.BytesIO(result.content))
            raw.columns = [str(column).strip().lower() for column in raw.columns]
            date_column = next(column for column in raw.columns if "date" in column)
            raw = raw.set_index(date_column)
            raw = raw.rename(columns={column: f"vix_{column}" for column in raw.columns})
            frame = self._clean_index(raw).apply(pd.to_numeric, errors="coerce")
            frame = frame.loc[self.start_date :]
            self._record("vix", "CBOE", frame, result.status)
            return frame
        except Exception as exc:
            self._record("vix", "CBOE", None, "failed", str(exc))
            return pd.DataFrame()

    def _ofr(self) -> pd.DataFrame:
        url = self.config["data"]["ofr"]["fsi_url"]
        try:
            result = self.client.get(url, "ofr_fsi.csv")
            raw = pd.read_csv(io.BytesIO(result.content))
            raw.columns = [re.sub(r"\W+", "_", str(column).strip().lower()).strip("_") for column in raw.columns]
            date_column = next(column for column in raw.columns if "date" in column)
            frame = self._clean_index(raw.set_index(date_column))
            frame = frame.apply(pd.to_numeric, errors="coerce").loc[self.start_date :]
            frame = frame.add_prefix("ofr_")
            self._record("financial_stress_index", "OFR", frame, result.status)
            return frame
        except Exception as exc:
            self._record("financial_stress_index", "OFR", None, "failed", str(exc))
            return pd.DataFrame()

    @staticmethod
    def _parse_french_zip(content: bytes) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            member = next(name for name in archive.namelist() if name.lower().endswith((".csv", ".txt")))
            text = archive.read(member).decode("cp1252")
        lines = text.splitlines()
        header_index = next(
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith(",") and len(line.split(",")) >= 2
        )
        header = "date" + lines[header_index]
        rows: list[str] = []
        for line in lines[header_index + 1 :]:
            first = line.split(",", 1)[0].strip()
            if re.fullmatch(r"\d{8}", first):
                rows.append(line)
            elif rows:
                break
        frame = pd.read_csv(io.StringIO("\n".join([header, *rows])))
        frame = frame.set_index("date")
        frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d", errors="coerce")
        frame.columns = [
            re.sub(r"\W+", "_", str(column).strip().lower()).strip("_")
            for column in frame.columns
        ]
        return frame.apply(pd.to_numeric, errors="coerce") / 100.0

    def _french(self) -> pd.DataFrame:
        section = self.config["data"]["french"]
        frames: list[pd.DataFrame] = []
        for name, filename in section.get("datasets", {}).items():
            url = f"{section['base_url'].rstrip('/')}/{filename}"
            try:
                result = self.client.get(url, filename)
                frame = self._clean_index(self._parse_french_zip(result.content))
                frame = frame.loc[self.start_date :]
                if name != "ff5":
                    frame = frame.rename(columns={frame.columns[0]: name})[[name]]
                frames.append(frame)
                self._record(name, "Kenneth French", frame, result.status)
            except Exception as exc:
                self._record(name, "Kenneth French", None, "failed", str(exc))
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, axis=1)
        return combined.loc[:, ~combined.columns.duplicated()].sort_index()

    def _nasdaq(self) -> pd.DataFrame:
        section = self.config["data"]["nasdaq"]
        frames: list[pd.Series] = []
        last_request = 0.0
        for name, symbol in section.get("symbols", {}).items():
            elapsed = time.monotonic() - last_request
            interval = float(section.get("min_request_interval_seconds", 1.5))
            interval += random.uniform(0.0, float(section.get("max_jitter_seconds", 0.5)))
            wait_for = interval - elapsed
            if wait_for > 0:
                time.sleep(wait_for)
            end_date = pd.Timestamp.today().normalize()
            start_date = max(
                self.start_date,
                end_date - pd.DateOffset(years=int(section.get("history_years", 10))),
            )
            query = urlencode(
                {
                    "assetclass": "etf",
                    "fromdate": start_date.date().isoformat(),
                    "todate": end_date.date().isoformat(),
                    "limit": 5000,
                }
            )
            url = f"{section['base_url'].rstrip('/')}/{symbol}/historical?{query}"
            headers = {
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com",
                "Referer": "https://www.nasdaq.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
                ),
            }
            try:
                result = self.client.get(url, f"nasdaq_{name}.json", headers=headers)
                last_request = time.monotonic()
                payload = json.loads(result.content)
                data = payload.get("data") or {}
                rows = ((data.get("tradesTable") or {}).get("rows") or [])
                if not rows:
                    status = payload.get("status") or {}
                    raise ValueError(f"Nasdaq response contains no historical rows: {status}")
                raw = pd.DataFrame(rows)
                close = (
                    raw["close"]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                )
                series = pd.to_numeric(close, errors="coerce")
                series.index = pd.to_datetime(raw["date"], errors="coerce")
                series.name = name
                frame = self._clean_index(series.to_frame()).loc[self.start_date :]
                frames.append(frame[name])
                self._record(name, "Nasdaq", frame, result.status)
            except Exception as exc:
                if "429" in str(exc):
                    time.sleep(float(section.get("cooldown_seconds", 60)))
                self._record(name, "Nasdaq", None, "failed", str(exc))
        return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    def load(self) -> PublicDataBundle:
        data = self.config["data"]
        macro = self._fred() if data.get("fred", {}).get("enabled", True) else pd.DataFrame()
        vix = self._vix() if data.get("cboe", {}).get("enabled", True) else pd.DataFrame()
        ofr = self._ofr() if data.get("ofr", {}).get("enabled", True) else pd.DataFrame()
        french = self._french() if data.get("french", {}).get("enabled", True) else pd.DataFrame()
        etf_close = self._nasdaq() if data.get("nasdaq", {}).get("enabled", True) else pd.DataFrame()
        manifest = pd.DataFrame(self._manifest)
        return PublicDataBundle(macro, vix, ofr, french, etf_close, manifest)
