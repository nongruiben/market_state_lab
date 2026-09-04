from __future__ import annotations

import io
import json
import os
import random
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

import pandas as pd

from market_state_lab.config import project_path
from market_state_lab.data.alfred import load_initial_release_series
from market_state_lab.data.http import CachedHttpClient

_SECRET_QUERY_PARAMETERS = ("api_key", "apikey", "token")


def _redact_secrets(text: str) -> str:
    """Strip credentials out of anything headed for the manifest.

    A failed ALFRED request raises with the full request URL attached, and that
    string is written to data_manifest.csv and rendered into the dashboard HTML -
    a file people share.
    """
    if not text:
        return text
    pattern = "|".join(_SECRET_QUERY_PARAMETERS)
    return re.sub(rf"(?i)\b({pattern})=[^&\s\"']*", r"\1=***", text)


@dataclass
class PublicDataBundle:
    macro: pd.DataFrame = field(default_factory=pd.DataFrame)
    vix: pd.DataFrame = field(default_factory=pd.DataFrame)
    ofr: pd.DataFrame = field(default_factory=pd.DataFrame)
    french: pd.DataFrame = field(default_factory=pd.DataFrame)
    etf_close: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifest: pd.DataFrame = field(default_factory=pd.DataFrame)
    point_in_time_status: str = "latest_vintage"


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
        vintage_mode: str = "not_applicable",
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
                "vintage_mode": vintage_mode,
                "error": _redact_secrets(error),
            }
        )

    def _fred_graph_series(self, series_id: str, name: str) -> tuple[pd.DataFrame, str]:
        """Keyless graph CSV. Full history for most series, but the licensed ICE
        BofA spreads come back capped at a rolling ~3-year window regardless of
        cosd/coed, which is why the coverage rules in source_health exist."""
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": series_id})
        result = self.client.get(url, f"fred_{series_id}.csv")
        raw = pd.read_csv(io.BytesIO(result.content))
        raw.columns = [str(column).strip() for column in raw.columns]
        series = pd.to_numeric(raw[raw.columns[-1]], errors="coerce")
        series.index = pd.to_datetime(raw[raw.columns[0]], errors="coerce")
        series.name = name
        return self._clean_index(series.to_frame()).loc[self.start_date :], result.status

    def _fred_api_series(self, series_id: str, name: str, api_key: str) -> tuple[pd.DataFrame, str]:
        """Official observations endpoint. Returns the complete series, including
        the ICE BofA spreads the graph CSV truncates."""
        base = str(
            self.config["data"]["fred"].get(
                "api_base_url", "https://api.stlouisfed.org/fred/series/observations"
            )
        )
        url = base + "?" + urlencode(
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": self.start_date.date().isoformat(),
                "limit": 100000,
            }
        )
        result = self.client.get(url, f"fred_api_{series_id}.json")
        payload: dict[str, Any] = json.loads(result.content.decode("utf-8"))
        raw = pd.DataFrame(payload.get("observations", []))
        if raw.empty:
            raise ValueError(f"FRED API returned no observations for {series_id}")
        series = pd.to_numeric(raw["value"], errors="coerce")
        series.index = pd.to_datetime(raw["date"], errors="coerce")
        series.name = name
        frame = self._clean_index(series.to_frame()).dropna()
        return frame.loc[self.start_date :], result.status

    def _fred(self) -> pd.DataFrame:
        section = self.config["data"]["fred"]
        vintage_mode = str(section.get("vintage_mode", "latest"))
        api_key = os.getenv("FRED_API_KEY", "").strip()
        use_api = bool(section.get("prefer_api", True)) and bool(api_key)
        frames: list[pd.Series] = []
        for name, series_id in section.get("series", {}).items():
            try:
                if vintage_mode == "point_in_time":
                    frame = load_initial_release_series(
                        self.client, str(series_id), str(name), self.start_date
                    )
                    provider = "ALFRED"
                    status = "success"
                    record_vintage = "initial_release"
                elif use_api:
                    try:
                        frame, status = self._fred_api_series(str(series_id), str(name), api_key)
                        provider = "FRED API"
                    except Exception:
                        frame, status = self._fred_graph_series(str(series_id), str(name))
                        provider = "FRED graph CSV (API fallback)"
                    record_vintage = "latest_revised"
                else:
                    frame, status = self._fred_graph_series(str(series_id), str(name))
                    provider = "FRED graph CSV"
                    record_vintage = "latest_revised"
                frames.append(frame[name])
                self._record(name, provider, frame, status, vintage_mode=record_vintage)
            except Exception as exc:  # One missing macro series must not stop the run.
                provider = "ALFRED" if vintage_mode == "point_in_time" else "FRED"
                self._record(
                    name,
                    provider,
                    None,
                    "failed",
                    str(exc),
                    "initial_release" if vintage_mode == "point_in_time" else "latest_revised",
                )
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

    def _cboe_indices(self) -> pd.DataFrame:
        """Other CBOE volatility indices, same feed and licence as the VIX.

        SKEW is the one that matters here: it prices the implied left tail, which
        is where this model's information actually lives. It also has history back
        to 1990, unlike VIX9D (2011) and VIX3M (2009), so only SKEW is deep enough
        to sit inside risk_score without reintroducing composition drift.
        """
        section = self.config["data"].get("cboe", {})
        frames: list[pd.Series] = []
        for name, url in (section.get("indices", {}) or {}).items():
            try:
                result = self.client.get(url, f"cboe_{name}.csv")
                raw = pd.read_csv(io.BytesIO(result.content))
                raw.columns = [str(column).strip().lower() for column in raw.columns]
                date_column = next(column for column in raw.columns if "date" in column)
                close_column = next(
                    (column for column in raw.columns if "close" in column),
                    raw.columns[-1],
                )
                series = pd.to_numeric(raw[close_column], errors="coerce")
                series.index = pd.to_datetime(raw[date_column], errors="coerce")
                series.name = str(name)
                frame = self._clean_index(series.to_frame()).loc[self.start_date :].dropna()
                frames.append(frame[str(name)])
                self._record(str(name), "CBOE", frame, result.status)
            except Exception as exc:
                self._record(str(name), "CBOE", None, "failed", str(exc))
        return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

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

    def _stooq(self, symbols: dict[str, str]) -> pd.DataFrame:
        section = self.config["data"]["stooq"]
        frames: list[pd.Series] = []
        end_date = pd.Timestamp.today().normalize()
        for name, symbol in symbols.items():
            query = urlencode(
                {
                    "s": f"{str(symbol).lower()}.us",
                    "d1": self.start_date.strftime("%Y%m%d"),
                    "d2": end_date.strftime("%Y%m%d"),
                    "i": "d",
                }
            )
            url = f"{section['base_url']}?{query}"
            try:
                result = self.client.get(url, f"stooq_{name}.csv")
                raw = pd.read_csv(io.BytesIO(result.content))
                raw.columns = [str(column).strip().lower() for column in raw.columns]
                if "date" not in raw or "close" not in raw:
                    raise ValueError("Stooq response has no Date/Close columns")
                series = pd.to_numeric(raw["close"], errors="coerce")
                series.index = pd.to_datetime(raw["date"], errors="coerce")
                series.name = name
                frame = self._clean_index(series.to_frame()).loc[self.start_date :]
                if frame.empty:
                    raise ValueError("Stooq response contains no usable rows")
                frames.append(frame[name])
                self._record(name, "Stooq", frame, result.status)
            except Exception as exc:
                self._record(name, "Stooq", None, "failed", str(exc))
        return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    def _nasdaq(self, symbols: dict[str, str]) -> pd.DataFrame:
        section = self.config["data"]["nasdaq"]
        frames: list[pd.Series] = []
        last_request = 0.0
        for name, symbol in symbols.items():
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

    def _yahoo(self, symbols: dict[str, str]) -> pd.DataFrame:
        section = self.config["data"]["yahoo"]
        frames: list[pd.Series] = []
        last_request = 0.0
        period1 = int(self.start_date.tz_localize("UTC").timestamp())
        period2 = int((pd.Timestamp.now(tz="UTC") + timedelta(days=1)).timestamp())
        headers = {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            ),
        }
        for name, symbol in symbols.items():
            elapsed = time.monotonic() - last_request
            interval = float(section.get("min_request_interval_seconds", 1.0))
            interval += random.uniform(0.0, float(section.get("max_jitter_seconds", 0.4)))
            if interval > elapsed:
                time.sleep(interval - elapsed)
            query = urlencode(
                {
                    "period1": period1,
                    "period2": period2,
                    "interval": "1d",
                    "events": "history",
                }
            )
            url = f"{section['base_url'].rstrip('/')}/{symbol}?{query}"
            try:
                result = self.client.get(url, f"yahoo_{name}.json", headers=headers)
                last_request = time.monotonic()
                payload = json.loads(result.content)
                chart = payload.get("chart") or {}
                entries = chart.get("result") or []
                if not entries:
                    raise ValueError(f"Yahoo chart contains no result: {chart.get('error')}")
                entry = entries[0]
                timestamps = entry.get("timestamp") or []
                indicators = entry.get("indicators") or {}
                adjusted = indicators.get("adjclose") or []
                adjusted_values = adjusted[0].get("adjclose", []) if adjusted else []
                quote = indicators.get("quote") or []
                close_values = quote[0].get("close", []) if quote else []
                values = adjusted_values if len(adjusted_values) == len(timestamps) else close_values
                if not timestamps or len(values) != len(timestamps):
                    raise ValueError("Yahoo chart timestamps and closes are inconsistent")
                series = pd.Series(
                    pd.to_numeric(values, errors="coerce"),
                    index=pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None),
                    name=name,
                )
                frame = self._clean_index(series.to_frame()).loc[self.start_date :]
                if frame.empty:
                    raise ValueError("Yahoo chart contains no usable rows")
                frames.append(frame[name])
                self._record(name, "Yahoo Finance Chart", frame, result.status)
            except Exception as exc:
                if "429" in str(exc):
                    time.sleep(float(section.get("cooldown_seconds", 120)))
                self._record(name, "Yahoo Finance Chart", None, "failed", str(exc))
        return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

    def load(self) -> PublicDataBundle:
        data = self.config["data"]
        macro = self._fred() if data.get("fred", {}).get("enabled", True) else pd.DataFrame()
        vix = self._vix() if data.get("cboe", {}).get("enabled", True) else pd.DataFrame()
        if not vix.empty and data.get("cboe", {}).get("indices"):
            extra = self._cboe_indices()
            if not extra.empty:
                vix = vix.join(extra, how="outer").sort_index()
        ofr = self._ofr() if data.get("ofr", {}).get("enabled", True) else pd.DataFrame()
        french = self._french() if data.get("french", {}).get("enabled", True) else pd.DataFrame()
        symbols = data.get("nasdaq", {}).get("symbols", {})
        etf_close = (
            self._stooq(symbols)
            if data.get("stooq", {}).get("enabled", True)
            else pd.DataFrame()
        )
        missing = {name: symbol for name, symbol in symbols.items() if name not in etf_close}
        if missing and data.get("yahoo", {}).get("enabled", True):
            fallback = self._yahoo(missing)
            etf_close = etf_close.combine_first(fallback)
            missing = {name: symbol for name, symbol in symbols.items() if name not in etf_close}
        if missing and data.get("nasdaq", {}).get("enabled", True):
            fallback = self._nasdaq(missing)
            etf_close = etf_close.combine_first(fallback)
        manifest = pd.DataFrame(self._manifest)
        macro_point_in_time = str(data.get("fred", {}).get("vintage_mode", "latest")) == "point_in_time"
        if macro_point_in_time and french.empty:
            point_in_time_status = "point_in_time"
        elif macro_point_in_time:
            point_in_time_status = "point_in_time_macro_latest_vintage_french"
        else:
            point_in_time_status = "latest_vintage_macro_and_french"
        return PublicDataBundle(
            macro, vix, ofr, french, etf_close, manifest, point_in_time_status
        )
