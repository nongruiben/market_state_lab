from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class FetchResult:
    content: bytes
    status: str
    cache_path: Path
    fetched_at: float


class CachedHttpClient:
    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: int = 30,
        retries: int = 3,
        max_age_hours: float = 24,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_age_seconds = max_age_hours * 3600
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _path_for(self, url: str, cache_name: str | None) -> Path:
        if cache_name:
            return self.cache_dir / cache_name
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.bin"

    def get(
        self,
        url: str,
        cache_name: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> FetchResult:
        cache_path = self._path_for(url, cache_name)
        now = time.time()
        if cache_path.exists() and now - cache_path.stat().st_mtime <= self.max_age_seconds:
            return FetchResult(cache_path.read_bytes(), "cache_fresh", cache_path, now)

        try:
            response = self.session.get(
                url,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(cache_path)
            return FetchResult(response.content, "network", cache_path, now)
        except requests.RequestException:
            if cache_path.exists():
                return FetchResult(cache_path.read_bytes(), "cache_stale", cache_path, now)
            raise
