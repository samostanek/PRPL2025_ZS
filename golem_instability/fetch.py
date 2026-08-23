"""HTTP access to the GOLEM shot database with an on-disk cache.

The cache mirrors the server tree one-to-one::

    <cache_dir>/<shot>/Diagnostics/BasicDiagnostics/Results/Ip.csv

so a cache populated by this module can also be inspected by hand, and the
whole package can be run in ``offline=True`` mode against a previously
downloaded (or synthesised) tree without touching the network.

Missing resources are a normal outcome, not an error: a diagnostic that did
not run simply 404s. Those are remembered in a negative cache so a shot scan
does not re-request them.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .config import SHOTDIR_TEMPLATE

logger = logging.getLogger(__name__)

#: Marker written next to a cache entry that is known to be absent upstream.
MISSING_SUFFIX = ".missing"


class ShotNotFound(Exception):
    """Raised when a shot number does not exist on the server."""


class OfflineError(Exception):
    """Raised when offline mode is requested but the cache lacks the file."""


@dataclass
class FetcherConfig:
    """Tunables for :class:`GolemFetcher`.

    Attributes:
        cache_dir: Root of the local mirror.
        offline: Serve exclusively from the cache; never open a socket.
        timeout: Per-request timeout in seconds.
        max_retries: Attempts per URL before giving up (404 is not retried).
        retry_delay: Base delay in seconds; doubled after each failed attempt.
        refresh: Ignore cached copies and re-download.
        strict_offline: In offline mode, raise :class:`OfflineError` on a cache
            miss instead of reporting the file as absent. Off by default, so an
            offline tree behaves like a shot whose diagnostics did not all run.
    """

    cache_dir: Path = Path("golem_cache")
    offline: bool = False
    strict_offline: bool = False
    timeout: float = 20.0
    max_retries: int = 3
    retry_delay: float = 2.0
    refresh: bool = False


class GolemFetcher:
    """Fetches per-shot files, caching every byte that comes back."""

    def __init__(self, config: Optional[FetcherConfig] = None) -> None:
        self.config = config or FetcherConfig()
        self.config.cache_dir = Path(self.config.cache_dir)
        self._local = threading.local()
        self.n_requests = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ paths
    #: Directory listings are cached under this name inside the directory, so
    #: that a listing and the directory it describes can coexist on disk.
    DIR_INDEX_NAME = "_dirindex.html"

    def cache_path(self, shot: int, rel_path: str) -> Path:
        if rel_path in ("", "/") or rel_path.endswith("/"):
            rel_path = rel_path + self.DIR_INDEX_NAME
        return self.config.cache_dir / str(shot) / rel_path

    @staticmethod
    def url(shot: int, rel_path: str) -> str:
        return SHOTDIR_TEMPLATE.format(shot=shot) + rel_path.lstrip("/")

    # ------------------------------------------------------------------ core
    def get_bytes(self, shot: int, rel_path: str) -> Optional[bytes]:
        """Return the raw file content, or ``None`` if it does not exist.

        Args:
            shot: Shot number (0 resolves to the latest shot server-side, but
                prefer :meth:`latest_shot` so the cache keys stay concrete).
            rel_path: Path relative to ``shotdir/<shot>/``.
        """
        path = self.cache_path(shot, rel_path)
        missing_marker = path.with_name(path.name + MISSING_SUFFIX)

        if not self.config.refresh:
            if path.exists():
                self.n_cache_hits += 1
                return path.read_bytes()
            if missing_marker.exists():
                self.n_cache_hits += 1
                return None

        if self.config.offline:
            if self.config.strict_offline:
                raise OfflineError(
                    f"offline mode: {self.url(shot, rel_path)} is not in {self.config.cache_dir}"
                )
            logger.debug("offline: %s not cached, treating as absent", rel_path)
            return None

        content = self._download(self.url(shot, rel_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            missing_marker.write_bytes(b"")
        else:
            path.write_bytes(content)
            if missing_marker.exists():
                missing_marker.unlink()
        return content

    def get_text(self, shot: int, rel_path: str) -> Optional[str]:
        """Same as :meth:`get_bytes` but decoded, ``None`` when absent."""
        raw = self.get_bytes(shot, rel_path)
        return None if raw is None else raw.decode("utf-8", errors="replace")

    def get_float(self, shot: int, rel_path: str) -> Optional[float]:
        """Read a single-number text file (``Ip_max``, ``b_plasma``, ...)."""
        text = self.get_text(shot, rel_path)
        if text is None:
            return None
        try:
            return float(text.strip().split()[0])
        except (ValueError, IndexError):
            logger.debug("could not parse %r from %s", text[:40], rel_path)
            return None

    @property
    def _session(self) -> requests.Session:
        """One :class:`requests.Session` per thread (they are not thread-safe)."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def _download(self, url: str) -> Optional[bytes]:
        delay = self.config.retry_delay
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.n_requests += 1
                response = self._session.get(url, timeout=self.config.timeout)
                if response.status_code == 404:
                    logger.debug("404 %s", url)
                    return None
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.config.max_retries:
                    logger.warning("%s (attempt %d/%d), retrying in %.0fs",
                                   exc, attempt, self.config.max_retries, delay)
                    time.sleep(delay)
                    delay *= 2
        raise ConnectionError(f"failed to fetch {url}: {last_error}") from last_error

    # --------------------------------------------------------------- helpers
    def latest_shot(self) -> int:
        """Resolve the ``shot=0`` alias to a concrete shot number."""
        if self.config.offline:
            raise OfflineError("latest_shot() needs network access")
        text = self._download(SHOTDIR_TEMPLATE.format(shot=0) + "shot_no")
        if text is None:
            raise ShotNotFound("shotdir/0/shot_no returned 404")
        return int(text.decode().strip())

    def exists(self, shot: int) -> bool:
        """True when the shot directory exists upstream (or in the cache)."""
        return self.get_text(shot, "shot_no") is not None
