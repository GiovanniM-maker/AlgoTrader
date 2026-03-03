"""
src/ml/features/feature_store.py

Feature Store
=============
A lightweight, Parquet-based versioning system for ML feature sets.

Directory layout
----------------
  {base_dir}/
    {symbol}/
      {timeframe}/
        20240101_120000.parquet   ← version
        20240115_083015.parquet
        latest -> 20240115_083015.parquet  (symlink, created on save)

Version naming
--------------
Versions are UTC timestamps in ``YYYYMMDD_HHMMSS`` format.  The store sorts
them lexicographically (which equals chronological order given the format).

Parquet format
--------------
Files are written with PyArrow + Snappy compression for a good balance of
read speed, write speed, and file size.  The engine can be overridden if
needed.

Thread safety
-------------
Each ``save_features`` call is an atomic write: the data is written to a
temporary file in the same directory and then renamed over the destination.
Reads and lists are stateless and safe to call concurrently.

Usage
-----
    from src.ml.features.feature_store import FeatureStore

    store = FeatureStore()

    # Save a feature DataFrame
    version = store.save_features(df, symbol="BTCUSDT", timeframe="60")

    # Load the latest version
    df_loaded = store.load_features("BTCUSDT", "60")

    # Load a specific version
    df_v = store.load_features("BTCUSDT", "60", version="20240101_120000")

    # List all versions
    versions = store.list_versions("BTCUSDT", "60")

    # Get the latest version string
    latest = store.get_latest_version("BTCUSDT", "60")
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Parquet engine and compression settings
_ENGINE: str = "pyarrow"
_COMPRESSION: str = "snappy"

# File extension for feature files
_EXT: str = ".parquet"


# ---------------------------------------------------------------------------
# FeatureStore
# ---------------------------------------------------------------------------


class FeatureStore:
    """
    Parquet-based feature versioning store.

    Parameters
    ----------
    base_dir : str or Path
        Root directory for all feature files.
        Defaults to ``./data/cache/features`` (relative to the current
        working directory when instantiated).
    """

    def __init__(self, base_dir: str = "./data/cache/features") -> None:
        self.base_dir: Path = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("FeatureStore initialised at: %s", self.base_dir)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        version: Optional[str] = None,
    ) -> str:
        """
        Persist *df* as a Parquet file in the feature store.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame to save.  Any index and column types are
            preserved; the store does not impose a schema.
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        timeframe : str
            Timeframe label, e.g. ``"60"`` (Bybit interval) or ``"1h"``.
        version : str, optional
            Version string (must be unique per symbol/timeframe).  Defaults
            to a UTC timestamp: ``YYYYMMDD_HHMMSS``.

        Returns
        -------
        str
            The version string used (caller needs this to retrieve the file later).

        Raises
        ------
        ValueError
            If *df* is empty.
        OSError
            If the file cannot be written.
        """
        if df is None or df.empty:
            raise ValueError(
                "Cannot save an empty DataFrame to the FeatureStore."
            )

        # Resolve version
        if not version:
            version = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Validate version string: no path separators or null bytes
        if any(c in version for c in ("/", "\\", "\0")):
            raise ValueError(
                f"version string must not contain path separators, got: {version!r}"
            )

        dest_dir = self._symbol_tf_dir(symbol, timeframe)
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / f"{version}{_EXT}"

        # Atomic write: write to a temp file in the same directory, then rename
        # so readers never see a partial file.
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=_EXT, prefix=f".tmp_{version}_", dir=dest_dir
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            df.to_parquet(
                tmp_path,
                engine=_ENGINE,
                compression=_COMPRESSION,
                index=True,   # preserve index (usually DatetimeIndex)
            )
            # Atomic rename
            tmp_path.replace(dest_path)
        except Exception:
            # Clean up the temp file on failure
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        self.logger.info(
            "FeatureStore: saved %s/%s version=%s path=%s shape=%s",
            symbol, timeframe, version, dest_path, df.shape,
        )

        return version

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_features(
        self,
        symbol: str,
        timeframe: str,
        version: str = "latest",
    ) -> pd.DataFrame:
        """
        Load a feature DataFrame from the store.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. ``"BTCUSDT"``.
        timeframe : str
            Timeframe label, e.g. ``"60"``.
        version : str
            Version to load.  Pass ``"latest"`` (default) to load the most
            recently saved version.

        Returns
        -------
        pd.DataFrame
            The stored feature DataFrame.

        Raises
        ------
        FileNotFoundError
            If the requested version does not exist, or if no versions are
            available when ``version="latest"`` is requested.
        """
        # Resolve "latest" alias
        if version == "latest":
            resolved = self.get_latest_version(symbol, timeframe)
            if resolved is None:
                raise FileNotFoundError(
                    f"No feature versions found for {symbol}/{timeframe} in "
                    f"{self._symbol_tf_dir(symbol, timeframe)}"
                )
            version = resolved

        path = self._version_path(symbol, timeframe, version)

        if not path.exists():
            raise FileNotFoundError(
                f"Feature version not found: {path}\n"
                f"Available versions: {self.list_versions(symbol, timeframe)}"
            )

        self.logger.info(
            "FeatureStore: loading %s/%s version=%s from %s",
            symbol, timeframe, version, path,
        )

        df = pd.read_parquet(path, engine=_ENGINE)

        self.logger.debug(
            "FeatureStore: loaded shape=%s dtypes=%s",
            df.shape, df.dtypes.to_dict(),
        )
        return df

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def list_versions(self, symbol: str, timeframe: str) -> List[str]:
        """
        Return a sorted list of version strings for *symbol*/*timeframe*.

        Versions are sorted lexicographically, which equals chronological
        order given the ``YYYYMMDD_HHMMSS`` naming convention.

        Returns an empty list if the directory does not exist or contains
        no Parquet files.
        """
        dir_path = self._symbol_tf_dir(symbol, timeframe)

        if not dir_path.exists():
            return []

        versions: List[str] = []
        for entry in dir_path.iterdir():
            if entry.is_file() and entry.suffix == _EXT and not entry.name.startswith(".tmp_"):
                version_str = entry.stem  # filename without extension
                versions.append(version_str)

        versions.sort()
        return versions

    def get_latest_version(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Return the most recent version string for *symbol*/*timeframe*.

        Returns
        -------
        str or None
            The latest version string, or ``None`` if no versions exist.
        """
        versions = self.list_versions(symbol, timeframe)
        if not versions:
            return None
        return versions[-1]

    # ------------------------------------------------------------------
    # Deletion / maintenance
    # ------------------------------------------------------------------

    def delete_version(self, symbol: str, timeframe: str, version: str) -> bool:
        """
        Delete a specific version from the store.

        Parameters
        ----------
        symbol : str
        timeframe : str
        version : str
            The version string to delete (not ``"latest"``).

        Returns
        -------
        bool
            ``True`` if the file was deleted, ``False`` if it did not exist.
        """
        if version == "latest":
            raise ValueError("Cannot delete the 'latest' alias directly; specify the version string.")

        path = self._version_path(symbol, timeframe, version)
        if path.exists():
            path.unlink()
            self.logger.info(
                "FeatureStore: deleted version %s for %s/%s", version, symbol, timeframe
            )
            return True

        self.logger.warning(
            "FeatureStore: version %s not found for %s/%s — nothing deleted.",
            version, symbol, timeframe,
        )
        return False

    def delete_old_versions(
        self,
        symbol: str,
        timeframe: str,
        keep: int = 5,
    ) -> List[str]:
        """
        Remove all but the most recent *keep* versions.

        Parameters
        ----------
        symbol : str
        timeframe : str
        keep : int
            Number of most-recent versions to retain.  Must be >= 1.

        Returns
        -------
        list of str
            Version strings that were deleted.
        """
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")

        versions = self.list_versions(symbol, timeframe)
        to_delete = versions[:-keep] if len(versions) > keep else []

        deleted: List[str] = []
        for v in to_delete:
            if self.delete_version(symbol, timeframe, v):
                deleted.append(v)

        if deleted:
            self.logger.info(
                "FeatureStore: pruned %d old version(s) for %s/%s: %s",
                len(deleted), symbol, timeframe, deleted,
            )
        return deleted

    # ------------------------------------------------------------------
    # Metadata / introspection
    # ------------------------------------------------------------------

    def get_version_info(self, symbol: str, timeframe: str, version: str) -> dict:
        """
        Return metadata about a stored version.

        Returns
        -------
        dict
            Keys: ``version``, ``symbol``, ``timeframe``, ``path``,
            ``size_bytes``, ``exists``, ``columns``, ``n_rows``.
        """
        if version == "latest":
            resolved = self.get_latest_version(symbol, timeframe)
            version = resolved or "NOT_FOUND"

        path = self._version_path(symbol, timeframe, version)
        info: dict = {
            "version": version,
            "symbol": symbol,
            "timeframe": timeframe,
            "path": str(path),
            "exists": path.exists(),
        }

        if path.exists():
            info["size_bytes"] = path.stat().st_size
            try:
                import pyarrow.parquet as pq  # type: ignore[import]
                pf = pq.ParquetFile(path)
                schema = pf.schema_arrow
                info["columns"] = schema.names
                info["n_rows"] = pf.metadata.num_rows
            except Exception as exc:
                self.logger.debug("Could not read Parquet metadata: %s", exc)
                info["columns"] = []
                info["n_rows"] = None
        else:
            info["size_bytes"] = 0
            info["columns"] = []
            info["n_rows"] = None

        return info

    def list_symbols(self) -> List[str]:
        """Return all symbols that have stored features."""
        if not self.base_dir.exists():
            return []
        return sorted(
            entry.name
            for entry in self.base_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def list_timeframes(self, symbol: str) -> List[str]:
        """Return all timeframes stored for *symbol*."""
        symbol_dir = self.base_dir / symbol
        if not symbol_dir.exists():
            return []
        return sorted(
            entry.name
            for entry in symbol_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symbol_tf_dir(self, symbol: str, timeframe: str) -> Path:
        """Return the directory path for a symbol/timeframe pair."""
        return self.base_dir / symbol / timeframe

    def _version_path(self, symbol: str, timeframe: str, version: str) -> Path:
        """Return the full file path for a specific version."""
        return self._symbol_tf_dir(symbol, timeframe) / f"{version}{_EXT}"

    def __repr__(self) -> str:
        symbols = self.list_symbols()
        return (
            f"FeatureStore(base_dir={self.base_dir!r}, "
            f"symbols={symbols})"
        )
