"""Cloud-agnostic storage abstraction supporting Local filesystem and Google Cloud Storage (gs://)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


class StorageBackend(Protocol):
    def read_bytes(self, uri: str) -> bytes: ...

    def write_bytes(self, uri: str, data: bytes) -> None: ...

    def exists(self, uri: str) -> bool: ...

    def resolve_local_file(self, uri: str, cache_dir: Path) -> Path: ...


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse ``gs://bucket/path/to/blob`` into ``(bucket, blob_name)``."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    parsed = urlparse(uri)
    bucket = parsed.netloc.strip()
    blob = parsed.path.lstrip("/")
    if not bucket or not blob:
        raise ValueError(f"Malformed GCS URI (requires bucket and object path): {uri}")
    return bucket, blob


def _local_path_from_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(url2pathname(unquote(parsed.path)))
    return Path(uri)


class LocalStorage:
    """Storage backend for local paths and file:// URIs."""

    def read_bytes(self, uri: str) -> bytes:
        return _local_path_from_uri(uri).read_bytes()

    def write_bytes(self, uri: str, data: bytes) -> None:
        path = _local_path_from_uri(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, uri: str) -> bool:
        return _local_path_from_uri(uri).exists()

    def resolve_local_file(self, uri: str, cache_dir: Path) -> Path:
        path = _local_path_from_uri(uri).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local dataset not found: {path}")
        return path


class GCSStorage:
    """Google Cloud Storage backend using lazy client creation or an injected client."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage
            except ImportError as exc:
                raise ImportError(
                    "google-cloud-storage is required for gs:// URIs. "
                    "Install with: pip install google-cloud-storage"
                ) from exc
            self._client = storage.Client()
        return self._client

    def _blob(self, uri: str) -> Any:
        bucket_name, blob_name = parse_gcs_uri(uri)
        return self.client.bucket(bucket_name).blob(blob_name)

    def read_bytes(self, uri: str) -> bytes:
        return bytes(self._blob(uri).download_as_bytes())

    def write_bytes(self, uri: str, data: bytes) -> None:
        self._blob(uri).upload_from_string(data)

    def exists(self, uri: str) -> bool:
        return bool(self._blob(uri).exists())

    def resolve_local_file(self, uri: str, cache_dir: Path) -> Path:
        _, blob_name = parse_gcs_uri(uri)
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / Path(blob_name).name
        local_path.write_bytes(self.read_bytes(uri))
        return local_path


def get_storage(uri: str, gcs_client: Any | None = None) -> StorageBackend:
    """Factory returning the appropriate storage backend for a URI."""
    if uri.startswith("gs://"):
        return GCSStorage(client=gcs_client)
    return LocalStorage()
