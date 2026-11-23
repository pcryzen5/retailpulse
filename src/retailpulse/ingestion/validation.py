"""Upload file streaming, security validation, and disk storage service.

Implements TRD §4.3, API §3.9, and SECURITY §5:
- SEC-INP-03: Streaming size enforcement against RP_MAX_UPLOAD_MB.
- SEC-INP-04: Extension allowlist (.csv, .xlsx) and magic bytes content check (rejects .xlsm).
- SEC-INP-05: Defenses against macro-enabled and corrupt XLSX spreadsheets.
- SEC-INP-06: Storage under server-generated UUID in uploads directory with mode 0600.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import openpyxl
from fastapi import HTTPException, UploadFile, status

from retailpulse.config import Settings
from retailpulse.errors import DomainValidationError, IngestionError
from retailpulse.ingestion.adapters.generic_csv import _normalize_generic_columns
from retailpulse.ingestion.adapters.online_retail_ii import _normalize_online_retail_columns
from retailpulse.ingestion.schemas import (
    GENERIC_CSV_REQUIRED,
    ONLINE_RETAIL_II_REQUIRED,
)
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Allowed file extensions and upload source adapters
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx"})
ALLOWED_SOURCES: frozenset[str] = frozenset({"online_retail_ii", "generic_csv"})

# ZIP archive signature (used by XLSX documents)
ZIP_MAGIC: bytes = b"PK\x03\x04"

# Default buffer chunk size for streaming file I/O (64 KB)
STREAM_CHUNK_SIZE: int = 65_536


@dataclass(frozen=True)
class ValidatedUpload:
    """Metadata resulting from successful file streaming and header validation."""

    batch_id: uuid.UUID
    saved_path: Path
    file_sha256: str
    total_bytes: int
    original_filename: str
    source_name: str
    extension: str


def sanitize_filename(filename: str | None) -> str:
    """Sanitize client-provided filename per SEC-INP-06.

    Extracts basename only, removes control/unsafe characters, and caps at 255 chars.
    """
    if not filename:
        return "upload.csv"
    base = os.path.basename(filename.replace("\\", "/"))
    # Filter printable characters excluding filesystem delimiters
    cleaned = "".join(c for c in base if c.isprintable() and c not in '<>:"/\\|?*')
    cleaned = cleaned[:255].strip()
    return cleaned if cleaned else "upload.csv"


def _validate_headers_csv(file_path: Path, source_name: str) -> None:
    """Validate that CSV header row contains all required columns for the source adapter."""
    try:
        with file_path.open("r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header_row = next(reader, None)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unable to read CSV file content: {e}",
        ) from e

    if not header_row or all(not col.strip() for col in header_row):
        raise IngestionError("CSV file is empty or missing a valid header line")

    raw_headers = [col.strip() for col in header_row if col.strip()]

    if source_name == "online_retail_ii":
        normalized = _normalize_online_retail_columns(raw_headers)
        present_set = set(normalized.values())
        missing = ONLINE_RETAIL_II_REQUIRED - present_set
        if missing:
            raise IngestionError(
                f"Missing required column(s) for Online Retail II: {sorted(missing)}. "
                f"Present columns: {sorted(present_set)}",
                details={
                    "missing_columns": sorted(missing),
                    "present_columns": sorted(present_set),
                    "source": source_name,
                },
            )
    elif source_name == "generic_csv":
        normalized = _normalize_generic_columns(raw_headers)
        present_set = set(normalized.values())
        missing = GENERIC_CSV_REQUIRED - present_set
        if missing:
            raise IngestionError(
                f"Missing required column(s) for Generic CSV: {sorted(missing)}. "
                f"Present columns: {sorted(present_set)}",
                details={
                    "missing_columns": sorted(missing),
                    "present_columns": sorted(present_set),
                    "source": source_name,
                },
            )


def _validate_headers_xlsx(file_path: Path, source_name: str) -> None:
    """Validate XLSX workbook headers using read-only openpyxl per SEC-INP-05."""
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Invalid or corrupted XLSX spreadsheet file",
        ) from e

    try:
        sheet_names = wb.sheetnames
        if not sheet_names:
            raise IngestionError("XLSX workbook contains no sheets")

        first_sheet = wb[sheet_names[0]]
        header_row = next(first_sheet.iter_rows(max_row=1, values_only=True), None)
        if not header_row or all(col is None for col in header_row):
            raise IngestionError("XLSX sheet is empty or missing a valid header row")

        raw_headers = [
            str(col).strip() for col in header_row if col is not None and str(col).strip()
        ]

        if source_name == "online_retail_ii":
            normalized = _normalize_online_retail_columns(raw_headers)
            present_set = set(normalized.values())
            missing = ONLINE_RETAIL_II_REQUIRED - present_set
            if missing:
                raise IngestionError(
                    f"Missing required column(s) for Online Retail II: {sorted(missing)}. "
                    f"Present columns: {sorted(present_set)}",
                    details={
                        "missing_columns": sorted(missing),
                        "present_columns": sorted(present_set),
                        "source": source_name,
                    },
                )
        elif source_name == "generic_csv":
            normalized = _normalize_generic_columns(raw_headers)
            present_set = set(normalized.values())
            missing = GENERIC_CSV_REQUIRED - present_set
            if missing:
                raise IngestionError(
                    f"Missing required column(s) for Generic CSV: {sorted(missing)}. "
                    f"Present columns: {sorted(present_set)}",
                    details={
                        "missing_columns": sorted(missing),
                        "present_columns": sorted(present_set),
                        "source": source_name,
                    },
                )
    finally:
        wb.close()


def _validate_magic_bytes(ext: str, first_chunk: bytes) -> None:
    """Validate magic bytes and text decoding per SEC-INP-04."""
    if ext == ".xlsx":
        if not first_chunk.startswith(ZIP_MAGIC):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid XLSX file: missing ZIP file signature.",
            )
    elif ext == ".csv":
        if first_chunk.startswith(ZIP_MAGIC):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid CSV file: archive or binary file disguised as CSV.",
            )
        if b"\x00" in first_chunk[:1024]:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid CSV file: null bytes detected; binary content is not allowed.",
            )
        try:
            first_chunk[:4096].decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid CSV file: content is not valid UTF-8 text.",
            ) from e


async def stream_and_validate_upload(
    upload_file: UploadFile,
    source_name: str,
    settings: Settings,
) -> tuple[str, Path, int, str]:
    """Stream incoming upload, calculate sha256, verify magic bytes, and save to temp file.

    Args:
        upload_file: FastAPI multipart UploadFile.
        source_name: Target adapter identifier.
        settings: Application settings.

    Returns:
        Tuple of (sha256_hex, temp_file_path, total_bytes, extension).

    Raises:
        HTTPException: On 413 (Payload Too Large), 415 (Unsupported Media Type).
        DomainValidationError: On 422 (Invalid source name).
    """
    clean_source = source_name.strip().lower()
    if clean_source not in ALLOWED_SOURCES:
        raise DomainValidationError(
            f"Invalid source_name '{source_name}'. Allowed sources: {sorted(ALLOWED_SOURCES)}"
        )

    client_filename = upload_file.filename or "upload.csv"
    ext = Path(client_filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS or ext == ".xlsm":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported media type. Only .csv and .xlsx files are supported "
                "(macro-enabled .xlsm rejected)."
            ),
        )

    uploads_dir = Path(settings.data_dir) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    hasher = hashlib.sha256()
    total_bytes = 0
    first_chunk: bytes | None = None

    # Create temporary file for streaming upload chunks
    temp_fd, temp_path_str = tempfile.mkstemp(dir=str(uploads_dir), suffix=ext)
    temp_path = Path(temp_path_str)

    try:
        with open(temp_fd, "wb") as temp_out:
            while True:
                chunk = await upload_file.read(STREAM_CHUNK_SIZE)
                if not chunk:
                    break

                if first_chunk is None:
                    first_chunk = chunk

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"File exceeds maximum allowed upload size of "
                            f"{settings.max_upload_mb} MB."
                        ),
                    )

                hasher.update(chunk)
                temp_out.write(chunk)

        if total_bytes == 0 or first_chunk is None:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Uploaded file is empty.",
            )

        # Magic bytes content validation (SEC-INP-04)
        _validate_magic_bytes(ext, first_chunk)

        return hasher.hexdigest(), temp_path, total_bytes, ext

    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def finalize_upload_storage(  # noqa: PLR0917
    temp_path: Path,
    batch_id: uuid.UUID,
    ext: str,
    source_name: str,
    original_filename: str,
    settings: Settings,
    total_bytes: int,
    file_sha256: str,
) -> ValidatedUpload:
    """Validate headers and move temporary upload file to canonical destination with mode 0600."""
    try:
        # Validate headers before making the file permanent
        if ext == ".csv":
            _validate_headers_csv(temp_path, source_name)
        elif ext == ".xlsx":
            _validate_headers_xlsx(temp_path, source_name)

        uploads_dir = Path(settings.data_dir) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        final_path = uploads_dir / f"{batch_id}{ext}"

        shutil.move(str(temp_path), str(final_path))

        # Restrict permissions to owner read/write (mode 0600 per SEC-INP-06)
        try:
            final_path.chmod(0o600)
        except OSError:
            logger.debug("chmod 0600 not fully supported on this platform", path=str(final_path))

        sanitized_name = sanitize_filename(original_filename)

        return ValidatedUpload(
            batch_id=batch_id,
            saved_path=final_path,
            file_sha256=file_sha256,
            total_bytes=total_bytes,
            original_filename=sanitized_name,
            source_name=source_name,
            extension=ext,
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
