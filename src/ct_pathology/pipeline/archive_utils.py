"""Безопасная распаковка входных медицинских ZIP-архивов."""

from __future__ import annotations

import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path


class ArchiveValidationError(ValueError):
    """Архив не прошёл проверку безопасности или ресурсных лимитов."""


@dataclass(frozen=True)
class ArchiveLimits:
    """Ограничения на распакованный медицинский архив."""

    max_files: int = 100_000
    max_uncompressed_bytes: int = 80 * 1024**3
    max_compression_ratio: float = 200.0

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be positive")
        if self.max_uncompressed_bytes < 1:
            raise ValueError("max_uncompressed_bytes must be positive")
        if self.max_compression_ratio <= 0:
            raise ValueError("max_compression_ratio must be positive")


def _validated_target(root: Path, member: zipfile.ZipInfo) -> Path:
    """Возвращает безопасный путь назначения внутри ``root``."""

    if member.flag_bits & 0x1:
        raise ArchiveValidationError("Encrypted ZIP entries are not supported")

    unix_mode = member.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise ArchiveValidationError(f"ZIP symlink is not allowed: {member.filename}")

    root_resolved = root.resolve()
    target = (root / member.filename).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ArchiveValidationError(
            f"ZIP entry escapes extraction directory: {member.filename}"
        ) from exc

    return target


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    limits: ArchiveLimits | None = None,
) -> list[Path]:
    """Проверяет и распаковывает ZIP без path traversal и zip-bomb сценариев."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    limits = limits or ArchiveLimits()

    if not archive_path.is_file():
        raise FileNotFoundError(f"ZIP archive not found: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ArchiveValidationError(f"Invalid ZIP archive: {archive_path.name}")

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        file_members = [member for member in members if not member.is_dir()]

        if len(file_members) > limits.max_files:
            raise ArchiveValidationError(
                f"ZIP contains {len(file_members)} files; limit is {limits.max_files}"
            )

        total_size = sum(member.file_size for member in file_members)
        if total_size > limits.max_uncompressed_bytes:
            raise ArchiveValidationError("ZIP uncompressed size exceeds configured limit")

        targets = {member: _validated_target(destination, member) for member in members}
        for member in file_members:
            compressed_size = max(member.compress_size, 1)
            ratio = member.file_size / compressed_size
            if ratio > limits.max_compression_ratio:
                raise ArchiveValidationError(f"Suspicious compression ratio for {member.filename}")

        for member in members:
            target = targets[member]
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)

    return extracted
