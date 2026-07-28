"""Typed discovery and loading helpers for DICOM and NIfTI studies."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
MAX_DICOM_SLICES = 1_000


class DiscoveredInput(TypedDict):
    """Normalized record produced by medical-data discovery."""

    type: str
    path: str
    file_size_mb: float
    files_count: int
    series_uid: str
    study_uid: str
    file_list: NotRequired[list[str]]


def _read_dicom_uids(path: Path) -> tuple[str, str] | None:
    """Reads the mandatory study and series identifiers without pixel data."""

    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception:
        return None

    study_uid = str(getattr(dataset, "StudyInstanceUID", "")).strip()
    series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
    if not study_uid or not series_uid:
        return None
    return study_uid, series_uid


def get_all_series_in_dicom_dir(dicom_dir: Path) -> list[tuple[str, str, list[Path]]]:
    """Groups readable DICOM files by study and series UID."""

    grouped: defaultdict[tuple[str, str], list[Path]] = defaultdict(list)
    for candidate in dicom_dir.rglob("*"):
        if not candidate.is_file():
            continue
        lower_name = candidate.name.lower()
        if lower_name.endswith((".nii", ".nii.gz")):
            continue

        uids = _read_dicom_uids(candidate)
        if uids is not None:
            grouped[uids].append(candidate)

    return [
        (study_uid, series_uid, sorted(files))
        for (study_uid, series_uid), files in sorted(grouped.items())
    ]


def discover_inputs_robust(
    input_dir: Path,
    logger: logging.Logger | None = None,
) -> list[DiscoveredInput]:
    """Discovers NIfTI volumes and UID-valid DICOM series recursively."""

    logger = logger or LOGGER
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Medical input directory not found: {input_dir}")

    discovered: list[DiscoveredInput] = []
    nifti_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    )
    for nifti_path in nifti_files:
        size_mb = nifti_path.stat().st_size / (1024**2)
        stem = nifti_path.name.removesuffix(".gz").removesuffix(".nii")
        discovered.append(
            {
                "type": "nifti",
                "path": str(nifti_path),
                "file_size_mb": size_mb,
                "files_count": 1,
                "series_uid": f"nifti_series_{stem}",
                "study_uid": f"nifti_study_{stem}",
            }
        )

    for study_uid, series_uid, files in get_all_series_in_dicom_dir(input_dir):
        total_size_mb = sum(path.stat().st_size for path in files) / (1024**2)
        discovered.append(
            {
                "type": "dicom_dir",
                "path": str(input_dir),
                "file_size_mb": total_size_mb,
                "files_count": len(files),
                "series_uid": series_uid,
                "study_uid": study_uid,
                "file_list": [str(path) for path in files],
            }
        )

    logger.info(
        "Discovered %d NIfTI volumes and %d DICOM series",
        len(nifti_files),
        len(discovered) - len(nifti_files),
    )
    return discovered


def robust_load_dicom_volume(
    dicom_dir: Path,
    file_list: list[Path] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Loads one DICOM series and preserves geometry and HU metadata."""

    logger = logger or LOGGER
    dicom_dir = dicom_dir.resolve()

    if file_list is None:
        series = get_all_series_in_dicom_dir(dicom_dir)
        if not series:
            raise ValueError(f"No UID-valid DICOM series found in {dicom_dir}")
        _, series_uid, file_list = max(series, key=lambda item: len(item[2]))
        logger.info("Selected largest DICOM series %s", series_uid)
    else:
        file_list = [
            path.resolve() if path.is_absolute() else (dicom_dir / path).resolve()
            for path in file_list
        ]

    if not file_list:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    if len(file_list) > MAX_DICOM_SLICES:
        raise ValueError(f"Series contains {len(file_list)} slices; limit is {MAX_DICOM_SLICES}")

    first_dataset = pydicom.dcmread(
        str(file_list[0]),
        stop_before_pixels=True,
        force=False,
    )
    rescale_slope = float(getattr(first_dataset, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(first_dataset, "RescaleIntercept", 0.0))
    series_uid = str(getattr(first_dataset, "SeriesInstanceUID", "")).strip()

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in file_list])
    image = reader.Execute()
    volume = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)

    metadata: dict[str, Any] = {
        "spacing": tuple(float(value) for value in image.GetSpacing()),
        "origin": tuple(float(value) for value in image.GetOrigin()),
        "direction": tuple(float(value) for value in image.GetDirection()),
        "series_uid": series_uid,
        "num_slices": len(file_list),
        "RescaleSlope": rescale_slope,
        "RescaleIntercept": rescale_intercept,
    }
    logger.info("Loaded DICOM volume with shape %s", volume.shape)
    return volume, metadata
