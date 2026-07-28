"""Discovery of DICOM series and NIfTI volumes inside uploaded archives."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import pydicom

from ct_pathology.medical_io.ct_preprocessor import (
    DiscoveredInput,
    discover_inputs_robust,
)

from .archive_utils import safe_extract_zip
from .data_models import DataType, StudyInfo


class DataDiscoveryService:
    """Extracts an archive and normalizes each supported medical study."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def discover_studies_in_zip(
        self,
        zip_path: str,
        extract_dir: str,
    ) -> list[StudyInfo]:
        """Returns unique, UID-valid studies discovered in one ZIP archive."""

        self.logger.info("Discovering studies in %s", zip_path)
        safe_extract_zip(zip_path, extract_dir)
        raw_inputs = discover_inputs_robust(Path(extract_dir), self.logger)

        studies: list[StudyInfo] = []
        seen_series: set[tuple[str, str]] = set()
        for raw_input in raw_inputs:
            study = self._convert_raw_input_to_study_info(raw_input)
            if study is None:
                continue

            series_key = (study.study_uid, study.series_uid)
            if series_key in seen_series:
                self.logger.warning("Skipping duplicate medical series %s", series_key)
                continue
            seen_series.add(series_key)
            studies.append(study)

        self.logger.info("Found %d unique studies in %s", len(studies), zip_path)
        return studies

    def _convert_raw_input_to_study_info(
        self,
        raw_input: DiscoveredInput,
    ) -> StudyInfo | None:
        input_type = raw_input["type"]
        if input_type == "dicom_dir":
            return self._process_dicom_input(raw_input)
        if input_type == "nifti":
            return self._process_nifti_input(raw_input)

        self.logger.warning("Unsupported discovered input type: %s", input_type)
        return None

    def _process_dicom_input(self, raw_input: DiscoveredInput) -> StudyInfo | None:
        """Validates mandatory DICOM UIDs without retaining patient identifiers."""

        dicom_dir = Path(raw_input["path"])
        file_list = raw_input.get("file_list", [])
        if not file_list:
            self.logger.error("DICOM series has no readable files: %s", dicom_dir)
            return None

        try:
            first_file = Path(file_list[0])
            dataset = pydicom.dcmread(
                str(first_file),
                stop_before_pixels=True,
                force=False,
            )
            study_uid = str(getattr(dataset, "StudyInstanceUID", "")).strip()
            series_uid = str(getattr(dataset, "SeriesInstanceUID", "")).strip()
        except Exception as exc:
            self.logger.warning("Failed to read DICOM metadata from %s: %s", first_file, exc)
            return None

        if not study_uid or not series_uid:
            self.logger.error("Mandatory DICOM study or series UID is missing in %s", first_file)
            return None
        if study_uid != raw_input["study_uid"] or series_uid != raw_input["series_uid"]:
            self.logger.error("DICOM discovery metadata changed during validation: %s", first_file)
            return None

        metadata = {
            "input_type": "dicom_dir",
            "file_list": file_list,
            "modality": str(getattr(dataset, "Modality", "")),
        }
        try:
            return StudyInfo(
                path_to_study=str(dicom_dir),
                study_uid=study_uid,
                series_uid=series_uid,
                data_type=DataType.DICOM,
                files_count=len(file_list),
                file_size_mb=raw_input["file_size_mb"],
                metadata=metadata,
            )
        except ValueError as exc:
            self.logger.error("Invalid DICOM study metadata: %s", exc)
            return None

    def _process_nifti_input(self, raw_input: DiscoveredInput) -> StudyInfo | None:
        """Reads non-identifying NIfTI geometry metadata."""

        nifti_path = Path(raw_input["path"])
        metadata: dict[str, Any] = {"input_type": "nifti"}
        try:
            image = nib.load(str(nifti_path))
            metadata.update(
                {
                    "shape": tuple(int(value) for value in image.shape),
                    "dtype": str(image.get_data_dtype()),
                    "zooms": tuple(float(value) for value in image.header.get_zooms()[:3]),
                }
            )
        except Exception as exc:
            self.logger.warning("Failed to read NIfTI metadata from %s: %s", nifti_path, exc)

        return StudyInfo(
            path_to_study=str(nifti_path),
            study_uid=raw_input["study_uid"],
            series_uid=raw_input["series_uid"],
            data_type=DataType.NIFTI,
            files_count=1,
            file_size_mb=raw_input["file_size_mb"],
            metadata=metadata,
        )
