import unittest

from ct_pathology.pipeline.data_models import (
    DataType,
    PipelineConfig,
    ProcessingResult,
    StudyInfo,
)


class StudyInfoTests(unittest.TestCase):
    def test_dicom_requires_study_and_series_uids(self) -> None:
        with self.assertRaises(ValueError):
            StudyInfo(
                path_to_study="/tmp/study",
                study_uid="",
                series_uid="series",
                data_type=DataType.DICOM,
                files_count=1,
                file_size_mb=1.0,
                metadata={},
            )

    def test_nifti_generates_stable_synthetic_uids(self) -> None:
        study = StudyInfo(
            path_to_study="/tmp/volume.nii.gz",
            study_uid="",
            series_uid="",
            data_type=DataType.NIFTI,
            files_count=1,
            file_size_mb=1.0,
            metadata={},
        )

        self.assertEqual(study.study_uid, "nifti_study_volume")
        self.assertEqual(study.series_uid, "nifti_series_volume")


class PipelineConfigTests(unittest.TestCase):
    def test_rejects_invalid_worker_count(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig("ct.pt", "classifier.cbm", max_workers=0)

    def test_rejects_unknown_device(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig("ct.pt", "classifier.cbm", device="tpu")


class ProcessingResultTests(unittest.TestCase):
    def test_steps_are_not_shared_between_instances(self) -> None:
        first = ProcessingResult("a", "b", "c", 0.1, 0, "Success", 1.0)
        second = ProcessingResult("a", "b", "c", 0.1, 0, "Success", 1.0)

        first.processing_steps_completed.append("volume_loading")

        self.assertEqual(second.processing_steps_completed, [])


if __name__ == "__main__":
    unittest.main()
