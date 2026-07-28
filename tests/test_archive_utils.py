import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from ct_pathology.pipeline.archive_utils import (
    ArchiveLimits,
    ArchiveValidationError,
    safe_extract_zip,
)


class SafeExtractZipTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_extracts_regular_files(self) -> None:
        archive = self.root / "study.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("study/series/slice.dcm", b"dicom")

        extracted = safe_extract_zip(archive, self.root / "output")

        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0].read_bytes(), b"dicom")

    def test_rejects_path_traversal(self) -> None:
        archive = self.root / "traversal.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../outside.txt", b"unsafe")

        with self.assertRaises(ArchiveValidationError):
            safe_extract_zip(archive, self.root / "output")

        self.assertFalse((self.root / "outside.txt").exists())

    def test_rejects_symlink(self) -> None:
        archive = self.root / "symlink.zip"
        member = zipfile.ZipInfo("study/link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(member, "../outside")

        with self.assertRaises(ArchiveValidationError):
            safe_extract_zip(archive, self.root / "output")

    def test_enforces_file_count_limit(self) -> None:
        archive = self.root / "many-files.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("one.dcm", b"1")
            output.writestr("two.dcm", b"2")

        with self.assertRaises(ArchiveValidationError):
            safe_extract_zip(
                archive,
                self.root / "output",
                limits=ArchiveLimits(max_files=1),
            )

    def test_enforces_uncompressed_size_limit(self) -> None:
        archive = self.root / "large.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("volume.nii", b"12345")

        with self.assertRaises(ArchiveValidationError):
            safe_extract_zip(
                archive,
                self.root / "output",
                limits=ArchiveLimits(max_uncompressed_bytes=4),
            )


if __name__ == "__main__":
    unittest.main()
