import io
import os
import tarfile
import zipfile

import pandas as pd
import pytest

from src.file_extracting import DataExtract


def make_zip(zip_path: str, inner_name: str, inner_content: bytes = b"dummy tgz bytes") -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(inner_name, inner_content)


def make_tgz(tgz_path: str, arcname: str, content: bytes = b"1,a,b\n2,c,d\n") -> None:
    with tarfile.open(tgz_path, "w:gz") as tar:
        info = tarfile.TarInfo(name=arcname)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))


@pytest.fixture
def extractor(tmp_path) -> DataExtract:
    base_folder = tmp_path / "raw"
    base_folder.mkdir()
    return DataExtract(base_folder=str(base_folder))


class TestExtractZip:
    def test_raises_when_zip_missing(self, extractor: DataExtract) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.extract_zip()

    def test_skips_when_tgz_already_exists(self, extractor: DataExtract) -> None:
        with open(extractor.tgz_path, "wb") as f:
            f.write(b"existing tgz")

        extractor.extract_zip()  # zip_path intentionally missing; should not raise

        assert os.path.exists(extractor.tgz_path)

    def test_extracts_inner_tgz_from_zip(self, extractor: DataExtract) -> None:
        make_zip(extractor.zip_path, "amazon_review_polarity_csv.tgz", b"payload-bytes")

        extractor.extract_zip()

        with open(extractor.tgz_path, "rb") as f:
            assert f.read() == b"payload-bytes"

    def test_raises_keyerror_when_inner_file_missing(self, extractor: DataExtract) -> None:
        make_zip(extractor.zip_path, "wrong_name.tgz")

        with pytest.raises(KeyError):
            extractor.extract_zip()

    def test_raises_badzipfile_for_corrupted_zip(self, extractor: DataExtract) -> None:
        with open(extractor.zip_path, "wb") as f:
            f.write(b"not a real zip")

        with pytest.raises(zipfile.BadZipFile):
            extractor.extract_zip()


class TestExtractTgz:
    def test_raises_when_tgz_missing(self, extractor: DataExtract) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.extract_tgz()

    def test_skips_when_source_csv_already_exists(self, extractor: DataExtract) -> None:
        os.makedirs(extractor.extracted_folder, exist_ok=True)
        with open(extractor.source_csv, "w") as f:
            f.write("1,title,text\n")

        extractor.extract_tgz()  # tgz_path intentionally missing; should not raise

        assert os.path.exists(extractor.source_csv)

    def test_extracts_csv_from_tgz(self, extractor: DataExtract) -> None:
        make_tgz(extractor.tgz_path, "amazon_review_polarity_csv/train.csv", b"1,a,b\n2,c,d\n")

        extractor.extract_tgz()

        with open(extractor.source_csv, "rb") as f:
            assert f.read() == b"1,a,b\n2,c,d\n"

    def test_raises_readerror_for_corrupted_tgz(self, extractor: DataExtract) -> None:
        with open(extractor.tgz_path, "wb") as f:
            f.write(b"not a real tgz")

        with pytest.raises(tarfile.ReadError):
            extractor.extract_tgz()


class TestGetSample:
    # output_sample_csv is hardcoded to "data/sample/sample.csv" (relative to cwd,
    # not base_folder) -- chdir into tmp_path so tests never write into the real
    # project data/ folder.
    @pytest.fixture(autouse=True)
    def _chdir_tmp(self, monkeypatch, tmp_path) -> None:
        monkeypatch.chdir(tmp_path)

    @staticmethod
    def _write_source_csv(extractor: DataExtract, n_rows: int) -> None:
        os.makedirs(extractor.extracted_folder, exist_ok=True)
        with open(extractor.source_csv, "w") as f:
            for i in range(n_rows):
                f.write(f"{i % 2 + 1},title{i},text body {i}\n")

    def test_raises_for_nonpositive_nums_row(self, extractor: DataExtract) -> None:
        with pytest.raises(ValueError):
            extractor.get_sample(nums_row=0)

    def test_raises_when_source_csv_missing(self, extractor: DataExtract) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.get_sample(nums_row=5)

    def test_raises_emptydataerror_for_empty_csv(self, extractor: DataExtract) -> None:
        os.makedirs(extractor.extracted_folder, exist_ok=True)
        open(extractor.source_csv, "w").close()

        with pytest.raises(pd.errors.EmptyDataError):
            extractor.get_sample(nums_row=5)

    def test_raises_when_nums_row_exceeds_total_rows(self, extractor: DataExtract) -> None:
        self._write_source_csv(extractor, n_rows=5)

        with pytest.raises(ValueError):
            extractor.get_sample(nums_row=10)

    def test_writes_sampled_csv_with_correct_row_count(self, extractor: DataExtract, tmp_path) -> None:
        self._write_source_csv(extractor, n_rows=20)

        extractor.get_sample(nums_row=5, random_state=42)

        output_path = tmp_path / extractor.output_sample_csv
        assert output_path.exists()
        assert len(pd.read_csv(output_path, header=None)) == 5

    def test_same_random_state_is_reproducible(self, extractor: DataExtract, tmp_path) -> None:
        self._write_source_csv(extractor, n_rows=20)
        output_path = tmp_path / extractor.output_sample_csv

        extractor.get_sample(nums_row=5, random_state=42)
        first = pd.read_csv(output_path, header=None)
        os.remove(output_path)
        extractor.get_sample(nums_row=5, random_state=42)
        second = pd.read_csv(output_path, header=None)

        pd.testing.assert_frame_equal(first, second)


class TestRunPipeline:
    def test_calls_steps_in_order_with_expected_args(self, extractor: DataExtract, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(extractor, "extract_zip", lambda: calls.append("extract_zip"))
        monkeypatch.setattr(extractor, "extract_tgz", lambda: calls.append("extract_tgz"))
        monkeypatch.setattr(extractor, "get_sample", lambda nums_row: calls.append(("get_sample", nums_row)))

        extractor.runpipeline()

        assert calls == ["extract_zip", "extract_tgz", ("get_sample", 50000)]
