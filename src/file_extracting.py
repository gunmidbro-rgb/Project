import logging
import os
import tarfile
import zipfile

import pandas as pd

# Basic logging setup so pipeline progress/errors are timestamped and traceable.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataExtract:
    """Unpack the Amazon Reviews Polarity dataset (.zip -> .tgz -> .csv) and build a row-sampled CSV."""

    def __init__(self, base_folder: str = "data/raw") -> None:
        """Build all file paths relative to `base_folder` (relative path only, per project rules)."""
        self.base_folder = base_folder
        self.zip_path = os.path.join(base_folder, "amazon_review_polarity_csv.tgz.zip")
        self.tgz_path = os.path.join(base_folder, "amazon_review_polarity_csv.tgz")
        self.extracted_folder = os.path.join(base_folder, "amazon_review_polarity_csv")
        self.source_csv = os.path.join(self.extracted_folder, "train.csv")
        self.output_sample_csv = os.path.join(self.extracted_folder, "sample.csv")

    def extract_zip(self) -> None:
        """Extract the inner .tgz archive out of the outer .zip file."""
        # Skip re-work if a previous run already produced the .tgz file.
        if os.path.exists(self.tgz_path):
            logger.info("Skip extract_zip: %s already exists", self.tgz_path)
            return

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")

        logger.info("Extracting zip: %s", self.zip_path)
        inner_file_name = "amazon_review_polarity_csv.tgz"
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                if inner_file_name not in zip_ref.namelist():
                    raise KeyError(f"'{inner_file_name}' not found inside {self.zip_path}")
                zip_ref.extract(inner_file_name, path=self.base_folder)
        except zipfile.BadZipFile as exc:
            raise zipfile.BadZipFile(f"Corrupted zip file: {self.zip_path}") from exc
        logger.info("extract_zip successful")

    def extract_tgz(self) -> None:
        """Extract the .tgz archive to reveal train.csv/test.csv."""
        # Skip re-work if a previous run already produced the source CSV.
        if os.path.exists(self.source_csv):
            logger.info("Skip extract_tgz: %s already exists", self.source_csv)
            return

        if not os.path.exists(self.tgz_path):
            raise FileNotFoundError(f"Tgz file not found: {self.tgz_path}")

        logger.info("Extracting tgz: %s", self.tgz_path)
        try:
            with tarfile.open(self.tgz_path, "r:gz") as tgz:
                # filter="data" blocks unsafe members (e.g. path traversal) per PEP 706.
                tgz.extractall(path=self.base_folder, filter="data")
        except tarfile.ReadError as exc:
            raise tarfile.ReadError(f"Corrupted tgz file: {self.tgz_path}") from exc
        logger.info("extract_tgz successful")

    def get_sample(self, nums_row: int) -> None:
        """Write the first `nums_row` rows of the source CSV to a sample CSV."""
        if nums_row <= 0:
            raise ValueError(f"nums_row must be a positive integer, got {nums_row}")

        if not os.path.exists(self.source_csv):
            raise FileNotFoundError(f"Source CSV not found: {self.source_csv}")

        logger.info("Get sample: %s rows", nums_row)
        try:
            # nrows stops parsing after the requested rows, keeping memory usage bounded.
            sample = pd.read_csv(self.source_csv, nrows=nums_row, header=None)
        except pd.errors.EmptyDataError as exc:
            raise pd.errors.EmptyDataError(f"Source CSV is empty: {self.source_csv}") from exc

        sample.to_csv(self.output_sample_csv, index=False, header=None)
        logger.info("Sample written to %s", self.output_sample_csv)

    def runpipeline(self) -> None:
        """Run the full pipeline: unzip -> untar -> sample 50,000 rows."""
        logger.info("Start run pipeline")
        self.extract_zip()
        self.extract_tgz()
        self.get_sample(nums_row=50000)
        logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    extractor = DataExtract()
    extractor.runpipeline()