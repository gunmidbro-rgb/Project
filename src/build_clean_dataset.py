import logging
import os

import pandas as pd

from data_cleaning import TextCleaner
from file_extracting import DataExtract

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_SIZE = 1_000_000
RANDOM_STATE = 42
SAMPLE_CSV = "data/sample/sample_1m.csv"
CLEANED_CSV = "data/cleaned/clean_train.csv"


def build_clean_dataset(
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    sample_csv: str = SAMPLE_CSV,
    cleaned_csv: str = CLEANED_CSV,
) -> None:
    """Sample `sample_size` rows from the raw dataset, clean them, and export to `cleaned_csv`."""
    extractor = DataExtract()
    extractor.output_sample_csv = sample_csv
    extractor.get_sample(nums_row=sample_size, random_state=random_state)

    raw_data = pd.read_csv(sample_csv, header=None, index_col=False)

    cleaner = TextCleaner()
    cleaned_data = cleaner.processing(raw_data)

    os.makedirs(os.path.dirname(cleaned_csv), exist_ok=True)
    cleaned_data.to_csv(cleaned_csv, index=False)
    logger.info("Saved %d rows to %s", len(cleaned_data), cleaned_csv)


if __name__ == "__main__":
    build_clean_dataset()
