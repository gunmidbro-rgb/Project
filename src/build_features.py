import logging
import os

import pandas as pd

from preprocessing import TextProcessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLEANED_CSV = "data/cleaned/clean_train.csv"
FEATURES_CSV = "data/cleaned/features_train.csv"
BATCH_SIZE = 500
N_PROCESS = 4
OUTPUT_COLUMNS = ["sentiment", "review", "emphasis_count", "tokens"]


def build_features(
    cleaned_csv: str = CLEANED_CSV,
    features_csv: str = FEATURES_CSV,
    batch_size: int = BATCH_SIZE,
    n_process: int = N_PROCESS,
) -> None:
    """Tokenize/lemmatize the cleaned dataset and export the training-ready feature columns."""
    logger.info("Loading cleaned data from %s", cleaned_csv)
    cleaned_data = pd.read_csv(cleaned_csv)

    processor = TextProcessing()
    logger.info(
        "Running spaCy preprocess (batch_size=%d, n_process=%d) on %d rows",
        batch_size, n_process, len(cleaned_data),
    )
    processed_data = processor.preprocess(cleaned_data, batch_size=batch_size, n_process=n_process)

    features = processed_data[OUTPUT_COLUMNS]

    os.makedirs(os.path.dirname(features_csv), exist_ok=True)
    features.to_csv(features_csv, index=False)
    logger.info("Saved %d rows to %s", len(features), features_csv)


if __name__ == "__main__":
    build_features()
