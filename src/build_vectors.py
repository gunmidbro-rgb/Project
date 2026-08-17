import logging
import os

import joblib
import pandas as pd
from scipy.sparse import save_npz

from feature_extracting import extraction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_CSV = "data/cleaned/features_train.csv"
MATRIX_OUTPUT = "data/cleaned/tfidf_train.npz"
VECTORIZER_OUTPUT = "data/cleaned/tfidf_vectorizer.joblib"


def build_vectors(
    features_csv: str = FEATURES_CSV,
    matrix_output: str = MATRIX_OUTPUT,
    vectorizer_output: str = VECTORIZER_OUTPUT,
) -> None:
    """TF-IDF vectorize the `tokens` column and save the matrix + vectorizer."""
    logger.info("Loading features from %s", features_csv)
    df = pd.read_csv(features_csv)

    logger.info("Vectorizing %d rows (TF-IDF)", len(df))
    X, vectorizer = extraction(df)  # handles tokens stored as string-of-list internally

    os.makedirs(os.path.dirname(matrix_output), exist_ok=True)
    save_npz(matrix_output, X)
    joblib.dump(vectorizer, vectorizer_output)
    logger.info("Saved matrix %s shape=%s and vectorizer to %s", matrix_output, X.shape, vectorizer_output)


if __name__ == "__main__":
    build_vectors()
