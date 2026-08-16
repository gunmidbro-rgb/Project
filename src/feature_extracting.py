import ast

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def _as_token_lists(tokens: pd.Series) -> pd.Series:
    """Parse `tokens` back into lists if they came from a CSV round-trip (stored as `"['a', 'b']"` strings)."""
    if len(tokens) > 0 and isinstance(tokens.iloc[0], str):
        return tokens.apply(ast.literal_eval)
    return tokens


def extraction(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """TF-IDF vectorize the pre-tokenized `tokens` column (unigrams + bigrams).

    Fits a new vectorizer if none is given (train set); reuses a fitted one
    via `.transform()` otherwise (test/val set) to avoid data leakage.
    """
    tokens = _as_token_lists(df['tokens'])

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer=lambda x: x, lowercase=False, ngram_range=(1, 2), min_df=5,
        )
        X = vectorizer.fit_transform(tokens)
    else:
        X = vectorizer.transform(tokens)
    return X, vectorizer
