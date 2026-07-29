import spacy
import pandas as pd


class TextProcessing:
    """Tokenization pipeline for cleaned review text."""

    def __init__(self) -> None:
        # TODO: swap for en_core_web_sm (tagger + lemmatizer) once stopword
        # removal/lemmatization are added; blank('en') is tokenizer-only.
        self.nlp = spacy.blank('en')

    def tokenization(self, df: pd.DataFrame, batch_size: int = 1000, n_process: int = 1) -> pd.DataFrame:
        data = df.copy()
        docs = self.nlp.pipe(data['review'].astype(str), batch_size=batch_size, n_process=n_process)
        data['tokens'] = [[token.text for token in doc] for doc in docs]
        return data

