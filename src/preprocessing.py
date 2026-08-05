import spacy
import pandas as pd


class TextProcessing:
    """Tokenization + POS filter + lemmatization pipeline for cleaned review text."""

    def __init__(self) -> None:
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    @staticmethod
    def _is_currency_amount(token) -> bool:
        """True for a currency symbol or a number adjacent to one (e.g. 10$, $3.99).

        Doesn't flag standalone numbers like ratings ("5 stars", "9/10").
        """
        if token.is_currency:
            return True
        if not token.like_num:
            return False
        doc = token.doc
        prev_is_currency = token.i > 0 and doc[token.i - 1].is_currency
        next_is_currency = token.i < len(doc) - 1 and doc[token.i + 1].is_currency
        return prev_is_currency or next_is_currency

    def preprocess(
        self,
        df: pd.DataFrame,
        pos_to_remove: set[str] = {'X', 'SPACE', 'PUNCT'},
        batch_size: int = 1000,
        n_process: int = 1,
        ) -> pd.DataFrame:
        """Tokenize, POS-filter, lemmatize — all in a single spaCy pass. Mutates df in place."""
        docs = self.nlp.pipe(df['review'], batch_size=batch_size, n_process=n_process)

        df['tokens'] = [
            [
                token.lemma_ for token in doc
                if token.pos_ not in pos_to_remove and not self._is_currency_amount(token)
            ]
            for doc in docs]
        return df

