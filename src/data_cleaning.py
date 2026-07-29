import re

import pandas as pd
import contractions
import ftfy

HTML_TAG_PATTERN = r'<[^>]+>|&[a-zA-Z]+;'
URL_PATTERN = r'https?://\S+|www\.\S+'
WHITESPACE_PATTERN = r'\s+'
NON_ASCII_PATTERN = r'[^\x00-\x7F]'

class TextCleaner:
    """Cleaning pipeline for the Amazon Reviews polarity dataset."""

    MIN_TEXT_LENGTH = 3

    def rename_and_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename positional columns and map sentiment codes (1/2) to labels."""
        data = df.rename(columns={0: 'sentiment', 1: 'title', 2: 'text'})
        data['sentiment'] = data['sentiment'].replace({1: 'negative', 2: 'positive'})
        return data

    def convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast sentiment to a category dtype."""
        data = df.copy()
        data['sentiment'] = data['sentiment'].astype('category')
        return data

    def clean_html_and_urls(self, series: pd.Series) -> pd.Series:
        """Fill missing text and strip HTML tags/entities and URLs (vectorized)."""
        cleaned = series.fillna('').astype(str)
        cleaned = cleaned.str.replace(HTML_TAG_PATTERN, '', regex=True)
        cleaned = cleaned.str.replace(URL_PATTERN, '', regex=True)
        return cleaned

    def _safe_fix_encoding(self, text) -> str:
        """Apply ftfy.fix_text to one row; non-str -> '', errors fall back to original text."""
        if not isinstance(text, str):
            return ''
        try:
            return ftfy.fix_text(text)
        except Exception as exc:
            print(f"Warning: ftfy.fix_text failed on a row ({exc!r}); keeping original text")
            return text

    def fix_encoding(self, series: pd.Series) -> pd.Series:
        """Fix mojibake/curly quotes via ftfy on non-ASCII rows only (ftfy isn't vectorized)."""
        result = series.copy()
        has_non_ascii = result.str.contains(NON_ASCII_PATTERN, regex=True).fillna(False)
        result.loc[has_non_ascii] = result.loc[has_non_ascii].apply(self._safe_fix_encoding)
        return result

    def normalize_whitespace(self, series: pd.Series) -> pd.Series:
        """Collapse newlines/tabs/repeated spaces, lowercase, and strip (vectorized)."""
        cleaned = series.str.replace(WHITESPACE_PATTERN, ' ', regex=True)
        return cleaned.str.lower().str.strip()

    def _safe_fix_contraction(self, text) -> str:
        """Apply contractions.fix to one row; non-str -> '', errors fall back to original text."""
        if not isinstance(text, str):
            return ''
        try:
            return contractions.fix(text)
        except Exception as exc:
            print(f"Warning: contractions.fix failed on a row ({exc!r}); keeping original text")
            return text

    def fix_contractions(self, series: pd.Series) -> pd.Series:
        """Expand contractions (e.g. don't -> do not) on rows with an apostrophe only (not vectorized)."""
        result = series.copy()
        has_apostrophe = result.str.contains("'", regex=False).fillna(False)
        result.loc[has_apostrophe] = result.loc[has_apostrophe].apply(self._safe_fix_contraction)
        return result

    def clean_text(self, series: pd.Series) -> pd.Series:
        """Run the full text pipeline on one column: HTML/URL removal, whitespace
        normalization, lowercasing, and contraction expansion."""
        cleaned = self.clean_html_and_urls(series)
        cleaned = self.fix_encoding(cleaned)
        cleaned = self.normalize_whitespace(cleaned)
        cleaned = self.fix_contractions(cleaned)
        return cleaned

    def combine_title_text(self, df: pd.DataFrame) -> pd.Series:
        """Combine title + text into one review-context column."""
        return (df['title'] + ' ' + df['text']).str.strip()

    def drop_duplicates_and_short(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate and too-short reviews, logging how many were dropped."""
        data = df.copy()
        before = len(data)

        data = data.drop_duplicates(subset=['review'])
        n_duplicates = before - len(data)
        print(f"Removed {n_duplicates} duplicate reviews")

        data = data[data['review'].str.len() >= self.MIN_TEXT_LENGTH]
        n_short = before - n_duplicates - len(data)
        print(f"Removed {n_short} empty/short reviews (< {self.MIN_TEXT_LENGTH} chars)")

        return data.reset_index(drop=True)

    def processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and print a before/after validation report."""
        before_rows = len(df)

        data = self.rename_and_map(df)
        data = self.convert_dtypes(data)

        data['title'] = self.clean_text(data['title'])
        data['text'] = self.clean_text(data['text'])
        data['review'] = self.combine_title_text(data)

        data = self.drop_duplicates_and_short(data)

        print(f"Rows: {before_rows} -> {len(data)}")
        print("Missing values per column:")
        print(data.isna().sum())
        data.info(memory_usage='deep')

        return data
