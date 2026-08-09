import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import TextCleaner


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


class TestRenameAndMap:
    def test_renames_columns_and_maps_sentiment(self, cleaner: TextCleaner) -> None:
        df = pd.DataFrame({0: [1, 2], 1: ["t1", "t2"], 2: ["text1", "text2"]})

        result = cleaner.rename_and_map(df)

        assert list(result.columns) == ["sentiment", "title", "text"]
        assert result["sentiment"].tolist() == ["negative", "positive"]


class TestConvertDtypes:
    def test_sentiment_becomes_category(self, cleaner: TextCleaner) -> None:
        df = pd.DataFrame({"sentiment": ["negative", "positive"]})

        result = cleaner.convert_dtypes(df)

        assert isinstance(result["sentiment"].dtype, pd.CategoricalDtype)


class TestCleanHtmlAndUrls:
    def test_strips_html_tags_entities_and_urls(self, cleaner: TextCleaner) -> None:
        series = pd.Series([
            "<b>great</b> product &amp; fast shipping",
            "check https://example.com/page or www.example.com out",
        ])

        result = cleaner.clean_html_and_urls(series)

        assert result[0] == "great product  fast shipping"
        assert result[1] == "check  or  out"

    def test_fills_missing_values(self, cleaner: TextCleaner) -> None:
        series = pd.Series([np.nan, "ok"])

        result = cleaner.clean_html_and_urls(series)

        assert result[0] == ""
        assert result[1] == "ok"


class TestFixEncoding:
    def test_fixes_mojibake_on_non_ascii_rows(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["itâ€™s great", "plain ascii text"])

        result = cleaner.fix_encoding(series)

        assert result[0] == "it's great"
        assert result[1] == "plain ascii text"


class TestNormalizeWhitespace:
    def test_collapses_whitespace_lowercases_and_strips(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["  Great\n\tProduct   Here  "])

        result = cleaner.normalize_whitespace(series)

        assert result[0] == "great product here"


class TestFixContractions:
    def test_expands_contractions_only_on_apostrophe_rows(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["don't like it", "no apostrophe here"])

        result = cleaner.fix_contractions(series)

        assert result[0] == "do not like it"
        assert result[1] == "no apostrophe here"


class TestSafeApply:
    def test_non_str_input_becomes_empty_string(self, cleaner: TextCleaner) -> None:
        series = pd.Series([np.nan, "text"])
        mask = pd.Series([True, True])

        result = cleaner._safe_apply(series, mask, lambda text: text.upper(), "upper")

        assert result[0] == ""
        assert result[1] == "TEXT"

    def test_exception_keeps_original_text(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["boom"])
        mask = pd.Series([True])

        def raising_func(_: str) -> str:
            raise ValueError("nope")

        result = cleaner._safe_apply(series, mask, raising_func, "raising_func")

        assert result[0] == "boom"

    def test_only_masked_rows_are_touched(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["a", "b"])
        mask = pd.Series([True, False])

        result = cleaner._safe_apply(series, mask, lambda text: text.upper(), "upper")

        assert result[0] == "A"
        assert result[1] == "b"


class TestCountElongatedWords:
    def test_counts_words_with_three_or_more_repeated_letters(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["sooo goood", "so good", "yesss"])

        result = cleaner.count_elongated_words(series)

        assert result[0] == 2
        assert result[1] == 0
        assert result[2] == 1

    def test_two_repeated_letters_do_not_count(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["book good"])

        result = cleaner.count_elongated_words(series)

        assert result[0] == 0


class TestCollapseElongatedChars:
    def test_collapses_runs_to_two_letters(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["sooo goood book"])

        result = cleaner.collapse_elongated_chars(series)

        assert result[0] == "soo good book"


class TestCombineTitleText:
    def test_combines_title_and_text(self, cleaner: TextCleaner) -> None:
        df = pd.DataFrame({"title": ["great"], "text": ["works well"]})

        result = cleaner.combine_title_text(df)

        assert result[0] == "great works well"


class TestDropDuplicatesAndShort:
    def test_drops_duplicate_and_short_reviews(self, cleaner: TextCleaner) -> None:
        df = pd.DataFrame({"review": ["same review", "same review", "okay", "hi"]})

        result = cleaner.drop_duplicates_and_short(df)

        assert result["review"].tolist() == ["same review", "okay"]
        assert list(result.index) == [0, 1]


class TestCleanText:
    def test_runs_full_pipeline_in_order(self, cleaner: TextCleaner) -> None:
        series = pd.Series(["  <b>I don't like it</b>  https://x.com  "])

        result = cleaner.clean_text(series)

        assert result[0] == "i do not like it"


class TestProcessing:
    def test_produces_expected_columns_and_row_counts(self, cleaner: TextCleaner) -> None:
        df = pd.DataFrame({
            0: [1, 2, 1],
            1: ["Bad", "Good", "Bad"],
            2: ["Did not work", "Works great!", "Did not work"],
        })

        result = cleaner.processing(df)

        assert set(["sentiment", "title", "text", "review", "emphasis_count"]).issubset(result.columns)
        # third row is a duplicate of the first (same title+text) -> dropped
        assert len(result) == 2
