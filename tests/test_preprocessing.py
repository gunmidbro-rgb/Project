import spacy

from src.preprocessing import TextProcessing

# A blank tokenizer is enough for _is_currency_amount: is_currency/like_num are
# rule-based lexeme attributes and don't require the full en_core_web_sm model.
nlp = spacy.blank("en")


class TestIsCurrencyAmount:
    def test_currency_symbol_itself_is_true(self) -> None:
        doc = nlp("$10 item")

        assert TextProcessing._is_currency_amount(doc[0]) is True

    def test_number_preceded_by_currency_symbol_is_true(self) -> None:
        doc = nlp("$10 item")

        assert TextProcessing._is_currency_amount(doc[1]) is True

    def test_number_followed_by_currency_symbol_is_true(self) -> None:
        doc = nlp("10$ item")

        assert TextProcessing._is_currency_amount(doc[0]) is True

    def test_standalone_number_is_false(self) -> None:
        doc = nlp("5 stars")

        assert TextProcessing._is_currency_amount(doc[0]) is False

    def test_rating_style_number_is_false(self) -> None:
        doc = nlp("9/10 rating")

        assert TextProcessing._is_currency_amount(doc[0]) is False

    def test_non_number_non_currency_token_is_false(self) -> None:
        doc = nlp("the food")

        assert TextProcessing._is_currency_amount(doc[0]) is False

    def test_number_at_start_of_doc_has_no_prev_token(self) -> None:
        doc = nlp("10 items")

        assert TextProcessing._is_currency_amount(doc[0]) is False

    def test_number_at_end_of_doc_has_no_next_token(self) -> None:
        doc = nlp("only 10")

        assert TextProcessing._is_currency_amount(doc[1]) is False
