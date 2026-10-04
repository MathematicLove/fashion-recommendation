"""Keyword parsing of the text input; runs without downloading the CLIP model."""
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.analyze import _has_word, analyze_text  # noqa: E402


class TestHasWord:
    @pytest.mark.parametrize("text,word", [
        ("red jacket", "red"),
        ("a red", "red"),
        ("navy blue jeans", "navy blue"),
        ("black jackets", "jacket"),   # plural
        ("men's shirt", "men"),         # apostrophe is a word boundary
        ("dresses", "dress"),
    ])
    def test_matches(self, text, word):
        assert _has_word(text, word)

    @pytest.mark.parametrize("text,word", [
        ("tailored blazer", "red"),
        ("embroidered shirt", "red"),
        ("romantic dress", "man"),
        ("german shepherd", "man"),
        ("girlfriend gift", "girl"),
        ("gold bracelet", "bra"),
        ("address book", "dress"),
        ("stop sign", "top"),
    ])
    def test_does_not_match_inside_other_words(self, text, word):
        assert not _has_word(text, word)


class TestColor:
    def test_plain_color(self):
        assert analyze_text("black leather jacket").color == "black"

    def test_two_word_color(self):
        assert analyze_text("navy blue jeans").color == "navy blue"

    @pytest.mark.parametrize("query", ["tailored blazer", "embroidered shirt", "colored hoodie"])
    def test_color_hidden_inside_a_word_is_ignored(self, query):
        assert analyze_text(query).color == "black"  # the default


class TestGenderAndAge:
    @pytest.mark.parametrize("query,gender", [
        ("white sneakers for men", "Men"),
        ("men's black jacket", "Men"),
        ("pink skirt for women", "Women"),
        ("womens jeans", "Women"),
    ])
    def test_explicit_gender(self, query, gender):
        assert analyze_text(query).gender == gender

    @pytest.mark.parametrize("query", ["romantic dress", "german shepherd print t-shirt"])
    def test_man_inside_another_word_is_not_a_gender(self, query):
        a = analyze_text(query)
        assert a.gender != "Men"

    def test_dress_defaults_to_women(self):
        assert analyze_text("romantic dress").gender == "Women"

    def test_girlfriend_is_not_a_kid(self):
        assert analyze_text("girlfriend gift hoodie").age == "adult"

    def test_girls_clothing_is_for_kids(self):
        a = analyze_text("girls pink dress")
        assert a.age == "kids" and a.gender == "Women"


class TestCategory:
    @pytest.mark.parametrize("query,category", [
        ("black leather jacket", "a jacket"),
        ("white sneakers", "sneakers"),
        ("black jackets", "a jacket"),
        ("navy blue jeans", "jeans"),
        ("red socks", "socks"),
        ("bra", "a bra"),
    ])
    def test_known_items(self, query, category):
        assert analyze_text(query).category == category

    def test_bracelet_is_not_a_bra(self):
        assert analyze_text("gold bracelet").category != "a bra"

    def test_unknown_text_falls_back_to_a_top(self):
        assert analyze_text("something unusual").category == "a top"

    def test_default_style(self):
        assert analyze_text("black jacket").style == "casual"

    def test_style_keyword(self):
        assert analyze_text("formal blazer").style == "formal"
