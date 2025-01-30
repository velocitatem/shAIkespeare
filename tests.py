"""
We will test our methods here.
"""

#     next_token_probs = from_ngram_to_next_token_probs[ngram]
import pytest
from shAIkespeare.generator import (
    load_data,
    create_grams,
    create_ngram_counts,
    create_ngram_probs,
    sample_next_token,
    generate_text_from_ngram
)

# Assuming GRAM_SIZE and PHRASE are imported or defined within the test context

def test_load_data():
    tokens = load_data()
    assert isinstance(tokens, list), "load_data should return a list of tokens"
    assert all(isinstance(token, str) for token in tokens), "Each item in the returned list should be a string (token)"
    # Optionally check length if known
    assert len(tokens) > 0, "The list of tokens should not be empty"

def test_create_grams():
    text = load_data()
    grams_size_2 = create_grams(text, 2)
    grams_size_3 = create_grams(text, 3)

    assert isinstance(grams_size_2, list), "create_grams should return a list"
    assert all(isinstance(gram, str) for gram in grams_size_2), "Each item in the returned list should be a string (n-gram)"
    assert len(grams_size_2[0].split()) == 2, "First 2-gram should have two words"

    assert len(grams_size_3) < len(text) - 1, "The number of 3-grams should be less than the total tokens minus one"
    assert len(grams_size_3[0].split()) == 3, "First 3-gram should have three words"

def test_create_ngram_counts():
    text = load_data()
    grams_size_2 = create_grams(text, 2)
    ngram_counts = create_ngram_counts(grams_size_2)

    assert isinstance(ngram_counts, dict), "create_ngram_counts should return a dictionary"
    for gram in ngram_counts:
        assert isinstance(gram, tuple), "Keys of the dictionary should be tuples representing n-grams"
        assert isinstance(ngram_counts[gram], dict),"Values of the dictionary should be dictionaries (maps of occurances)"

def test_create_ngram_probs():
    text = load_data()
    grams_size_2 = create_grams(text, 2)
    ngram_counts = create_ngram_counts(grams_size_2)
    ngram_probs = create_ngram_probs(ngram_counts)

    assert isinstance(ngram_probs, dict), "create_ngram_probs should return a dictionary"
    for gram, probs in ngram_probs.items():
        assert isinstance(gram, tuple), "Keys of the dictionary should be tuples representing n-grams"
        assert all(isinstance(p, float) for p in probs.values()), "Probabilities should be floats"
        total_prob = sum(probs.values())
        assert abs(total_prob - 1.0) < 1e-6, "Total probabilities should sum to approximately 1"

def test_sample_next_token():
    text = load_data()
    grams_size_2 = create_grams(text, 2)
    ngram_counts = create_ngram_counts(grams_size_2)
    ngram_probs = create_ngram_probs(ngram_counts)

    # Sample a valid n-gram from the keys
    if ngram_probs:
        sample_gram = next(iter(ngram_probs))
        token = sample_next_token(sample_gram, ngram_probs)
        assert isinstance(token, str), "sample_next_token should return a string"

def test_generate_text_from_ngram():
    text = load_data()
    grams_size_2 = create_grams(text, 2)
    ngram_counts = create_ngram_counts(grams_size_2)
    ngram_probs = create_ngram_probs(ngram_counts)

    if ngram_probs:
        sample_gram = next(iter(ngram_probs))
        generated_text = generate_text_from_ngram(sample_gram, 10, ngram_probs, 2)
        assert isinstance(generated_text, str), "generate_text_from_ngram should return a string"
        words = generated_text.split()
        assert len(words) == 12, "Generated text should have k+n words hwere k is generated size and n is base gram size"

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
