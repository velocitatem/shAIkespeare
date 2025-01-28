# Sh _AI_ kespeare

This is an implementation of an n-gram text generator for shakespears work.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Main Functions

The main functions in `shAIkespeare/generator.py` are:

- `load_data()`: Loads and preprocesses the works of William Shakespeare.
- `create_grams(tokens, gram_size)`: Creates a list of grams based on the specified gram size.
- `create_ngram_counts(ngrams)`: Creates a dictionary of ngram counts.
- `create_ngram_probs(ngram_counts)`: Creates a dictionary of ngram probabilities.
- `sample_next_token(ngram, from_ngram_to_next_token_probs)`: Samples the next token based on the probability distribution.
- `generate_text_from_ngram(ngram, num_words, from_ngram_to_next_token_probs, gram_size)`: Generates text by starting with an initial ngram and sampling the next token iteratively.

### Running the Main Function

To run the `main` function, execute the following command:

```bash
python shAIkespeare/generator.py
```

### Example

Here is an example of generating text using the model:

```python
from shAIkespeare.generator import load_data, create_grams, create_ngram_counts, create_ngram_probs, generate_text_from_ngram

text = load_data()
grams = create_grams(text, 2)
ngram_counts = create_ngram_counts(grams)
ngram_probs = create_ngram_probs(ngram_counts)

# Generate text from the ngram ('to', 'be')
ngram = ('to', 'be')
num_words = 10
generated_text = generate_text_from_ngram(ngram, num_words, ngram_probs, 2)
```

Expected output:

```
Generated text: to be or not to be that is the question
```

### Running Tests

To run the available tests in `tests.py`, use the following command:

```bash
pytest tests.py
```
