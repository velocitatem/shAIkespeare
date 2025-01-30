
"""
shAIkespeare:
In this assignment, you will build a text generation model that works with 2-grams/3-grams/4-grams to imitate the style of William Shakespeare. Your task is to implement various functions that will help in generating text based on ngrams, trigrams and quadgrams.
"""
import nltk
from nltk.tokenize import RegexpTokenizer

nltk.download('gutenberg')

# Data Preparation: We load the works of William Shakespeare from the nltk corpus and preprocess the text.
# Create a list of grams
GRAM_SIZE = 2 # instructed to use ngrams but just in case we want to change it to trigrams or quadgrams
PHRASE = ["to", "be", "or", "not", "to", "be"]

def load_data() -> list:
    """
    Load the works of William Shakespeare from the nltk corpus and preprocess the text.
    """
    # Load the works of William Shakespeare from the nltk corpus
    shakespeare = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt') # load the text of Hamlet which is the most interesting one imo
    shakespeare = shakespeare.lower().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ') # remove newlines and carriage returns
    tokenizer = RegexpTokenizer(r'\w+') # remove punctuation and special characters - regex to match words only
    shakespeare = tokenizer.tokenize(shakespeare)
    return shakespeare

def create_grams(tokens, gram_size=GRAM_SIZE) -> list:
    """
    Create a list of grams based on the specified gram size.
    """
    # Tokenize the text
    # Create a list of grams
    grams = []
    for i in range(len(tokens) - gram_size + 1):
        gram = ' '.join(tokens[i:i + gram_size])
        grams.append(gram)

    return grams

def create_ngram_counts(ngrams) -> dict:
    from_ngram_to_next_token_counts = {}
    for i in range(len(ngrams) - 1):
        ngram = tuple(ngrams[i].split()) # convert the ngram string to a tuple
        next_token = ngrams[i + 1].split()[-1] # get the last token of the next ngram
        if ngram not in from_ngram_to_next_token_counts:
            from_ngram_to_next_token_counts[ngram] = {} # we start for our listing
        if next_token not in from_ngram_to_next_token_counts[ngram]:
            from_ngram_to_next_token_counts[ngram][next_token] = 0
        from_ngram_to_next_token_counts[ngram][next_token] += 1

    return from_ngram_to_next_token_counts

def create_ngram_probs(ngram_counts) -> dict:
    """
    Create a dictionary of ngram probabilities.
    """
    from_ngram_to_next_token_probs = {}
    for ngram, next_token_counts in ngram_counts.items():
        total_count = sum(next_token_counts.values()) # just a process of dividing the counts by the total count
        from_ngram_to_next_token_probs[ngram] = {next_token: count / total_count for next_token, count in next_token_counts.items()}

    return from_ngram_to_next_token_probs

def sample_next_token(ngram, from_ngram_to_next_token_probs):
    """
    Sample the next token based on the probability distribution from from_ngram_to_next_token_probs.
    """
    import random
    next_token_probs = from_ngram_to_next_token_probs[ngram]
    next_tokens = list(next_token_probs.keys())
    probs = list(next_token_probs.values()) # our weights
    return random.choices(next_tokens, weights=probs)[0] # nice helper for use to use

def generate_text_from_ngram(ngram, num_words, from_ngram_to_next_token_probs, gram_size=GRAM_SIZE) -> str:
    """
    Generate text by starting with an initial ngram and sampling the next token iteratively.
    """
    text = list(ngram)
    for _ in range(num_words):
        next_token = sample_next_token(tuple(text[-gram_size:]), from_ngram_to_next_token_probs)
        text.append(next_token)
    return ' '.join(text)

def main():
    """
    This defines our main pipeline.
    """

#  _____         _         _
# |_   _|_ _ ___| | __    / |
#   | |/ _` / __| |/ /    | |
#   | | (_| \__ \   <     | |
#   |_|\__,_|___/_|\_\    |_|
#

    # Load the data
    text = load_data()
    print(f"Loaded {len(text)} tokens from the works of William Shakespeare.")
    # Create a list of grams
    grams = create_grams(text, GRAM_SIZE)
    print(f"Created {len(grams)} {GRAM_SIZE}-grams.")
    # Create a dictionary of ngram counts
    ngram_counts = create_ngram_counts(grams)
    CASE = tuple(PHRASE[0:GRAM_SIZE])
    print(f"Created {len(ngram_counts)} {GRAM_SIZE}-grams . Example: {ngram_counts[CASE]}")

#  _____         _         ____
# |_   _|_ _ ___| | __    |___ \
#   | |/ _` / __| |/ /      __) |
#   | | (_| \__ \   <      / __/
#   |_|\__,_|___/_|\_\    |_____|


    # Create a dictionary of ngram probabilities
    ngram_probs = create_ngram_probs(ngram_counts)
    print(f"Created {len(ngram_probs)} {GRAM_SIZE}-gram probabilities. Example: {ngram_probs[CASE]}")

#  _____         _         _____
# |_   _|_ _ ___| | __    |___ /
#   | |/ _` / __| |/ /      |_ \
#   | | (_| \__ \   <      ___) |
#   |_|\__,_|___/_|\_\    |____/

    # Sample the next token
    ngram = CASE
    next_token = sample_next_token(ngram, ngram_probs)
    print(f"Sampled next token: {next_token}")


#  _____         _         _  _
# |_   _|_ _ ___| | __    | || |
#   | |/ _` / __| |/ /    | || |_
#   | | (_| \__ \   <     |__   _|
#   |_|\__,_|___/_|\_\       |_|

    # Generate text

    # Generate text from the ngram ('to', 'be')
    num_words = 10
    generated_text = generate_text_from_ngram(ngram, num_words, ngram_probs)
    print(f"Generated text: {generated_text}")

#  _____         _         ____
# |_   _|_ _ ___| | __    | ___|
#   | |/ _` / __| |/ /    |___ \
#   | | (_| \__ \   <      ___) |
#   |_|\__,_|___/_|\_\    |____/

    # Lets try different ngram sizes
    sizes = [2, 3, 4]
    for size in sizes: # we just repeat the process we have been doing throughout the tasks into a loop for diff sizes
        print(f"Generating text for {size}-grams")
        grams = create_grams(text, size)
        ngram_counts = create_ngram_counts(grams)
        ngram_probs = create_ngram_probs(ngram_counts)
        ngram = tuple(PHRASE[0:size])
        generated_text = generate_text_from_ngram(ngram, 20, ngram_probs, size)
        print(f"Generated text: {generated_text}")


#  _____         _          __
# |_   _|_ _ ___| | __     / /_
#   | |/ _` / __| |/ /    | '_ \
#   | | (_| \__ \   <     | (_) |
#   |_|\__,_|___/_|\_\     \___/
# The results of the survey can be seen on the README.md file.

if __name__ == "__main__":
    main()
