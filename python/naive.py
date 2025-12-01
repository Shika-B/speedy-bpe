from collections import Counter


class Token:
    def __init__(self, s, tok_id, word_id):
        self.s = s
        self.tok_id = tok_id
        self.word_id = word_id

    def __repr__(self):
        return f"Token({self.s}, tok_id: {self.tok_id} word_id:{self.word_id})"


def initial_vocab(words):
    fresh_tok = 0
    vocab = dict()
    for word in words:
        for char in word:
            if not char in vocab:
                vocab[char] = fresh_tok
                fresh_tok += 1
    return vocab


def get_stats(tokens):
    stats = Counter()
    for tok1, tok2 in zip(tokens, tokens[1:]):
        if tok1.word_id == tok2.word_id:
            stats[(tok1.tok_id, tok2.tok_id)] += 1
    return stats


def merge(tokens, pair, fresh_token):
    new_tokens = []
    i = 0
    while i < len(tokens) - 1:
        tok1, tok2 = tokens[i], tokens[i + 1]
        if tok1.word_id == tok2.word_id and (tok1.tok_id, tok2.tok_id) == pair:
            new_tokens.append(Token(tok1.s + tok2.s, fresh_token, tok1.word_id))
            i += 2
        else:
            new_tokens.append(tok1)
            i += 1
    if i == len(tokens) - 1:
        new_tokens.append(tokens[i])
    return new_tokens


def train(words, num_merges, verbose=True):
    vocab = initial_vocab(words)
    reverse_vocab = {vocab[c]: c for c in vocab.keys()}
    tokens = [
        Token(c, vocab[c], word_id) for word_id, word in enumerate(words) for c in word
    ]

    fresh_token = len(vocab)
    merge_tree = []
    for _ in range(num_merges):
        stats = get_stats(tokens)
        try:
            ((left, right), _count) = stats.most_common(1)[0]
        except IndexError:
            break

        if verbose:
            print(f"Merging the pair ({reverse_vocab[left]}, {reverse_vocab[right]})")

        tokens = merge(tokens, (left, right), fresh_token)

        new_token_s = reverse_vocab[left] + reverse_vocab[right]
        vocab[new_token_s] = fresh_token
        reverse_vocab[fresh_token] = new_token_s

        merge_tree.append(((left, right), fresh_token))
        fresh_token += 1

    return vocab, merge_tree


def encode(vocab, merge_tree, words):
    tokens = [
        Token(c, vocab[c], word_id) for word_id, word in enumerate(words) for c in word
    ]

    for (left, right), new in merge_tree:
        tokens = merge(tokens, (left, right), new)
    return tokens


def decode(tokens):
    words = []
    curr_word = []
    i = 0
    while i < len(tokens):
        if i > 0 and tokens[i - 1].word_id != tokens[i].word_id:
            words.append("".join(curr_word))
            curr_word = []
        curr_word.append(tokens[i].s)
        i += 1
    words.append("".join(curr_word))
    return words


def test_train():
    words = ["that", "this", "the", "he"]

    vocab, merge_tree = train(words, 6)
    tokens = encode(vocab, merge_tree, words)
    print("Tokens: ", [tok.tok_id for tok in tokens])
    print("Decoded tokens: ", decode(tokens))


def test_train_large():
    with open("data/fra.txt", "r", encoding="utf-8") as file:
        txt = file.read()
        lines = txt.strip().split("\n")
        fra, eng = [], []
        for line in lines:
            cols = line.split("\t")
            eng.append(cols[0])
            fra.append(cols[1])
    import re

    regex = re.compile("\\s|\\.|\\!|\\?")

    eng_words = []
    for sentence in eng:
        for word in regex.split(sentence.lower()):
            if len(word) > 0:
                eng_words.append(word)

    train(eng_words, 40, verbose=True)


if __name__ == "__main__":
    test_train()
