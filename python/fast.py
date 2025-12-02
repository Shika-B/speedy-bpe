from multiheap import MultisetHeap

from collections import defaultdict

import re


class TokenNode:
    def __init__(self, s, tok_id, word_id, nxt=None, prev=None):
        self.s = s
        self.tok_id = tok_id
        self.word_id = word_id
        self.nxt = nxt
        self.prev = prev

    def append_node(self, other):
        self.nxt = other
        other.prev = self

    def merge_with_nxt(self, new_tok_id):
        self.s = self.s + self.nxt.s
        self.tok_id = new_tok_id
        self.nxt.tok_id = -1  # Make it invalid
        self.nxt = self.nxt.nxt
        if self.nxt is not None:
            self.nxt.prev = self


def initial_vocab(words):
    fresh_tok = 0
    vocab = {}
    for word in words:
        for char in word:
            if not char in vocab:
                vocab[char] = fresh_tok
                fresh_tok += 1
    return vocab

def tokens_pairs_and_stats(words, vocab, keep_stats=False):
    dummy = TokenNode("", -1, -1)
    node = dummy
    pairs = defaultdict(list)
    if keep_stats:
        stats = MultisetHeap()

    for word_id, word in enumerate(words):
        for c in word:
            tok = TokenNode(c, vocab[c], word_id)
            node.append_node(tok)
            if node.word_id == tok.word_id:
                pairs[(node.tok_id, tok.tok_id)].append(node)
                if keep_stats:
                    stats.add((node.tok_id, tok.tok_id), 1)
            node = tok
    root = dummy.nxt
    root.prev = None

    if keep_stats:
        return root, pairs, stats
    return root, pairs, None

def merge(pair_nodes, pair_to_merge, fresh_token, stats=None):
    for node in pair_nodes[pair_to_merge]:
        if node.nxt is None or (node.tok_id, node.nxt.tok_id) != pair_to_merge:
            continue
        if node.prev is not None and node.prev.word_id == node.word_id:
            pair_nodes[(node.prev.tok_id, fresh_token)].append(node.prev)
            if stats is not None:
                stats.add((node.prev.tok_id, fresh_token), 1)
                if (node.prev.tok_id, node.tok_id) != pair_to_merge:
                    stats.sub((node.prev.tok_id, node.tok_id), 1)
        if node.nxt.nxt is not None and node.nxt.word_id == node.nxt.nxt.word_id:
            pair_nodes[(fresh_token, node.nxt.nxt.tok_id)].append(node)
            if stats is not None:
                stats.add((fresh_token, node.nxt.nxt.tok_id), 1)
                if (node.nxt.tok_id, node.nxt.nxt.tok_id) != pair_to_merge:
                    stats.sub((node.nxt.tok_id, node.nxt.nxt.tok_id), 1)
        node.merge_with_nxt(fresh_token)


def train(words, num_merges, verbose=0):
    vocab = initial_vocab(words)
    reverse_vocab = {vocab[c]: c for c in vocab}
    _root, pairs, stats = tokens_pairs_and_stats(words, vocab, keep_stats=True)

    fresh_token = len(vocab)
    merge_tree = []

    # stats = get_stats(root)
    for i in range(num_merges):
        if verbose >= 1 and i % 100 == 0:
            print(f"Finalized {i} merges")
        try:
            (_count, (left, right)) = stats.popmax()
        except IndexError:
            break
        if verbose == 2:
            print(f"Merging the pair ({reverse_vocab[left]}, {reverse_vocab[right]})")
        merge(pairs, (left, right), fresh_token, stats=stats)

        new_token_s = reverse_vocab[left] + reverse_vocab[right]
        vocab[new_token_s] = fresh_token
        reverse_vocab[fresh_token] = new_token_s

        merge_tree.append(((left, right), fresh_token))
        fresh_token += 1
    return vocab, merge_tree


def encode(vocab, merge_tree, words):
    tokens, pairs, _ = tokens_pairs_and_stats(words, vocab, keep_stats=False)
    for (left, right), new in merge_tree:
        merge(pairs, (left, right), new)
    return tokens


def decode(root):
    words = []
    curr_word = []
    node = root
    while node is not None:
        if node.prev is not None and node.prev.word_id != node.word_id:
            words.append("".join(curr_word))
            curr_word = []
        curr_word.append(node.s)
        node = node.nxt
    words.append("".join(curr_word))
    return words


def test_train():
    words = ["low", "lower", "hard", "harder"]

    vocab, merge_tree = train(words, 6, verbose=2)
    root = encode(vocab, merge_tree, words)

    node = root
    while node is not None:
        print(node.tok_id, end=", ")
        node = node.nxt
    print()
    print("Decoded tokens: ", decode(root))


def test_train_large(interact=False):
    with open("../data/eng_preprocessed.txt", "r", encoding="utf-8") as file:
        txt = file.read()
        eng_words = txt.strip().split()

    print("Starting training")
    vocab, merge_tree = train(eng_words, 10000, verbose=0)
    print("Done training")
    while interact:
        word = input('word: ').strip().lower()
        root = encode(vocab, merge_tree, [word])
        node = root
        tokens = []
        while node is not None:
            tokens.append(node.s)
            node = node.nxt
        print(tokens)


if __name__ == "__main__":
    # test_train()
    test_train_large(interact=True)
