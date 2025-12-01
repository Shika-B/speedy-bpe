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


def get_stats(root):
    stats = MultisetHeap()
    while root.nxt is not None:
        if root.word_id == root.nxt.word_id:
            stats.add((root.tok_id, root.nxt.tok_id), 1)
        root = root.nxt
    return stats


def tokens_and_pairs(words, vocab):
    dummy = TokenNode("", -1, -1)
    node = dummy
    pairs = defaultdict(list)
    for word_id, word in enumerate(words):
        for c in word:
            tok = TokenNode(c, vocab[c], word_id)
            node.append_node(tok)
            if node.word_id == tok.word_id:
                pairs[(node.tok_id, tok.tok_id)].append(node)
            node = tok
    root = dummy.nxt
    root.prev = None
    return root, pairs


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
    root, pairs = tokens_and_pairs(words, vocab)

    fresh_token = len(vocab)
    merge_tree = []

    stats = get_stats(root)
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
    tokens, pairs = tokens_and_pairs(words, vocab)

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
    words = ["aaa"]

    vocab, merge_tree = train(words, 6, verbose=2)
    root = encode(vocab, merge_tree, words)

    node = root
    while node is not None:
        print(node.tok_id, end=", ")
        node = node.nxt
    print()
    print("Decoded tokens: ", decode(root))


def test_train_large():
    with open("data/fra.txt", "r", encoding="utf-8") as file:
        txt = file.read()
        lines = txt.strip().split("\n")
        fra, eng = [], []
        for line in lines:
            cols = line.split("\t")
            eng.append(cols[0])
            fra.append(cols[1])
    regex = re.compile("\\s|\\.|\\!|\\?")

    eng_words = []
    for sentence in eng:
        for word in regex.split(sentence.lower()):
            if len(word) > 0:
                eng_words.append(word)
    print("Starting training")
    train(eng_words, 10000, verbose=1)
    print("Done training")


if __name__ == "__main__":
    # test_train()
    test_train_large()
