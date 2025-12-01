use compact_str::{CompactString, ToCompactString};
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Token {
    slice: CompactString,
    tok_id: usize,
    word_id: usize,
}

type Vocab = HashMap<CompactString, usize>;
type MergeTree = Vec<((usize, usize), usize)>;

fn build_initial_vocab(words: &[&str]) -> Vocab {
    let mut vocab = Vocab::new();
    let mut fresh_token_id = 0;
    for word in words {
        for char in word.chars() {
            let c = char.to_compact_string();
            if !vocab.contains_key(&c) {
                vocab.insert(c, fresh_token_id);
                fresh_token_id += 1;
            }
        }
    }
    vocab
}

fn get_stats(tokens: &Vec<Token>) -> HashMap<(usize, usize), usize> {
    let mut map = HashMap::new();
    for (tok1, tok2) in tokens.iter().zip(tokens.iter().skip(1)) {
        if tok1.word_id == tok2.word_id {
            *map.entry((tok1.tok_id, tok2.tok_id)).or_insert(0) += 1;
        }
    }
    map
}

fn merge(mut tokens: Vec<Token>, pair: (usize, usize), fresh: usize) -> Vec<Token> {
    let mut new_tokens = Vec::with_capacity(tokens.len());

    let mut i = 0;
    while i < tokens.len() - 1 {
        let (tok1, tok2) = (&tokens[i], &tokens[i + 1]);
        if tok1.word_id == tok2.word_id && (tok1.tok_id, tok2.tok_id) == pair {
            new_tokens.push(Token {
                slice: tok1.slice.clone() + &tok2.slice,
                tok_id: fresh,
                word_id: tok1.word_id,
            });
            i += 2;
        } else {
            new_tokens.push(tok1.clone());
            i += 1;
        }
    }
    if i == tokens.len() - 1 {
        new_tokens.push(tokens.pop().unwrap());
    }
    new_tokens
}

fn _words_to_tokens(words: &[&str], vocab: &Vocab) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(words.len() * 10);

    for (word_id, word) in words.into_iter().enumerate() {
        for c in word.chars() {
            let c = c.to_compact_string();
            tokens.push(Token {
                tok_id: *vocab.get(&c).unwrap(),
                slice: c,
                word_id,
            });
        }
    }
    tokens
}

fn train(words: &[&str], num_merges: usize, verbose: bool) -> (Vocab, MergeTree) {
    let mut vocab = build_initial_vocab(&words);
    let mut reverse_vocab = vocab
        .clone()
        .into_iter()
        .map(|(k, v)| (v, k))
        .collect::<HashMap<usize, CompactString>>();
    let mut tokens = _words_to_tokens(words, &vocab);
    let mut fresh_token_id = vocab.len();
    let mut merge_tree = vec![];

    for _ in 0..num_merges {
        let stats = get_stats(&tokens);
        if stats.len() == 0 {
            break;
        }
        let (left_id, right_id) = *stats
            .iter()
            .max_by(|(_, c1), (_, c2)| c1.cmp(c2))
            .unwrap()
            .0;
        let (left_s, right_s) = (
            reverse_vocab.get(&left_id).unwrap(),
            reverse_vocab.get(&right_id).unwrap(),
        );
        if verbose {
            eprintln!("Merging the pair ({}, {})", left_s, right_s);
        }
        tokens = merge(tokens, (left_id, right_id), fresh_token_id);
        let new_token_s = left_s.clone() + right_s;
        vocab.insert(new_token_s.clone(), fresh_token_id);
        reverse_vocab.insert(fresh_token_id, new_token_s);

        merge_tree.push(((left_id, right_id), fresh_token_id));
        fresh_token_id += 1;
    }

    (vocab, merge_tree)
}

fn encode(words: &[&str], vocab: &Vocab, merge_tree: &MergeTree) -> Vec<Token> {
    let mut tokens = _words_to_tokens(words, vocab);

    for ((left, right), merged) in merge_tree {
        tokens = merge(tokens, (*left, *right), *merged);
    }
    tokens
}

fn decode(tokens: &[Token]) -> Vec<CompactString> {
    let mut words = vec![];
    let mut word = CompactString::const_new("");

    let mut last_word_id = 0;

    for (idx, token) in tokens.iter().enumerate() {
        if idx > 0 && last_word_id != token.word_id {
            words.push(word);
            word = CompactString::const_new("");
        }
        last_word_id = token.word_id;
        word += &token.slice;
    }
    words.push(word);
    words
}
fn test_train() {
    let words = ["low", "lower", "hard", "harder"];

    let (vocab, merge_tree) = train(&words, 6, true);
    let tokens = encode(&words, &vocab, &merge_tree);
    println!(
        "Tokens: {:?}",
        tokens.iter().map(|tok| tok.tok_id).collect::<Vec<usize>>()
    );
    println!("Decoded tokens: {:?}", decode(&tokens));
}

fn test_train_large() {
    use std::fs::File;
    use std::io::prelude::*;

    let mut file = File::open("../data/fra.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let lines = contents.trim().split('\n');
    let (mut fra, mut eng) = (vec![], vec![]);
    for line in lines {
        let cols = line.split('\t').collect::<Vec<&str>>();
        eng.push(cols[0].to_lowercase());
        fra.push(cols[1].to_lowercase());
    }
    let regex = Regex::new("\\s|\\.|\\!|\\?").unwrap();

    let mut eng_words = Vec::new();
    for sentence in eng.iter() {
        for word in regex.split(sentence) {
            if word.len() > 0 {
                eng_words.push(word)
            }
        }
    }
    train(&eng_words, 10_000, true);
}

fn main() {
    test_train_large();
}
