Some experiments about byte pair encoding in the context of large language models. 

## Description

See the associated [blog post](https://shika-b.github.io/blog/byte_pair_encoding)

## Benchmarks
Training a BPE on 130 000 [english sentences](data/fra.txt) with a vocabulary size of approximately 10 000 tokens takes the following times on my laptop (yeah I left the Python code run that long):

| Code                       |   Time    |
|:---------------------------|:---------:|
| Python Naive               | 4h48m     |
| Rust Naive + small strings | 5m40s     |
| Python Optimized           | 17s       |
