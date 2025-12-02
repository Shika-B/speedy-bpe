from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=10000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train(["../data/eng_preprocessed.txt"], trainer)


# encoded = tokenizer.encode('anticonstitutionnally')
# print(encoded.tokens)
tokenizer.save("bpe_tokenizer.json")