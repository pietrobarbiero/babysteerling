from tokenizers import ByteLevelBPETokenizer
from download_tinystories import path

vocab_size = 2000
save_path = "./data/tinystories_tokenizer.json"

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[path],
    vocab_size=vocab_size,
    min_frequency=2,
    special_tokens=["<|endoftext|>"],
)
tokenizer.save(save_path)
print(
    f"Trained tokenizer with vocab_size={tokenizer.get_vocab_size()}, saved to {save_path}"
)
