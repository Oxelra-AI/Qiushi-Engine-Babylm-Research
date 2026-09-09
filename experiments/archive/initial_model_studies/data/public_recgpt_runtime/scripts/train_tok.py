import argparse
import json

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from main.common import get_base_dir

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset filename (Has to be located under data directory)')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument("--vocab-size", type=int, default=32768)
args = parser.parse_args()

root = get_base_dir()
data_path = root / "data" / args.dataset
out_path = root / "tokenizers" / f"{data_path.stem}-bpe"

# Extract JSONL text fields and stop after max_chars total characters.
def texts():
    total = 0
    for line in data_path.open(encoding="utf-8"):
        text = json.loads(line)["text"][: args.max_chars - total] # At each document limit by remaining char limit
        if not text:
            break
        total += len(text)
        yield text

tok = Tokenizer(BPE())
tok.pre_tokenizer = ByteLevel()
tok.decoder = ByteLevelDecoder()

tok.train_from_iterator(
    texts(),
    trainer=BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>"],
        initial_alphabet=ByteLevel.alphabet(),
    ),
)

out_path.mkdir(parents=True, exist_ok=True)
PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>").save_pretrained(out_path)
print(f"Saved tokenizer to {out_path}")
