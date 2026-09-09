import argparse
import json

from transformers import AutoTokenizer

from main.common import get_base_dir


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="JSONL dataset filename under data/")
parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer directory under tokenizers/")
parser.add_argument("--batch-size", type=int, default=1024) # Number of documents to process before writing to parquet (default: 1024)
args = parser.parse_args()

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as e:
    raise SystemExit("pyarrow is required to write parquet files. Install it with: uv add pyarrow") from e

root = get_base_dir()
data_path = root / "data" / args.dataset
tokenizer_path = root / "tokenizers" / args.tokenizer
out_dir = root / "data" / "tokenized"
out_path = out_dir / f"{data_path.stem}.parquet"

out_dir.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
total_docs = sum(1 for line in data_path.open(encoding="utf-8") if line.strip())

schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
writer = pq.ParquetWriter(out_path, schema)

rows = []
docs = 0
try:
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            rows.append(tokenizer.encode(text, add_special_tokens=False))
            docs += 1

            if len(rows) == args.batch_size:
                writer.write_table(pa.table({"input_ids": rows}, schema=schema))
                rows.clear()
                print(f"Tokenized {docs}/{total_docs} documents", flush=True)

    if rows:
        writer.write_table(pa.table({"input_ids": rows}, schema=schema))
finally:
    writer.close()

print(f"Tokenized {docs} documents to {out_path}")
