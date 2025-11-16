from pathlib import Path
import json
import math
import sys

CHUNK_SIZE = 3

root = Path(__file__).resolve().parent.parent.parent.parent
input_path = root / 'data' / 'raw' / 'youtube_data.json'
output_dir = root / 'data' / 'raw'

if not input_path.exists():
    print(f"Input file not found: {input_path}")
    sys.exit(1)


with input_path.open('r', encoding='utf-8') as f:
    data = json.load(f)

if not isinstance(data, list):
    print(f"Unexpected format: expected top-level JSON array, got {type(data)}")
    sys.exit(1)

count = len(data)
if count == 0:
    print("No video objects found in input file.")
    sys.exit(0)

num_parts = math.ceil(count / CHUNK_SIZE)
created_files = []
for i in range(num_parts):
    start = i * CHUNK_SIZE
    end = start + CHUNK_SIZE
    chunk = data[start:end]
    out_name = output_dir / f'youtube_data_part_{i+1}.json'
    with out_name.open('w', encoding='utf-8') as out_f:
        json.dump(chunk, out_f, ensure_ascii=False, indent=2)
    created_files.append(str(out_name))

print(f"Split {count} videos into {num_parts} files (chunk size {CHUNK_SIZE}).")
for p in created_files:
    print(p)

sys.exit(0)
