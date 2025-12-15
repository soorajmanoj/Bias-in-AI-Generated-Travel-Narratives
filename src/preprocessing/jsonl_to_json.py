import json

"""
@file jsonl_to_json.py
@brief Convert a .jsonl file of comment objects into a merged JSON list separated by language.

Reads `relevant.jsonl`, extracts `comment` values grouped by `language`, and writes
the consolidated JSON file `merged_output.json` with `rom_hindi` and `english` lists.
"""

input_file = "../../data/clean/filtered/relevant.jsonl"
output_file = "../../data/clean/filtered/merged_output.json"

rom_hindi_list = []
english_list = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        comment = obj.get("comment", "")
        lang = obj.get("language", "").lower()

        if lang == "rom_hindi":
            rom_hindi_list.append(comment)
        elif lang == "english":
            english_list.append(comment)

final_output = {
    "rom_hindi": rom_hindi_list,
    "english": english_list
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print("✅ Done! Saved to", output_file)
print(f"{len(rom_hindi_list)} romanized Hindi comments")
print(f"{len(english_list)} English comments")
