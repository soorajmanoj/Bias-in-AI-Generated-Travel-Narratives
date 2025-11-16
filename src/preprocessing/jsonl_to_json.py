import json

input_file = "../../data/clean/filtered/relevant.jsonl"
output_file = "../../data/clean/filtered/merged_output.json"

rom_hindi_list = []
english_list = []

with open(input_file, "r") as f:
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

# Final JSON
final_output = {
    "rom_hindi": rom_hindi_list,
    "english": english_list
}

with open(output_file, "w") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

print("✅ Done! Saved to", output_file)
print(f"{len(rom_hindi_list)} romanized Hindi comments")
print(f"{len(english_list)} English comments")
