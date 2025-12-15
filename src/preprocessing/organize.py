import json

"""
@file organize.py
@brief Load cleaned API output, aggregate comments by language, deduplicate and sort,
and write a combined JSON of `rom_hindi` and `english` comments.
"""

input_path = "../../data/clean/API_cleaned_data_full.json"
output_path = "../../data/clean/sorted/combined_sorted_comments.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rom_hindi_comments = []
english_comments = []

for video in data:
    if "rom_hindi" in video:
        rom_hindi_comments.extend(video["rom_hindi"])
    if "english" in video:
        english_comments.extend(video["english"])

    video.pop("other", None)


def clean_comments(comments):
    """
    @brief Remove empty entries and return sorted unique list.

    @param comments Iterable of comment strings.
    @return Sorted list of unique, stripped comments.
    """
    return sorted(set(c.strip() for c in comments if c and c.strip()))


rom_hindi_comments = clean_comments(rom_hindi_comments)
english_comments = clean_comments(english_comments)

output_data = [{
    "rom_hindi": rom_hindi_comments,
    "english": english_comments
}]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Cleaned, deduplicated, and sorted comments saved to {output_path}")