import json

# Input and output paths
input_path = "../../data/clean/API_cleaned_data_full.json"
output_path = "../../data/clean/sorted/combined_sorted_comments.json"

# Load the data (list of videos)
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize lists
rom_hindi_comments = []
english_comments = []

# Iterate through each video
for video in data:
    # Extend comments if present
    if "rom_hindi" in video:
        rom_hindi_comments.extend(video["rom_hindi"])
    if "english" in video:
        english_comments.extend(video["english"])

    # Remove "other" key if it exists
    video.pop("other", None)

# Clean function — removes empty or whitespace-only entries
def clean_comments(comments):
    return sorted(set(c.strip() for c in comments if c and c.strip()))

# Clean and sort
rom_hindi_comments = clean_comments(rom_hindi_comments)
english_comments = clean_comments(english_comments)

# Final JSON structure
output_data = [{
    "rom_hindi": rom_hindi_comments,
    "english": english_comments
}]

# Save cleaned, sorted output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print(f"Cleaned, deduplicated, and sorted comments saved to {output_path}")