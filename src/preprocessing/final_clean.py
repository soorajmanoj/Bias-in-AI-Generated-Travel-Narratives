import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Optional: expand with common informal/YouTube-specific additions
extra_stopwords = {
    "im", "ive", "dont", "didnt", "doesnt", "youre", "wasnt", "isnt", "couldnt",
    "u", "ur", "ya", "thats", "wanna", "gonna", "wouldnt", "cant", "aint", "also"
}
stop_words |= extra_stopwords

def clean_for_topic_modeling(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = text.encode("ascii", "ignore").decode()  # Remove emojis/non-ASCII
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    tokens = re.findall(r"\b\w+\b", text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

# Load
df = pd.read_csv("../../data/processed/clean/open_coding_sample.csv")

# Clean
df["comment_clean"] = df["comment"].astype(str).apply(clean_for_topic_modeling)

# Save
df.to_csv("../../data/processed/clean/open_coding_final.csv", index=False)
print("✅ Cleaned file saved to: open_coding_final.csv")
